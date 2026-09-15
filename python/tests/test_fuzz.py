"""Random graphs of supported ops, verified against onnxruntime.

The golden set is 21 models a person wrote, so it tests the shapes a person
thought of. The bugs this branch actually shipped were not those: an
activation bounded by the *previous* op's output length (only wrong on a
branching graph), a `Gemm` after an LSTM sequence (only wrong when the leading
axis is not 1), an LSTM output list compacted out of its positional slots
(only wrong when the model reads Y_c and not Y_h). Each needed a graph nobody
had drawn.

So this draws them. Every case is a seed: a failure prints the seed, and
`_model_for_seed` rebuilds exactly that graph for debugging. SEEDS is a fixed
list rather than a fresh draw per run, so CI is deterministic and a failure is
reproducible from the parameter id alone; widen it to hunt.
"""
import re

import numpy as np
import onnx
import pytest
from onnx import helper, numpy_helper, TensorProto

from rosenna.verify import VerificationError, verify_model

# Fixed, not random per run: a fuzzer whose corpus changes every run reports a
# failure nobody can reproduce, and turns an unrelated CI run red.
SEEDS = list(range(48))

_ACTS = ["Relu", "Tanh", "Sigmoid"]


def _dense_chain(rng, nodes, inits, cur, width, n_gemm):
    """A few dense layers with activations, returning the new value and width."""
    for _ in range(int(rng.integers(1, 4))):
        out_w = int(rng.integers(1, 9))
        w = numpy_helper.from_array(
            rng.uniform(-1, 1, (width, out_w)).astype(np.float32), f"w{n_gemm[0]}")
        inits.append(w)
        nxt = f"v{len(nodes)}"
        if rng.random() < 0.5:
            b = numpy_helper.from_array(
                rng.uniform(-1, 1, (out_w,)).astype(np.float32), f"b{n_gemm[0]}")
            inits.append(b)
            nodes.append(helper.make_node("Gemm", [cur, w.name, b.name], [nxt],
                                          name=f"g{len(nodes)}"))
        else:
            nodes.append(helper.make_node("MatMul", [cur, w.name], [nxt], name=f"m{len(nodes)}"))
        n_gemm[0] += 1
        cur, width = nxt, out_w
        if rng.random() < 0.35:
            # Add a constant, broadcast from one of the shapes that reaches the
            # per-axis stride arithmetic: a full row, a single value, or a
            # rank-1 vector that has to be right-aligned against a rank-2 value.
            shp = [(1, width), (1, 1), (width,)][int(rng.integers(0, 3))]
            c = numpy_helper.from_array(
                rng.uniform(-1, 1, shp).astype(np.float32), f"ad{n_gemm[0]}_{len(nodes)}")
            inits.append(c)
            nxt = f"v{len(nodes)}"
            nodes.append(helper.make_node("Add", [cur, c.name], [nxt], name=f"ad{len(nodes)}"))
            cur = nxt
        if rng.random() < 0.7:
            act = _ACTS[int(rng.integers(0, len(_ACTS)))]
            nxt = f"v{len(nodes)}"
            nodes.append(helper.make_node(act, [cur], [nxt], name=f"a{len(nodes)}"))
            cur = nxt
    return cur, width


def _spatial_chain(rng, nodes, inits, cur, c, h, w, n_conv):
    """Conv / MaxPool / AveragePool until the field is too small to shrink."""
    for _ in range(int(rng.integers(1, 4))):
        if h < 3 or w < 3:
            break
        pick = rng.random()
        if pick < 0.5:
            oc, k = int(rng.integers(1, 5)), int(rng.integers(1, 4))
            pad = int(rng.integers(0, 2))
            weight = numpy_helper.from_array(
                rng.uniform(-1, 1, (oc, c, k, k)).astype(np.float32), f"cw{n_conv[0]}")
            inits.append(weight)
            args = [cur, weight.name]
            if rng.random() < 0.5:
                bias = numpy_helper.from_array(
                    rng.uniform(-1, 1, (oc,)).astype(np.float32), f"cb{n_conv[0]}")
                inits.append(bias)
                args.append(bias.name)
            nxt = f"v{len(nodes)}"
            nodes.append(helper.make_node("Conv", args, [nxt], name=f"c{len(nodes)}",
                                          kernel_shape=[k, k], pads=[pad] * 4,
                                          strides=[1, 1], group=1))
            h, w, c = h + 2 * pad - k + 1, w + 2 * pad - k + 1, oc
            n_conv[0] += 1
            cur = nxt
            if rng.random() < 0.35:
                # A per-channel bias: (C,1,1) against (1,C,H,W) is the mnist
                # shape, and the one whose strides are (0, 1, 0, 0).
                cb = numpy_helper.from_array(
                    rng.uniform(-1, 1, (c, 1, 1)).astype(np.float32), f"pc{n_conv[0]}")
                inits.append(cb)
                nxt = f"v{len(nodes)}"
                nodes.append(helper.make_node("Add", [cur, cb.name], [nxt], name=f"pa{len(nodes)}"))
                cur = nxt
        else:
            k = int(rng.integers(2, 4))
            st = int(rng.integers(1, k + 1))
            op = "MaxPool" if rng.random() < 0.5 else "AveragePool"
            if (h - k) // st + 1 < 1 or (w - k) // st + 1 < 1:
                break
            nxt = f"v{len(nodes)}"
            nodes.append(helper.make_node(op, [cur], [nxt], name=f"p{len(nodes)}",
                                          kernel_shape=[k, k], strides=[st, st],
                                          pads=[0, 0, 0, 0]))
            h, w = (h - k) // st + 1, (w - k) // st + 1
            cur = nxt
        if rng.random() < 0.5:
            act = _ACTS[int(rng.integers(0, len(_ACTS)))]
            nxt = f"v{len(nodes)}"
            nodes.append(helper.make_node(act, [cur], [nxt], name=f"a{len(nodes)}"))
            cur = nxt
    return cur, c, h, w


def _lstm_head(rng, nodes, inits, graph_in):
    """An LSTM over a random sequence, reduced to a rank-2 value for the tail.

    ONNX wants X as (seq, batch, input) and hands back Y (seq, 1, batch, H),
    Y_h and Y_c (1, batch, H). batch is 1, as in every golden LSTM, so the
    Squeeze/Transpose a PyTorch export emits are the ones plan.py folds into
    buffer aliases. Which of the three outputs the tail reads is drawn, because
    an unread one has to be dropped without shifting the others out of their
    positional slots -- that shipped as a bug.
    """
    seq = int(rng.integers(2, 5))
    inp = int(rng.integers(2, 5))
    hid = int(rng.integers(2, 5))
    graph_in.append(helper.make_tensor_value_info("x", TensorProto.FLOAT, [seq, 1, inp]))
    W = numpy_helper.from_array(rng.uniform(-1, 1, (1, 4 * hid, inp)).astype(np.float32), "lw")
    R = numpy_helper.from_array(rng.uniform(-1, 1, (1, 4 * hid, hid)).astype(np.float32), "lr")
    args = ["x", "lw", "lr"]
    inits += [W, R]
    if rng.random() < 0.6:
        B = numpy_helper.from_array(rng.uniform(-1, 1, (1, 8 * hid)).astype(np.float32), "lb")
        inits.append(B)
        args.append("lb")
    outs = ["Y", "Y_h", "Y_c"]
    nodes.append(helper.make_node("LSTM", args, outs, name="lstm0", hidden_size=hid))
    pick = int(rng.integers(0, 3))
    if pick == 0:                      # Y: (seq,1,1,H) -> squeeze to (seq,H)
        sq_axes = numpy_helper.from_array(np.array([1, 2], np.int64), "sq_axes")
        inits.append(sq_axes)
        nodes.append(helper.make_node("Squeeze", ["Y", "sq_axes"], ["yflat"], name="sq0"))
        return "yflat", hid, seq
    # Y_h or Y_c: (1,1,H) -> squeeze the leading axis to (1,H)
    src = outs[1 + (pick - 1)]
    ax = numpy_helper.from_array(np.array([0], np.int64), f"sq_{src}")
    inits.append(ax)
    nodes.append(helper.make_node("Squeeze", [src, ax.name], ["yflat"], name="sq1"))
    return "yflat", hid, 1


def _model_for_seed(seed: int, path):
    """Build one random supported model. Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    nodes, inits = [], []
    n_gemm = [0]
    kind = rng.random()
    if kind < 0.25:
        graph_in = []
        cur, width, rows = _lstm_head(rng, nodes, inits, graph_in)
        cur, width = _dense_chain(rng, nodes, inits, cur, width, n_gemm)
        graph = helper.make_graph(
            nodes, f"fuzz{seed}", graph_in,
            [helper.make_tensor_value_info(cur, TensorProto.FLOAT, [rows, width])], inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
        return model
    spatial = kind < 0.6
    if spatial:
        c, h, w = int(rng.integers(1, 3)), int(rng.integers(6, 13)), int(rng.integers(6, 13))
        in_shape = [1, c, h, w]
        cur, c, h, w = _spatial_chain(rng, nodes, inits, "x", c, h, w, [0])
        width = c * h * w
        # ONNX Gemm/MatMul take rank 2, so the field is flattened first -- the
        # same Reshape mnist has, and the one plan.py turns into a buffer alias.
        shape = numpy_helper.from_array(np.array([1, width], np.int64), "flat_shape")
        inits.append(shape)
        nxt = f"v{len(nodes)}"
        nodes.append(helper.make_node("Reshape", [cur, shape.name], [nxt], name=f"r{len(nodes)}"))
        cur = nxt
    else:
        width = int(rng.integers(2, 7))
        in_shape = [1, width]
        cur = "x"
    cur, width = _dense_chain(rng, nodes, inits, cur, width, n_gemm)
    # A branch: one value feeding two chains that are then concatenated. This is
    # the shape the buffer planner is least safe on -- liveness has to keep the
    # shared value alive across both arms, and an activation's loop bound has to
    # come from its own value rather than from whatever op ran last. Both of
    # those shipped as bugs, and neither is expressible in the golden set.
    if rng.random() < 0.5:
        left, lw = _dense_chain(rng, nodes, inits, cur, width, n_gemm)
        right, rw = _dense_chain(rng, nodes, inits, cur, width, n_gemm)
        nxt = f"v{len(nodes)}"
        nodes.append(helper.make_node("Concat", [left, right], [nxt],
                                      name=f"cat{len(nodes)}", axis=1))
        cur, width = nxt, lw + rw
        cur, width = _dense_chain(rng, nodes, inits, cur, width, n_gemm)
    # A second graph input, consumed by its own chain and merged in. The
    # generated entry point takes every input concatenated in x, in declaration
    # order, so this exercises the slice-copy that hands each secondary input
    # its part -- and the offsets, which nothing else here varies.
    graph_in = [helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)]
    if rng.random() < 0.4:
        w2 = int(rng.integers(2, 6))
        graph_in.append(helper.make_tensor_value_info("x2", TensorProto.FLOAT, [1, w2]))
        side, sw = _dense_chain(rng, nodes, inits, "x2", w2, n_gemm)
        nxt = f"v{len(nodes)}"
        nodes.append(helper.make_node("Concat", [cur, side], [nxt],
                                      name=f"jin{len(nodes)}", axis=1))
        cur, width = nxt, width + sw
        cur, width = _dense_chain(rng, nodes, inits, cur, width, n_gemm)

    # A second graph output, which leaves concatenated in y.
    graph_out = [helper.make_tensor_value_info(cur, TensorProto.FLOAT, [1, width])]
    if rng.random() < 0.4:
        tail, tw = _dense_chain(rng, nodes, inits, cur, width, n_gemm)
        if tail != cur:
            graph_out.append(helper.make_tensor_value_info(tail, TensorProto.FLOAT, [1, tw]))

    graph = helper.make_graph(nodes, f"fuzz{seed}", graph_in, graph_out, inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return model


@pytest.mark.parametrize("seed", SEEDS)
def test_random_supported_graph_matches_onnxruntime(seed, tmp_path):
    path = tmp_path / f"fuzz{seed}.onnx"
    _model_for_seed(seed, path)
    try:
        results = verify_model(path, "both", None, 8, tmp_path, name=f"fuzz{seed}")
    except VerificationError as e:
        # A random net whose last layer is a ReLU is sometimes dead -- every
        # output zero for every input. verify refuses to compare against that,
        # rightly: reproducing an all-zero reference demonstrates nothing. It
        # is a property of the draw, not of the code under test.
        pytest.skip(f"seed {seed} drew a dead network: {e}")
    for r in results:
        assert r.ok, (f"seed {seed} / {r.lang}: max_abs={r.max_abs:.3e} "
                      f"max_rel={r.max_rel:.3e}; rebuild with "
                      f"tests.test_fuzz._model_for_seed({seed}, path)")


def test_the_corpus_is_reproducible(tmp_path):
    """The same seed builds the same bytes, so a reported failure can be rebuilt."""
    a, b = tmp_path / "a.onnx", tmp_path / "b.onnx"
    _model_for_seed(7, a)
    _model_for_seed(7, b)
    assert a.read_bytes() == b.read_bytes()


def test_the_corpus_covers_every_shape_it_claims_to(tmp_path):
    """The pinned seeds actually draw each construction this file is for.

    A fuzzer that quietly stops drawing LSTMs, or branches, looks exactly like
    one that is finding nothing. This asserts the corpus still reaches each
    construction, so a change to the generator that narrows it fails here
    rather than silently reducing what the other tests cover.
    """
    seen = {k: 0 for k in ("lstm", "spatial", "branch", "add", "multi_in", "multi_out")}
    for seed in SEEDS:
        model = _model_for_seed(seed, tmp_path / f"c{seed}.onnx")
        ops = {n.op_type for n in model.graph.node}
        seen["lstm"] += "LSTM" in ops
        seen["spatial"] += bool(ops & {"Conv", "MaxPool", "AveragePool"})
        seen["branch"] += "Concat" in ops
        seen["add"] += "Add" in ops
        seen["multi_in"] += len(model.graph.input) > 1
        seen["multi_out"] += len(model.graph.output) > 1
    thin = {k: v for k, v in seen.items() if v < 3}
    assert not thin, f"corpus covers these too thinly: {thin} (full tally {seen})"
