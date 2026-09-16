"""Every diagnostic `read_weights` promises for a bad `.rwt` file.

The reader's error paths were the last uncovered block in the package (78%),
and they are the same kind of promise `validate.py` makes: a file that is
truncated, from another plan, or written by another version must be *named*,
not silently half-read into weights that then compute plausible nonsense. The
existing tests cover two truncations; these cover the rest.

Each case takes a real file and damages exactly one thing, so the diagnostic
under test is the one that fires rather than an earlier check catching it
first. The offsets come from the format in weights.py:

    0   MAGIC (8)
    8   version, dtype_code, endian, count (4x4)
    24  plan hash (32)
    56  TOC length (4)
    60  TOC, then the body
"""
import struct

import numpy as np
import pytest
from onnx import helper, numpy_helper, TensorProto

from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from rosenna.weights import MAGIC, VERSION, read_weights, write_weights


@pytest.fixture
def rwt(tmp_path):
    """A valid two-weight file, plus its bytes."""
    rng = np.random.default_rng(3)
    w = numpy_helper.from_array(rng.uniform(-1, 1, (3, 2)).astype(np.float32), "w")
    b = numpy_helper.from_array(rng.uniform(-1, 1, (2,)).astype(np.float32), "b")
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="g0")], "wt",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])], [w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx_path = tmp_path / "wt.onnx"
    import onnx
    onnx.save(model, str(onnx_path))
    g = load_graph(onnx_path)
    plan = build_plan(g, dtype="f32", embed=False)
    path = tmp_path / "wt.rwt"
    write_weights(plan, g, path)
    return path, path.read_bytes()


def _expect(path, blob, fragment):
    path.write_bytes(blob)
    with pytest.raises(ValueError, match=fragment):
        read_weights(path)


def test_a_valid_file_reads_back(rwt):
    path, _ = rwt
    tensors, header = read_weights(path)
    assert header["version"] == VERSION and header["count"] == 2
    assert set(tensors) == {"w", "b"}, "tensors is a dict keyed by name"


def test_a_file_that_is_not_a_weights_file_is_named(rwt):
    path, blob = rwt
    _expect(path, b"NOTROSEN" + blob[8:], "not a roseNNa weights file")


def test_a_header_shorter_than_24_bytes(rwt):
    path, blob = rwt
    _expect(path, blob[:20], "header requires 24 bytes")


def test_a_version_from_another_release(rwt):
    path, blob = rwt
    _expect(path, blob[:8] + struct.pack("<i", VERSION + 7) + blob[12:], "version 8")


def test_a_file_written_on_the_other_endian(rwt):
    path, blob = rwt
    _expect(path, blob[:16] + struct.pack("<i", 0x04030201) + blob[20:], "endian marker")


def test_an_unknown_dtype_code(rwt):
    path, blob = rwt
    _expect(path, blob[:12] + struct.pack("<i", 99) + blob[16:], "unknown dtype code 99")


def test_truncated_before_the_plan_hash_and_toc_length(rwt):
    path, blob = rwt
    _expect(path, blob[:40], "plan hash and TOC length require 60 bytes")


def test_a_table_of_contents_that_runs_past_the_end(rwt):
    path, blob = rwt
    _expect(path, blob[:56] + struct.pack("<i", 10_000) + blob[60:], "table of contents")


@pytest.mark.parametrize("cut,fragment", [
    (62, "reading name length"),
    (64, "reading name"),
    (65, "reading rank"),
    (69, "dimensions"),
    (85, "reading offset and length"),
])
def test_each_stage_of_a_tensor_record_names_where_it_ran_out(rwt, cut, fragment):
    """Every truncation point inside one TOC record, named separately.

    The per-tensor reads are bounded by the file length, while the TOC check
    just above them uses 60 + toclen -- so reaching them needs a *small*
    declared toclen as well as a short file, or the TOC check fires first and
    none of these five is ever exercised. Setting toclen to exactly the bytes
    that remain is what walks the reader into each stage in turn.
    """
    path, blob = rwt
    forged = blob[:56] + struct.pack("<i", cut - 60) + blob[60:cut]
    _expect(path, forged, fragment)


def test_a_malformed_rank_is_wrapped_rather_than_escaping_as_struct_error(rwt):
    """The reader's own catch: struct's error is re-raised as a named ValueError.

    A negative rank is not a length problem, so it passes the bounds checks and
    blows up inside struct.unpack. Without the wrapper the caller would see
    `struct.error: bad char in struct format`, naming neither the file nor the
    tensor.
    """
    path, blob = rwt
    toclen = struct.unpack("<i", blob[56:60])[0]
    toc = bytearray(blob[60:60 + toclen])
    namelen = struct.unpack("<i", toc[0:4])[0]
    toc[4 + namelen:8 + namelen] = struct.pack("<i", -1)      # rank = -1
    _expect(path, blob[:60] + bytes(toc) + blob[60 + toclen:], "tensor 0:")


def test_a_tensor_whose_data_extends_past_the_end(rwt):
    path, blob = rwt
    toclen = struct.unpack("<i", blob[56:60])[0]
    toc = bytearray(blob[60:60 + toclen])
    # The first tensor's record is: namelen, name, rank, dims..., offset, length.
    namelen = struct.unpack("<i", toc[0:4])[0]
    pos = 4 + namelen
    rank = struct.unpack("<i", toc[pos:pos + 4])[0]
    tail = pos + 4 + 8 * rank
    offset, length = struct.unpack("<qq", toc[tail:tail + 16])
    toc[tail:tail + 16] = struct.pack("<qq", offset, length + 10_000)
    _expect(path, blob[:60] + bytes(toc) + blob[60 + toclen:], "extends to byte")


def test_write_refuses_an_array_the_plan_does_not_expect(tmp_path):
    """The writer's own guard: a plan and a graph that disagree on a size."""
    rng = np.random.default_rng(4)
    w = numpy_helper.from_array(rng.uniform(-1, 1, (3, 2)).astype(np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"], name="m0")], "wt",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    import onnx
    path = tmp_path / "wt.onnx"
    onnx.save(model, str(path))
    g = load_graph(path)
    plan = build_plan(g, dtype="f32", embed=False)
    # Swap in an array of a different size than the plan recorded.
    g.initializers["w"] = np.zeros((5, 5), np.float32)
    with pytest.raises(ValueError, match="plan says .* bytes, array has"):
        write_weights(plan, g, tmp_path / "bad.rwt")
