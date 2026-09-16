from rosenna.frontend import load_graph
from rosenna.plan import build_plan, EMBED_THRESHOLD
from rosenna.emit_c import emit_c


def test_small_model_embeds_by_default(golden_model):
    plan = build_plan(load_graph(golden_model("gemm_small")), dtype="f64")
    assert plan.embed is True and plan.n_params < EMBED_THRESHOLD
    source, header = emit_c(plan)
    assert "ROSENNA_CONST double w0[4] = {" not in header  # symbols carry the model prefix (ruling R1)
    assert "ROSENNA_CONST double gemm_small_w0[4] = {" in header
    assert "_init(" not in header and "fopen" not in source


def test_no_embed_flag_keeps_the_file_path(golden_model):
    plan = build_plan(load_graph(golden_model("gemm_small")), dtype="f64", embed=False)
    source, header = emit_c(plan)
    assert "_init(" in header and "fopen" in source


def test_embed_changes_the_hash(golden_model):
    g = load_graph(golden_model("gemm_small"))
    assert build_plan(g, dtype="f64", embed=True).hash() != build_plan(g, dtype="f64", embed=False).hash()


def test_embedded_and_file_loaded_agree_exactly(tmp_path, golden_model):
    # Embedding prints every weight at full precision; the two builds must produce identical bits.
    from tests.test_device_c import _build_and_run, _omp_cc
    import numpy as np
    name = "gemm_big"; graph = load_graph(golden_model(name))
    inputs = np.random.default_rng(11).uniform(-2, 2, (8, build_plan(graph).input.shape[0]))
    outs = []
    for embed in (True, False):
        d = tmp_path / ("e" if embed else "f"); d.mkdir()
        r = _build_and_run(d, name, build_plan(graph, dtype="f64", embed=embed), graph, _omp_cc(),
                           ["-O2", "-std=c11", "-fopenmp"], inputs)
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout)
    assert outs[0] == outs[1]
