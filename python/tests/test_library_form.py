import re
import shutil
import subprocess
import numpy as np
import onnxruntime as ort
import pytest
from rosenna.frontend import load_graph
from rosenna.plan import build_plan
from rosenna.weights import write_weights
from rosenna.emit_c import emit_c, emit_c_recipe
from tests.test_emit_fortran import _live_reference


def _cc():
    for cand in ("gcc-15", "gcc-14", "gcc-13", "gcc"):
        if shutil.which(cand):
            return cand
    pytest.skip("no C compiler found")


def test_header_defines_inline_infer_and_source_does_not(golden_model):
    # embed=False: this test is specifically about the file-loaded contract
    # (extern declaration in the header, definition in the source).
    plan = build_plan(load_graph(golden_model("gemm_small")), dtype="f64", embed=False)
    source, header = emit_c(plan)
    assert "static inline ROSENNA_DEVICE_FN void gemm_small_infer(" in header
    # Structural check (controller ruling P1): infer must be defined only in
    # the header, never in the source, regardless of how the source happens
    # to spell a call to it.
    assert "static inline" not in source
    assert re.search(r"^void gemm_small_infer\(", source, re.MULTILINE) is None
    # Weight symbols carry the model-name prefix (controller ruling R1) so
    # two different models' weight arrays never collide once `static` is
    # dropped and they gain external linkage.
    assert "extern double gemm_small_w0[" in header
    assert "\ndouble gemm_small_w0[" in source and "static double gemm_small_w0[" not in source


def test_library_and_header_inline_agree(tmp_path, golden_model):
    name = "gemm_small"
    graph = load_graph(golden_model(name))
    plan = build_plan(graph, dtype="f64", embed=False)
    source, header = emit_c(plan)
    (tmp_path / f"{name}.c").write_text(source)
    (tmp_path / f"{name}.h").write_text(header)
    write_weights(plan, graph, tmp_path / f"{name}.rwt")
    n_in, n_out = plan.input.shape[0], plan.output.shape[0]
    (tmp_path / "host.c").write_text(f"""
#include <stdio.h>
#include "{name}.h"
int main(void) {{
    double x[{n_in}], y[{n_out}];
    int n; if ({name}_init("{name}.rwt")) return 2;
    if (scanf("%d", &n) != 1) return 1;
    for (int c = 0; c < n; ++c) {{
        for (int i = 0; i < {n_in}; ++i) if (scanf("%lf", &x[i]) != 1) return 1;
        {name}_infer(x, y);                       /* the header inline, called from the host TU */
        for (int i = 0; i < {n_out}; ++i) printf("%.17e ", y[i]);
        printf("\\n");
    }}
    return 0;
}}
""")
    cc = _cc()
    subprocess.run([cc, "-O2", "-Wall", "-Wextra", "-std=c11", "-c", f"{name}.c"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["ar", "rcs", f"lib{name}.a", f"{name}.o"], cwd=tmp_path, check=True)
    subprocess.run([cc, "-O2", "-Wall", "-Wextra", "-std=c11", "host.c", f"lib{name}.a", "-lm", "-o", "host"], cwd=tmp_path, check=True, capture_output=True, text=True)
    session = ort.InferenceSession(golden_model(name))
    inputs, expected = _live_reference(session, session.get_inputs()[0].shape, np.float64, seed=3, batch=8)
    if inputs is None:
        pytest.skip(f"{name}: onnxruntime reference is all-zero across 10 resampled "
                    f"batches; its golden-file weights produced a dead model")
    stdin = f"{len(inputs)}\n" + "\n".join(" ".join(repr(float(v)) for v in row) for row in inputs)
    out = subprocess.run(["./host"], cwd=tmp_path, input=stdin, capture_output=True, text=True, check=True).stdout
    got = np.array([[float(v) for v in line.split()] for line in out.strip().splitlines()])
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_two_models_link_into_one_host(tmp_path, golden_model):
    """Two different models' libraries must link into one host binary.

    Reproduces the defect the reviewer found in 2e60a5d: dropping `static`
    from the weight definitions, without also prefixing them with the model
    name, gives every model's `w0`/`b0`/... external linkage under the same
    names (plan.py names weights identically across models), so `gemm_small`
    and `gemm_nobias` linked into one binary fail with
    `ld: duplicate symbols '_w0'`. Controller ruling R1: every C weight
    symbol is prefixed with the model name, so this must link, run, and each
    model's output must match its own onnxruntime reference.
    """
    names = ["gemm_small", "gemm_nobias"]
    plans, sessions = {}, {}
    cc = _cc()
    obj_args = []
    for name in names:
        graph = load_graph(golden_model(name))
        plan = build_plan(graph, dtype="f64", embed=False)
        plans[name] = plan
        source, header = emit_c(plan)
        (tmp_path / f"{name}.c").write_text(source)
        (tmp_path / f"{name}.h").write_text(header)
        write_weights(plan, graph, tmp_path / f"{name}.rwt")
        subprocess.run([cc, "-O2", "-Wall", "-Wextra", "-std=c11", "-c", f"{name}.c"],
                        cwd=tmp_path, check=True, capture_output=True, text=True)
        subprocess.run(["ar", "rcs", f"lib{name}.a", f"{name}.o"], cwd=tmp_path, check=True)
        obj_args.append(f"lib{name}.a")
        sessions[name] = ort.InferenceSession(golden_model(name))

    references = {}
    for name in names:
        session = sessions[name]
        shape = session.get_inputs()[0].shape
        inputs, expected = _live_reference(session, shape, np.float64, seed=5, batch=8)
        if inputs is None:
            pytest.skip(f"{name}: onnxruntime reference is all-zero across 10 resampled "
                        f"batches; its golden-file weights produced a dead model")
        references[name] = (inputs, expected)

    host_lines = ["#include <stdio.h>"]
    host_lines += [f'#include "{name}.h"' for name in names]
    host_lines.append("int main(void) {")
    for name in names:
        plan = plans[name]
        n_in, n_out = plan.input.shape[0], plan.output.shape[0]
        host_lines.append(f"    double {name}_x[{n_in}], {name}_y[{n_out}];")
        host_lines.append(f'    if ({name}_init("{name}.rwt")) return 2;')
    for name in names:
        plan = plans[name]
        n_in, n_out = plan.input.shape[0], plan.output.shape[0]
        host_lines.append("    { int n; if (scanf(\"%d\", &n) != 1) return 1;")
        host_lines.append("    for (int c = 0; c < n; ++c) {")
        host_lines.append(
            f"        for (int i = 0; i < {n_in}; ++i) "
            f"if (scanf(\"%lf\", &{name}_x[i]) != 1) return 1;")
        host_lines.append(f"        {name}_infer({name}_x, {name}_y);")
        host_lines.append(
            f"        for (int i = 0; i < {n_out}; ++i) printf(\"%.17e \", {name}_y[i]);")
        host_lines.append('        printf("\\n");')
        host_lines.append("    } }")
    host_lines.append("    return 0;")
    host_lines.append("}")
    (tmp_path / "host.c").write_text("\n".join(host_lines) + "\n")

    subprocess.run([cc, "-O2", "-Wall", "-Wextra", "-std=c11", "host.c", *obj_args, "-lm", "-o", "host"],
                    cwd=tmp_path, check=True, capture_output=True, text=True)

    stdin_parts = []
    for name in names:
        inputs, _ = references[name]
        stdin_parts.append(str(len(inputs)))
        stdin_parts.append("\n".join(" ".join(repr(float(v)) for v in row) for row in inputs))
    stdin = "\n".join(stdin_parts) + "\n"
    out = subprocess.run(["./host"], cwd=tmp_path, input=stdin, capture_output=True, text=True, check=True).stdout
    all_lines = out.strip().splitlines()
    pos = 0
    for name in names:
        inputs, expected = references[name]
        got = np.array([[float(v) for v in line.split()] for line in all_lines[pos:pos + len(inputs)]])
        pos += len(inputs)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_two_embedded_models_link_into_one_host(tmp_path, golden_model):
    """Two different models' EMBEDDED weights must link into one host binary.

    Mirrors test_two_models_link_into_one_host above, but for embed=True
    (the default for these small models). ROSENNA_CONST resolves to `static
    const` on the host, which gives each array internal linkage -- so an
    unprefixed `w0` in two headers would not raise a linker collision the
    way the file-loaded case's external `w0` did -- but both headers still
    land in the same translation unit here (host.c #includes both), and an
    unprefixed `w0` would be a duplicate *definition* inside that one TU
    regardless of linkage. Ruling R1 already prefixes embedded weight
    symbols with the model name (see emit_c._emit_embedded_weights); this
    test proves that rather than assuming it.
    """
    names = ["gemm_small", "gemm_nobias"]
    plans, sessions = {}, {}
    cc = _cc()
    for name in names:
        graph = load_graph(golden_model(name))
        plan = build_plan(graph, dtype="f64")
        assert plan.embed is True, f"{name}: expected to auto-embed for this test to be meaningful"
        plans[name] = plan
        source, header = emit_c(plan)
        (tmp_path / f"{name}.c").write_text(source)
        (tmp_path / f"{name}.h").write_text(header)
        sessions[name] = ort.InferenceSession(golden_model(name))

    references = {}
    for name in names:
        session = sessions[name]
        shape = session.get_inputs()[0].shape
        inputs, expected = _live_reference(session, shape, np.float64, seed=7, batch=8)
        if inputs is None:
            pytest.skip(f"{name}: onnxruntime reference is all-zero across 10 resampled "
                        f"batches; its golden-file weights produced a dead model")
        references[name] = (inputs, expected)

    host_lines = ["#include <stdio.h>"]
    host_lines += [f'#include "{name}.h"' for name in names]
    host_lines.append("int main(void) {")
    for name in names:
        plan = plans[name]
        n_in, n_out = plan.input.shape[0], plan.output.shape[0]
        host_lines.append(f"    double {name}_x[{n_in}], {name}_y[{n_out}];")
    for name in names:
        plan = plans[name]
        n_in, n_out = plan.input.shape[0], plan.output.shape[0]
        host_lines.append("    { int n; if (scanf(\"%d\", &n) != 1) return 1;")
        host_lines.append("    for (int c = 0; c < n; ++c) {")
        host_lines.append(
            f"        for (int i = 0; i < {n_in}; ++i) "
            f"if (scanf(\"%lf\", &{name}_x[i]) != 1) return 1;")
        host_lines.append(f"        {name}_infer({name}_x, {name}_y);")
        host_lines.append(
            f"        for (int i = 0; i < {n_out}; ++i) printf(\"%.17e \", {name}_y[i]);")
        host_lines.append('        printf("\\n");')
        host_lines.append("    } }")
    host_lines.append("    return 0;")
    host_lines.append("}")
    (tmp_path / "host.c").write_text("\n".join(host_lines) + "\n")

    # No lib{name}.a to link: an embedded plan's .c is nearly empty and
    # infer lives entirely in the header, so the host TU alone suffices.
    subprocess.run([cc, "-O2", "-Wall", "-Wextra", "-std=c11", "host.c", "-lm", "-o", "host"],
                    cwd=tmp_path, check=True, capture_output=True, text=True)

    stdin_parts = []
    for name in names:
        inputs, _ = references[name]
        stdin_parts.append(str(len(inputs)))
        stdin_parts.append("\n".join(" ".join(repr(float(v)) for v in row) for row in inputs))
    stdin = "\n".join(stdin_parts) + "\n"
    out = subprocess.run(["./host"], cwd=tmp_path, input=stdin, capture_output=True, text=True, check=True).stdout
    all_lines = out.strip().splitlines()
    pos = 0
    for name in names:
        inputs, expected = references[name]
        got = np.array([[float(v) for v in line.split()] for line in all_lines[pos:pos + len(inputs)]])
        pos += len(inputs)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_recipe_builds_the_library(tmp_path, golden_model):
    name = "gemm_small"
    graph = load_graph(golden_model(name))
    plan = build_plan(graph, dtype="f64", embed=False)
    source, header = emit_c(plan)
    (tmp_path / f"{name}.c").write_text(source)
    (tmp_path / f"{name}.h").write_text(header)
    (tmp_path / "Makefile").write_text(emit_c_recipe(plan))
    subprocess.run(["make", f"CC={_cc()}"], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert (tmp_path / f"lib{name}.a").exists()
