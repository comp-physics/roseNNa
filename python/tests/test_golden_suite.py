"""Every golden model, both backends, against onnxruntime.

This is the replacement for `test/run.sh`, the shell suite that drove the
original `fLibrary/` runtime parser: same 21 models, but generated rather than
parsed at runtime, and compared against onnxruntime instead of against recorded
`.txt` output from the PyTorch script that built the model.

Comparing against onnxruntime rather than a recorded file is the point. A
recorded golden file pins whatever the library did on the day it was recorded,
so a wrong-but-stable implementation records its own error as the expectation;
onnxruntime is an independent implementation of the same ONNX semantics.
"""
import re
import subprocess
import sys

import pytest

from rosenna.verify import verify_model

# Kept as a literal list rather than a glob so that a golden model quietly
# disappearing is a failure, not a silently smaller suite.
GOLDEN = [
    "avgpool_basic", "batchnet", "conv_basic", "conv_grouped", "conv_padding",
    "conv1d_stack", "conv_padding-stride", "conv_strides", "droplet", "gemm_big",
    "gemm_nobias", "gemm_small", "gru_cell", "lstm_cell", "lstm_gemm", "lstm_gemm_hid",
    "lstm_nostate", "lstm_output", "maxpool_basic", "maxpool_nonsquare",
    "maxpool_padding", "maxpool_strides", "mnist", "pool_batch",
    "softmax_head",
]


def test_the_suite_covers_every_golden_directory(repo_root):
    """The list above is the whole golden set, not a subset someone trimmed."""
    on_disk = {d.name for d in (repo_root / "goldenFiles").iterdir()
               if d.is_dir() and (d / f"{d.name}.py").exists()}
    assert on_disk == set(GOLDEN), (
        f"golden set changed: only on disk {sorted(on_disk - set(GOLDEN))}, "
        f"only in this list {sorted(set(GOLDEN) - on_disk)}")


@pytest.mark.parametrize("model", GOLDEN)
def test_golden_model_matches_onnxruntime_on_both_backends(model, tmp_path, golden_model):
    onnx_path = golden_model(model)
    # The generated symbols become Fortran/C identifiers, and one golden model
    # is named with a hyphen.
    name = re.sub(r"[^0-9A-Za-z_]", "_", model)
    results = verify_model(onnx_path, "both", None, 8, tmp_path, name=name)
    assert {r.lang for r in results} == {"fortran", "c"}
    for r in results:
        assert r.ok, (f"{model}/{r.lang}: max_abs={r.max_abs:.3e} "
                      f"max_rel={r.max_rel:.3e} against onnxruntime")


@pytest.mark.parametrize("model", GOLDEN)
def test_both_backends_agree_with_each_other(model, tmp_path, golden_model):
    """C and Fortran are rendered from one plan, so they must agree far more
    tightly than either agrees with onnxruntime: same order, same arithmetic."""
    onnx_path = golden_model(model)
    name = re.sub(r"[^0-9A-Za-z_]", "_", model)
    results = verify_model(onnx_path, "both", "f64", 8, tmp_path, name=name)
    by_lang = {r.lang: r for r in results}
    # Both are compared against the same reference, so equal error figures mean
    # equal outputs -- to the last digit printed, across every case.
    assert by_lang["c"].max_abs == pytest.approx(by_lang["fortran"].max_abs, rel=1e-12), \
        f"{model}: backends disagree (c {by_lang['c'].max_abs}, f {by_lang['fortran'].max_abs})"
