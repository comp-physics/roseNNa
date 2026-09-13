from rosenna.cli import main


def test_generate_writes_all_artifacts(tmp_path, capsys, golden_model):
    onnx_path = golden_model("gemm_small")
    rc = main(["generate", str(onnx_path),
               "--lang", "both", "--precision", "double", "--out", str(tmp_path)])
    assert rc == 0
    for f in ["gemm_small_model.f90", "gemm_small.c", "gemm_small.h", "gemm_small.rwt"]:
        assert (tmp_path / f).exists(), f
    out = capsys.readouterr().out
    assert "gemm_small.rwt" in out


def test_verify_passes_on_a_dense_model(capsys, golden_model):
    onnx_path = golden_model("gemm_small")
    rc = main(["verify", str(onnx_path), "--cases", "4"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fortran" in out and "c" in out and "ok" in out


def test_verify_double_precision_build_on_a_float32_model_still_passes(capsys, golden_model):
    # Controller ruling R11: the golden dense models are all float32 ONNX exports, so
    # onnxruntime's reference is computed in float32 no matter what precision we ask
    # rosenna to generate. Before this fix, --precision double compared against the
    # f64 tolerance (rtol=1e-9, atol=1e-12) -- far tighter than an fp32 reference can
    # support -- and this would have failed. The tolerance must be keyed on the model's
    # own (float32) dtype, so this build is compared at the fp32-appropriate tolerance
    # and passes.
    onnx_path = golden_model("gemm_big")
    rc = main(["verify", str(onnx_path), "--precision", "double", "--cases", "32"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "fortran" in out and "c" in out and "ok" in out
    assert "FAIL" not in out


def test_info_reports_unsupported(capsys, golden_model):
    onnx_path = golden_model("mnist")
    rc = main(["info", str(onnx_path)])
    assert rc == 1
    assert "Conv" in capsys.readouterr().out


def test_generate_rejects_unsupported_model(tmp_path, capsys, golden_model):
    onnx_path = golden_model("mnist")
    rc = main(["generate", str(onnx_path), "--out", str(tmp_path)])
    assert rc == 1
    assert "rosenna:" in capsys.readouterr().err
