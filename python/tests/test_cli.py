import rosenna.verify as verify_mod
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


def test_verify_reports_compiler_diagnostic_on_a_compile_failure(capsys, live_gemm_model, monkeypatch):
    # Finding 1 (review round 2): a failing gfortran/gcc used to reach the user as an
    # unhandled CalledProcessError traceback, throwing away the compiler's own message.
    # Force a compile failure by making the Fortran emitter return garbage, and check
    # that the user instead sees which backend and step failed plus gfortran's actual
    # diagnostic -- not a traceback -- with a non-zero exit code.
    # On a deterministic inline model (ruling R22): this test is about the
    # compile step, and an unseeded golden model can be dead, which makes verify
    # return 1 with its all-zero message before the compiler is ever run.
    monkeypatch.setattr(verify_mod, "emit_fortran", lambda plan: "this is not fortran\n")
    rc = main(["verify", str(live_gemm_model), "--lang", "fortran", "--cases", "2"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "rosenna:" in err
    assert "fortran" in err
    assert "compile" in err
    assert "Error" in err  # gfortran's own diagnostic text


def test_info_reports_unsupported(capsys, golden_model):
    onnx_path = golden_model("mnist")
    rc = main(["info", str(onnx_path)])
    out = capsys.readouterr().out
    assert rc == 1
    # The actual rejection: mnist's node list is not in execution order, and the first
    # unsupported op validate() hits is a Reshape, not a Conv -- assert on the real
    # rejection text, not merely on an op name that _describe_ops would print either way.
    assert "Reshape is not supported" in out
    # If the rejection branch silently disappeared, build_plan would have to have
    # succeeded, and the success branch's bare "supported" line would appear instead.
    assert "supported" not in out.splitlines()


def test_generate_rejects_unsupported_model(tmp_path, capsys, golden_model):
    onnx_path = golden_model("mnist")
    rc = main(["generate", str(onnx_path), "--out", str(tmp_path)])
    assert rc == 1
    assert "rosenna:" in capsys.readouterr().err


def test_missing_model_file_is_reported_not_traced(capsys):
    # A mistyped path used to escape as a FileNotFoundError traceback.
    rc = main(["info", "no/such/model.onnx"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "rosenna:" in err
    assert "model.onnx" in err


def test_non_onnx_file_is_reported_not_traced(tmp_path, capsys):
    # A protobuf DecodeError used to escape as a traceback.
    bogus = tmp_path / "notes.onnx"
    bogus.write_text("this is not a protobuf\n")
    rc = main(["info", str(bogus)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "rosenna:" in err


def test_missing_compiler_is_reported_not_traced(capsys, monkeypatch, live_gemm_model):
    # verify._run caught CalledProcessError but not FileNotFoundError, so a
    # gfortran that is simply not on PATH reached the user as a traceback.
    # Deterministic inline model for the same reason as the test above (R22).
    monkeypatch.setenv("PATH", "")
    rc = main(["verify", str(live_gemm_model), "--lang", "fortran", "--cases", "2"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "rosenna:" in err
    assert "gfortran" in err


def test_generate_rejects_a_name_that_is_not_an_identifier(tmp_path, capsys, golden_model):
    # The name is interpolated into `module <name>_model` and function names.
    onnx_path = golden_model("gemm_small")
    rc = main(["generate", str(onnx_path), "--out", str(tmp_path), "--name", "my-model.v2"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "rosenna:" in err
    assert "--name" in err


def test_generate_rejects_a_dotted_file_stem(tmp_path, capsys, golden_model):
    # The default name is the file stem, and dotted ONNX filenames are routine.
    onnx_path = golden_model("gemm_small")
    dotted = tmp_path / "my-model.v2.onnx"
    dotted.write_bytes(onnx_path.read_bytes())
    rc = main(["generate", str(dotted), "--out", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "--name" in err


def test_live_gemm_model_verifies_end_to_end(capsys, live_gemm_model):
    # The fixture's premise, checked: verify on the inline model reaches the
    # compile and run steps and passes, so the two error-path tests above are
    # exercising the code they name rather than the dead-model guard.
    rc = main(["verify", str(live_gemm_model), "--cases", "2"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "all-zero" not in out and "FAIL" not in out
