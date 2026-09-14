import subprocess

import onnx
from onnx import helper, TensorProto

import rosenna.verify as verify_mod
from rosenna.cli import main
from tests.test_library_form import _cc
from tests.test_device_fortran import _omp_fc


def _unsupported_model(tmp_path, name="softmaxed"):
    """A minimal model using an op the generator does not lower.

    Not a golden file: the golden set is what the generator is growing to
    cover, so pinning a rejection test to one of them turns every genuine
    coverage win into a spurious failure (mnist did exactly that once Conv,
    MaxPool, Add and the shape ops landed). Softmax is unsupported on purpose.
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Softmax", ["x"], ["y"], axis=1, name="sm0")
    m = helper.make_model(helper.make_graph([node], "t", [x], [y]),
                          opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    path = tmp_path / f"{name}.onnx"
    onnx.save(m, str(path))
    return path


def test_generate_writes_all_artifacts(tmp_path, capsys, golden_model):
    # --no-embed: gemm_small auto-embeds (well under EMBED_THRESHOLD) in both
    # languages now (Task 4), so this test forces the file-loaded contract to
    # exercise the .rwt-writing path it asserts on.
    onnx_path = golden_model("gemm_small")
    rc = main(["generate", str(onnx_path),
               "--lang", "both", "--precision", "double", "--out", str(tmp_path), "--no-embed"])
    assert rc == 0
    for f in ["gemm_small_model.F90", "gemm_small.c", "gemm_small.h", "gemm_small.rwt"]:
        assert (tmp_path / f).exists(), f
    out = capsys.readouterr().out
    assert "gemm_small.rwt" in out


def test_generate_writes_rwt_only_when_not_embedding(tmp_path, capsys, golden_model):
    # gemm_small auto-embeds (well under EMBED_THRESHOLD) as of Task 4 in
    # both languages, so neither --lang both nor --lang c writes a .rwt by
    # default; --no-embed is what brings it back, regardless of --lang.
    onnx_path = golden_model("gemm_small")

    rc = main(["generate", str(onnx_path), "--lang", "both", "--out", str(tmp_path / "both")])
    assert rc == 0
    assert not (tmp_path / "both" / "gemm_small.rwt").exists()

    rc = main(["generate", str(onnx_path), "--lang", "c", "--out", str(tmp_path / "c")])
    assert rc == 0
    assert not (tmp_path / "c" / "gemm_small.rwt").exists()
    out = capsys.readouterr().out
    assert "embedded weights" in out

    rc = main(["generate", str(onnx_path), "--lang", "both", "--no-embed",
               "--out", str(tmp_path / "noembed")])
    assert rc == 0
    assert (tmp_path / "noembed" / "gemm_small.rwt").exists()


def test_generate_writes_the_fortran_recipe(tmp_path, golden_model):
    onnx_path = golden_model("gemm_small")
    rc = main(["generate", str(onnx_path), "--lang", "fortran", "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "gemm_small_model.F90").exists()
    mk = tmp_path / "gemm_small_fortran.mk"
    assert mk.exists()
    # Ruling R13: the Fortran archive is lib<name>_f.a, not lib<name>.a --
    # the latter is the C recipe's archive, and the two must never collide
    # when both recipes build in the same directory (see the two-archive
    # test below, which is what would have caught that defect).
    assert "libgemm_small_f.a: gemm_small_model.o" in mk.read_text()
    # The C recipe (a separate file, a separate object) is untouched by a
    # Fortran-only generate.
    assert not (tmp_path / "gemm_small.mk").exists()


def test_both_recipes_build_distinct_archives_in_one_directory(tmp_path, golden_model):
    # Controller ruling R13, reproducing the reviewer's finding on 352c14a:
    # `generate --lang both` writes both <name>.mk (C) and <name>_fortran.mk
    # (Fortran) into ONE output directory, and both recipes used to archive
    # into the same lib<name>.a -- `ar rcs` APPENDS, so building both there
    # in sequence silently merged gemm_small_model.o into gemm_small.a's own
    # archive, and either recipe's `clean` then deleted the shared file.
    # This is exactly the scenario the bind(C) interface serves: a Fortran
    # host that `use`s the module AND links the native CUDA/HIP kernel needs
    # both archives to coexist, distinctly, in one place.
    name = "gemm_small"
    onnx_path = golden_model(name)
    rc = main(["generate", str(onnx_path), "--lang", "both", "--out", str(tmp_path), "--no-embed"])
    assert rc == 0

    cc, fc = _cc(), _omp_fc()
    subprocess.run(["make", "-f", f"{name}.mk", f"CC={cc}"], cwd=tmp_path,
                   check=True, capture_output=True, text=True)
    subprocess.run(["make", "-f", f"{name}_fortran.mk", f"FC={fc}"], cwd=tmp_path,
                   check=True, capture_output=True, text=True)

    c_archive, f_archive = tmp_path / f"lib{name}.a", tmp_path / f"lib{name}_f.a"
    assert c_archive.exists() and f_archive.exists()
    assert c_archive != f_archive

    def _members(archive):
        # One member per line; BSD ar (macOS) also lists a "__.SYMDEF SORTED"
        # pseudo-member (its own symbol table) that GNU ar's `ar t` omits --
        # filter it out rather than split() on whitespace, since its name
        # itself contains a space.
        out = subprocess.run(["ar", "t", str(archive)], cwd=tmp_path,
                             check=True, capture_output=True, text=True).stdout
        return [line for line in out.splitlines() if not line.startswith("__.SYMDEF")]

    # Each archive holds only its own object -- not the other's, and not both
    # (the merged-archive defect: one .a holding gemm_small_model.o AND
    # gemm_small.o together, silently, because `ar rcs` appends).
    assert _members(c_archive) == [f"{name}.o"]
    assert _members(f_archive) == [f"{name}_model.o"]

    # `clean` on one recipe never touches the other's archive or object.
    subprocess.run(["make", "-f", f"{name}_fortran.mk", "clean"], cwd=tmp_path,
                   check=True, capture_output=True, text=True)
    assert not f_archive.exists() and not (tmp_path / f"{name}_model.o").exists()
    assert c_archive.exists() and (tmp_path / f"{name}.o").exists()

    subprocess.run(["make", "-f", f"{name}.mk", "clean"], cwd=tmp_path,
                   check=True, capture_output=True, text=True)
    assert not c_archive.exists() and not (tmp_path / f"{name}.o").exists()


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


def test_info_reports_unsupported(tmp_path, capsys):
    onnx_path = _unsupported_model(tmp_path)
    rc = main(["info", str(onnx_path)])
    out = capsys.readouterr().out
    assert rc == 1
    # Assert on the real rejection text, not merely on an op name _describe_ops
    # would print either way.
    assert "Softmax is not supported" in out
    # If the rejection branch silently disappeared, build_plan would have to have
    # succeeded, and the success branch's bare "supported" line would appear instead.
    assert "supported" not in out.splitlines()


def test_generate_rejects_unsupported_model(tmp_path, capsys):
    onnx_path = _unsupported_model(tmp_path)
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
