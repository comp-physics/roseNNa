import pytest
from rosenna.cli import main

def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert "generate" in capsys.readouterr().out

def test_generate_writes_model_file(tmp_path, capsys, golden_model):
    onnx_path = golden_model("gemm_small")
    rc = main(["generate", str(onnx_path), "--lang", "both", "--precision", "double",
               "--out", str(tmp_path), "--name", "mymodel"])
    assert rc == 0
    assert (tmp_path / "mymodel_model.f90").exists()
    assert (tmp_path / "mymodel.c").exists()
    assert (tmp_path / "mymodel.h").exists()
    assert (tmp_path / "mymodel.rwt").exists()
    out = capsys.readouterr().out
    assert "mymodel.rwt" in out
