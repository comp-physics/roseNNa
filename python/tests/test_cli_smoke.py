import pytest
from rosenna.cli import main

def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert "generate" in capsys.readouterr().out

def test_generate_parses_arguments(capsys):
    rc = main(["generate", "model.onnx", "--lang", "both", "--precision", "double",
               "--out", "outdir", "--name", "mymodel"])
    assert rc == 2
    assert "not implemented yet" in capsys.readouterr().err
