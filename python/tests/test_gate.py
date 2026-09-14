import shutil
import pytest
from rosenna.cli import main
from tests.test_device_c import _omp_cc


def test_gate_runs_in_host_fallback_mode_and_writes_a_report(tmp_path, golden_model):
    # On a machine without a GPU the gate runs with --host-fallback, which drops the
    # MANDATORY requirement but exercises every other step, so the script itself is tested.
    golden_model("gemm_big")
    cc = _omp_cc()
    fc = shutil.which("gfortran") or pytest.skip("no gfortran")
    rc = main(["gpu-gate", "--cc", cc, "--fc", fc, "--flags", "-fopenmp", "--backend", "omp",
              "--host-fallback", "--out", str(tmp_path)])
    assert rc == 0
    report = (tmp_path / "gate-report.md").read_text()
    for key in ("gemm_big", "embedded", "file-loaded", "fortran", "c", "infer_batch",
               "backend: omp", "ns per point", "host-fallback"):
        assert key in report


def test_gate_fails_loudly_when_offload_is_mandatory_and_absent(tmp_path, golden_model):
    golden_model("gemm_big")
    cc = _omp_cc()
    fc = shutil.which("gfortran") or pytest.skip("no gfortran")
    rc = main(["gpu-gate", "--cc", cc, "--fc", fc, "--flags", "-fopenmp", "--backend", "omp",
              "--out", str(tmp_path)])
    assert rc == 1
    assert "MANDATORY" in (tmp_path / "gate-report.md").read_text()
