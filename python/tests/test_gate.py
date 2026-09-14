import shutil
import pytest
from rosenna.cli import main
from rosenna.gate import _scope_hipmemcpy_calls, _sum_cudamemcpy_calls
from tests.conftest import skip_unless_libgomp_enforces_mandatory
from tests.test_device_c import _omp_cc

# One header shape `nsys stats --report cuda_api_sum --format csv` actually
# produces, close enough to exercise the parser's real column-matching path
# rather than a hand-simplified stand-in.
_NSYS_CSV_HEADER = (
    '"Time (%)","Total Time (ns)","Num Calls","Avg (ns)","Med (ns)",'
    '"Min (ns)","Max (ns)","StdDev (ns)","Name"\n'
)


def test_sum_cudamemcpy_calls_sums_the_matching_rows():
    # Ruling R18 (a): two cudaMemcpy* rows (3 + 2 = 5 calls) plus one
    # unrelated cudaLaunchKernel row; only the memcpy rows count.
    csv_text = _NSYS_CSV_HEADER + (
        '45.0,12345,3,4115.0,4000.0,3900.0,4500.0,120.5,"cudaMemcpyAsync"\n'
        '30.0,8000,2,4000.0,4000.0,3900.0,4100.0,50.0,"cudaMemcpyHtoD"\n'
        '25.0,6000,10,600.0,600.0,500.0,700.0,20.0,"cudaLaunchKernel"\n'
    )
    result = _sum_cudamemcpy_calls(csv_text)
    assert result.parsed is True
    assert result.count == 5


def test_sum_cudamemcpy_calls_is_a_parsed_zero_with_no_memcpy_rows():
    # Ruling R18 (b): a genuinely parsed export with zero cudaMemcpy* rows
    # (only a launch row) is a real pass, not a fallback/unparsed zero --
    # `parsed` distinguishes the two.
    csv_text = _NSYS_CSV_HEADER + (
        '100.0,6000,10,600.0,600.0,500.0,700.0,20.0,"cudaLaunchKernel"\n'
    )
    result = _sum_cudamemcpy_calls(csv_text)
    assert result.parsed is True
    assert result.count == 0


def test_sum_cudamemcpy_calls_reports_not_parsed_rather_than_a_false_zero():
    # Ruling R18 (c): neither an empty string nor an unrelated-columns CSV
    # may come back as `parsed=True, count=0` -- that would be a silent
    # pass on a check whose whole purpose is R5 evidence.
    empty = _sum_cudamemcpy_calls("")
    assert empty.parsed is False
    assert empty.count == 0

    unrelated = _sum_cudamemcpy_calls("foo,bar\n1,2\n3,4\n")
    assert unrelated.parsed is False
    assert unrelated.count == 0


def test_sum_cudamemcpy_calls_skips_the_stdout_preamble():
    # `nsys stats --format csv` on stdout is preceded by progress lines and a
    # report title; the header is the first line naming Num Calls and Name,
    # not the first line of stdout.
    preamble = (
        "Generating SQLite file gate_nsys_profile.sqlite from gate_nsys_profile.nsys-rep\n"
        "Processing [gate_nsys_profile.sqlite] with [/opt/nvidia/nsight-systems/reports/cuda_api_sum.py]...\n"
        "\n"
        " ** CUDA API Summary (cuda_api_sum):\n"
        "\n"
    )
    csv_text = preamble + _NSYS_CSV_HEADER + (
        '60.0,12000,3,4000.0,4000.0,3900.0,4100.0,50.0,"cudaMemcpy"\n'
        '40.0,6000,10,600.0,600.0,500.0,700.0,20.0,"cudaLaunchKernel"\n'
    )
    result = _sum_cudamemcpy_calls(csv_text)
    assert result.parsed is True
    assert result.count == 3
    # A preamble with no CSV after it is still not parsed.
    assert _sum_cudamemcpy_calls(preamble) == _sum_cudamemcpy_calls("")


def test_gate_flags_follow_the_compiler_basename():
    # Ruling R24: the gcc-style warning and -std flags only for gcc, gfortran,
    # cc and clang; a vendor compiler gets -O2 and the user's --flags.
    from rosenna.gate import _c_flags, _f_flags
    assert _c_flags("gcc-15") == ["-O2", "-Wall", "-Wextra", "-std=c11"]
    assert _c_flags("/usr/bin/clang") == ["-O2", "-Wall", "-Wextra", "-std=c11"]
    assert _c_flags("cc") == ["-O2", "-Wall", "-Wextra", "-std=c11"]
    assert _f_flags("gfortran") == ["-O2", "-Wall", "-Wextra", "-std=f2008"]
    for vendor in ("nvc", "nvfortran", "amdclang", "amdflang", "flang", "icx", "ifx",
                   "/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/compilers/bin/nvc"):
        assert _c_flags(vendor) == ["-O2"], vendor
        assert _f_flags(vendor) == ["-O2"], vendor


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
    # Rulings R21/R24: the recipes get CFLAGS/FFLAGS explicitly and the host
    # compiler links the per-point harness against the omp-backend archive.
    assert f"CC={cc} 'CFLAGS=-O2 -Wall -Wextra -std=c11' ROSENNA_OFFLOAD_FLAGS=-fopenmp" in report
    assert f"FC={fc} 'FFLAGS=-O2 -Wall -Wextra -std=f2008' ROSENNA_OFFLOAD_FLAGS=-fopenmp" in report
    assert "backend=omp, host compiler: serves the per-point harness" in report
    assert f"$ {cc} -fopenmp gate_harness1.o libgemm_big.a -lm -o gate_harness1" in report


def test_gate_fails_loudly_when_offload_is_mandatory_and_absent(tmp_path, golden_model):
    # The gate's rc=1 here IS libgomp refusing the harness under MANDATORY;
    # where libgomp ignores MANDATORY there is nothing to observe, so skip.
    golden_model("gemm_big")
    cc = _omp_cc()
    fc = shutil.which("gfortran") or pytest.skip("no gfortran")
    skip_unless_libgomp_enforces_mandatory(cc, tmp_path)
    rc = main(["gpu-gate", "--cc", cc, "--fc", fc, "--flags", "-fopenmp", "--backend", "omp",
              "--out", str(tmp_path)])
    assert rc == 1
    assert "MANDATORY" in (tmp_path / "gate-report.md").read_text()


# rocprofv3 --hip-trace --marker-trace -f csv writes one CSV per domain with
# this header; the marker CSV names the roctx range, the HIP one every API
# call, both with the same clock. These are the real headers ROCm 7.2 wrote.
_ROCPROF_HEADER = ('"Domain","Function","Process_Id","Thread_Id","Correlation_Id",'
                   '"Start_Timestamp","End_Timestamp"\n')
_MARKER_CSV = _ROCPROF_HEADER + '"MARKER_CORE_RANGE_API","rosenna_timed",1,1,14,1000,2000\n'


def test_scope_hipmemcpy_calls_counts_only_inside_the_timed_range():
    # One hipMemcpy before the range (setup), two inside, one after: 2.
    hip_csv = _ROCPROF_HEADER + (
        '"HIP_RUNTIME_API","hipMemcpy",1,1,2,500,600\n'
        '"HIP_RUNTIME_API","hipMemcpy",1,1,3,1100,1200\n'
        '"HIP_RUNTIME_API","hipMemcpyAsync",1,1,4,1300,1400\n'
        '"HIP_RUNTIME_API","hipLaunchKernel",1,1,5,1500,1600\n'
        '"HIP_RUNTIME_API","hipMemcpy",1,1,6,2100,2200\n'
    )
    result = _scope_hipmemcpy_calls(hip_csv, _MARKER_CSV)
    assert result.parsed is True
    assert result.count == 2


def test_scope_hipmemcpy_calls_is_a_parsed_zero_when_only_launches_are_inside():
    hip_csv = _ROCPROF_HEADER + (
        '"HIP_RUNTIME_API","hipMemcpy",1,1,2,500,600\n'
        '"HIP_RUNTIME_API","hipLaunchKernel",1,1,5,1500,1600\n'
        '"HIP_RUNTIME_API","hipGetLastError",1,1,6,1700,1750\n'
    )
    result = _scope_hipmemcpy_calls(hip_csv, _MARKER_CSV)
    assert result.parsed is True
    assert result.count == 0


def test_scope_hipmemcpy_calls_reports_not_parsed_rather_than_a_false_zero():
    # No range named rosenna_timed, an empty HIP trace, or unrelated columns:
    # none of these is a passing zero.
    hip_csv = _ROCPROF_HEADER + '"HIP_RUNTIME_API","hipLaunchKernel",1,1,5,1500,1600\n'
    assert _scope_hipmemcpy_calls(hip_csv, _ROCPROF_HEADER).parsed is False
    assert _scope_hipmemcpy_calls(_ROCPROF_HEADER, _MARKER_CSV).parsed is False
    assert _scope_hipmemcpy_calls("foo,bar\n1,2\n", _MARKER_CSV).parsed is False
    assert _scope_hipmemcpy_calls("", "").parsed is False
