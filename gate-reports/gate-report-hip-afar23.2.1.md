
# rosenna gpu-gate report

- model: gemm_big
- backend: hip
- host-fallback: False
- cc: /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdclang
- fc: /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdflang
- flags: '-fopenmp --offload-arch=gfx90a'
- devcc: /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc
- devflags: ''
- OMP_TARGET_OFFLOAD=MANDATORY is set for every omp-backend harness: a machine with no working offload device must fail here, loudly, rather than silently pass by falling back to the host.

### toolchain

platform: Linux-5.14.0-611.54.1.el9_7.x86_64-x86_64-with-glibc2.34
cc (/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdclang) --version:
```
AMD AFAR drop #23.2.0 04/18/26 clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 35849413f758a222a8094acf1ec81eb80f601335+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/lib/llvm/bin

```
fc (/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdflang) --version:
```
AMD AFAR drop #23.2.0 04/18/26 flang version 23.0.0git (https://github.com/ROCm/llvm-project.git 35849413f758a222a8094acf1ec81eb80f601335+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/lib/llvm/bin

```
devcc (/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc) --version:
```
HIP version: 7.13.26154-92b7431876
AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/rocm-7.2.0/lib/llvm/bin
Configuration file: /opt/rocm-7.2.0/lib/llvm/bin/clang++.cfg

```

## gemm_big: embedded


**build c library (embedded, backend=omp, host compiler: serves the per-point harness)** (in gate-hip-afar/embedded)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=omp CC=/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdclang CFLAGS=-O2 'ROSENNA_OFFLOAD_FLAGS=-fopenmp --offload-arch=gfx90a'
```
exit status: 0
stdout:
```
/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdclang -O2 -fopenmp --offload-arch=gfx90a -c gemm_big.c -o gemm_big.o
ar rcs libgemm_big.a gemm_big.o

```

**build c library (embedded, backend=hip, device compiler, in hip_lib/: serves the infer_batch harness)** (in gate-hip-afar/embedded/hip_lib)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=hip DEVCC=/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc
```
exit status: 0
stdout:
```
/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc -O2 -x hip -c gemm_big.c -o gemm_big.o
/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc -O2 -x hip -c gemm_big_kernel.cu -o gemm_big_kernel.o
ar rcs libgemm_big.a gemm_big.o gemm_big_kernel.o

```

**build fortran library (embedded)** (in gate-hip-afar/embedded)

```
$ make -f gemm_big_fortran.mk FC=/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdflang FFLAGS=-O2 'ROSENNA_OFFLOAD_FLAGS=-fopenmp --offload-arch=gfx90a'
```
exit status: 0
stdout:
```
/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdflang -O2 -fopenmp --offload-arch=gfx90a -c gemm_big_model.F90 -o gemm_big_model.o
ar rcs libgemm_big_f.a gemm_big_model.o

```

### c harness: per-point infer via target teams distribute parallel for


**compile c per-point harness (host compiler, host offload flags)** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdclang -O2 -fopenmp --offload-arch=gfx90a -c gate_harness1.c -o gate_harness1.o
```
exit status: 0

**link c per-point harness (host compiler, host offload flags, omp-backend archive)** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdclang -fopenmp --offload-arch=gfx90a gate_harness1.o -lm -o gate_harness1
```
exit status: 0

**run c per-point harness** (in gate-hip-afar/embedded)

```
$ ./gate_harness1
```
exit status: 0
stdout:
```
5.06990979886064563e-01 
5.07096493191199316e-01 
5.09079712744220259e-01 
5.07638116652864846e-01 
5.05334062551293983e-01 
5.09260268684951223e-01 
5.08417446220905567e-01 
5.03998585398111043e-01 
TIMING 3.108025

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
3.108 ns per point

### fortran harness: per-point infer via target teams distribute parallel do


**compile fortran per-point harness** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdflang -O2 -fopenmp --offload-arch=gfx90a gate_harness2.f90 libgemm_big_f.a -o gate_harness2
```
exit status: 0
stderr:
```
ld.lld: warning: <unknown>:0:0: in function __keep_alive void (): local memory global used by non-kernel function


```

**run fortran per-point harness** (in gate-hip-afar/embedded)

```
$ ./gate_harness2
```
exit status: 0
stdout:
```
  5.0699097988606456E-01
  5.0709649319119932E-01
  5.0907971274422026E-01
  5.0763811665286485E-01
  5.0533406255129398E-01
  5.0926026868495122E-01
  5.0841744622090557E-01
  5.0399858539811104E-01
TIMING   4.6791049999999998E+00

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
4.679 ns per point

### infer_batch harness: device-resident data


**compile and link hip infer_batch harness (device compiler)** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc gate_harness3.cu -Lhip_lib -lgemm_big -o gate_harness3_dev
```
exit status: 0
stderr:
```
gate_harness3.cu:65:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   65 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:69:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:69:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:69:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:69:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                                             ^~~~~~~~~~~~
5 warnings generated when compiling for gfx90a.
gate_harness3.cu:65:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   65 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:69:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:69:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:69:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:69:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                                             ^~~~~~~~~~~~
5 warnings generated when compiling for host.

```

**run hip infer_batch harness** (in gate-hip-afar/embedded)

```
$ ./gate_harness3_dev
```
exit status: 0
stdout:
```
5.06990979886064563e-01 
5.07096493191199316e-01 
5.09079712744220259e-01 
5.07638116652864846e-01 
5.05334062551293983e-01 
5.09260268684951223e-01 
5.08417446220905567e-01 
5.03998585398111043e-01 
TIMING 4.049408

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
4.049 ns per point

#### rocprof check: hipMemcpy count inside the roctx-scoped infer_batch call (ruling R15)


**probe for <rocprofiler-sdk-roctx/roctx.h>** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc -c gate_marker_probe.cu -o gate_marker_probe.o
```
exit status: 0

**compile roctx-bracketed infer_batch harness** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc -DROSENNA_GATE_MARKERS=1 gate_harness3.cu -Lhip_lib -lgemm_big -lrocprofiler-sdk-roctx -o gate_harness3_roctx
```
exit status: 0
stderr:
```
gate_harness3.cu:65:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   65 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:69:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:69:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:69:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:69:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                                             ^~~~~~~~~~~~
5 warnings generated when compiling for gfx90a.
gate_harness3.cu:65:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   65 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:69:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:69:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:69:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:69:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                                             ^~~~~~~~~~~~
5 warnings generated when compiling for host.

```

**rocprofv3 --hip-trace --marker-trace -f csv** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/rocprofv3 --hip-trace --marker-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof -o prof -- ./gate_harness3_roctx
```
exit status: 0
stdout:
```
5.06990979886064563e-01 
5.07096493191199316e-01 
5.09079712744220259e-01 
5.07638116652864846e-01 
5.05334062551293983e-01 
5.09260268684951223e-01 
5.08417446220905567e-01 
5.03998585398111043e-01 
TIMING 4.748946

```
stderr:
```
W20260914 18:29:04.811827 140542279145920 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.002735 sec
W20260914 18:29:04.814598 140542279145920 tool.cpp:2693] HIP (compiler) version 7.2.0 initialized (instance=0)
W20260914 18:29:04.814685 140542279145920 simple_timer.cpp:55] [rocprofv3] './gate_harness3_roctx' ::     0.000000 sec
W20260914 18:29:04.815863 140542279145920 tool.cpp:2693] HIP (runtime) version 7.2.0 initialized (instance=0)
W20260914 18:29:04.856001 140542279145920 tool.cpp:2693] HSA version 8.20.0 initialized (instance=0)
W20260914 18:29:04.945172 140542279145920 tool.cpp:2693] MARKER (ROCTx) version 1.2.3 initialized (instance=0)
W20260914 18:29:04.949859 140542279145920 simple_timer.cpp:55] [rocprofv3] './gate_harness3_roctx' ::     0.135174 sec
E20260914 18:29:04.985285 140542279145920 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof/prof_hip_api_trace.csv
E20260914 18:29:05.007173 140542279145920 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof/prof_marker_api_trace.csv
E20260914 18:29:05.011072 140542279145920 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof/prof_agent_info.csv
W20260914 18:29:05.015614 140542279145920 simple_timer.cpp:55] [rocprofv3] output generation ::     0.063641 sec
W20260914 18:29:05.015691 140542279145920 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.063782 sec

```
hipMemcpy* calls inside the roctx-scoped infer_batch call, from hip_api_trace cut to marker_api_trace: 0

## gemm_big: file-loaded


**build c library (file-loaded, backend=omp, host compiler: serves the per-point harness)** (in gate-hip-afar/file_loaded)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=omp CC=/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdclang CFLAGS=-O2 'ROSENNA_OFFLOAD_FLAGS=-fopenmp --offload-arch=gfx90a'
```
exit status: 0
stdout:
```
/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdclang -O2 -fopenmp --offload-arch=gfx90a -c gemm_big.c -o gemm_big.o
ar rcs libgemm_big.a gemm_big.o

```

**build c library (file-loaded, backend=hip, device compiler, in hip_lib/: serves the infer_batch harness)** (in gate-hip-afar/file_loaded/hip_lib)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=hip DEVCC=/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc
```
exit status: 0
stdout:
```
/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc -O2 -x hip -c gemm_big.c -o gemm_big.o
/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc -O2 -x hip -c gemm_big_kernel.cu -o gemm_big_kernel.o
ar rcs libgemm_big.a gemm_big.o gemm_big_kernel.o

```

**build fortran library (file-loaded)** (in gate-hip-afar/file_loaded)

```
$ make -f gemm_big_fortran.mk FC=/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdflang FFLAGS=-O2 'ROSENNA_OFFLOAD_FLAGS=-fopenmp --offload-arch=gfx90a'
```
exit status: 0
stdout:
```
/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdflang -O2 -fopenmp --offload-arch=gfx90a -c gemm_big_model.F90 -o gemm_big_model.o
ar rcs libgemm_big_f.a gemm_big_model.o

```

### c harness: per-point infer via target teams distribute parallel for


**compile c per-point harness (host compiler, host offload flags)** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdclang -O2 -fopenmp --offload-arch=gfx90a -c gate_harness1.c -o gate_harness1.o
```
exit status: 0

**link c per-point harness (host compiler, host offload flags, omp-backend archive)** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdclang -fopenmp --offload-arch=gfx90a gate_harness1.o libgemm_big.a -lm -o gate_harness1
```
exit status: 0
stderr:
```
ld.lld: warning: <unknown>:0:0: in function __keep_alive void (): local memory global used by non-kernel function


```

**run c per-point harness** (in gate-hip-afar/file_loaded)

```
$ ./gate_harness1
```
exit status: 0
stdout:
```
5.06990979886064563e-01 
5.07096493191199316e-01 
5.09079712744220259e-01 
5.07638116652864846e-01 
5.05334062551293983e-01 
5.09260268684951223e-01 
5.08417446220905567e-01 
5.03998585398111043e-01 
TIMING 5.787134

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
5.787 ns per point

### fortran harness: per-point infer via target teams distribute parallel do


**compile fortran per-point harness** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdflang -O2 -fopenmp --offload-arch=gfx90a gate_harness2.f90 libgemm_big_f.a -o gate_harness2
```
exit status: 0
stderr:
```
ld.lld: warning: <unknown>:0:0: in function __keep_alive void (): local memory global used by non-kernel function


```

**run fortran per-point harness** (in gate-hip-afar/file_loaded)

```
$ ./gate_harness2
```
exit status: 0
stdout:
```
  5.0699097988606456E-01
  5.0709649319119932E-01
  5.0907971274422026E-01
  5.0763811665286485E-01
  5.0533406255129398E-01
  5.0926026868495122E-01
  5.0841744622090557E-01
  5.0399858539811104E-01
TIMING   4.6752870000000000E+00

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
4.675 ns per point

### infer_batch harness: device-resident data


**compile and link hip infer_batch harness (device compiler)** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc gate_harness3.cu -Lhip_lib -lgemm_big -o gate_harness3_dev
```
exit status: 0
stderr:
```
gate_harness3.cu:65:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   65 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:69:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:69:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:69:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:69:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                                             ^~~~~~~~~~~~
5 warnings generated when compiling for gfx90a.
gate_harness3.cu:65:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   65 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:69:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:69:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:69:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:69:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                                             ^~~~~~~~~~~~
5 warnings generated when compiling for host.

```

**run hip infer_batch harness** (in gate-hip-afar/file_loaded)

```
$ ./gate_harness3_dev
```
exit status: 0
stdout:
```
5.06990979886064563e-01 
5.07096493191199316e-01 
5.09079712744220259e-01 
5.07638116652864846e-01 
5.05334062551293983e-01 
5.09260268684951223e-01 
5.08417446220905567e-01 
5.03998585398111043e-01 
TIMING 6.764579

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
6.765 ns per point

#### rocprof check: hipMemcpy count inside the roctx-scoped infer_batch call (ruling R15)


**probe for <rocprofiler-sdk-roctx/roctx.h>** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc -c gate_marker_probe.cu -o gate_marker_probe.o
```
exit status: 0

**compile roctx-bracketed infer_batch harness** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc -DROSENNA_GATE_MARKERS=1 gate_harness3.cu -Lhip_lib -lgemm_big -lrocprofiler-sdk-roctx -o gate_harness3_roctx
```
exit status: 0
stderr:
```
gate_harness3.cu:65:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   65 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:69:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:69:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:69:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:69:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                                             ^~~~~~~~~~~~
5 warnings generated when compiling for gfx90a.
gate_harness3.cu:65:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   65 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:69:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:69:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:69:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:69:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   69 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                                             ^~~~~~~~~~~~
5 warnings generated when compiling for host.

```

**rocprofv3 --hip-trace --marker-trace -f csv** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/rocprofv3 --hip-trace --marker-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof -o prof -- ./gate_harness3_roctx
```
exit status: 0
stdout:
```
5.06990979886064563e-01 
5.07096493191199316e-01 
5.09079712744220259e-01 
5.07638116652864846e-01 
5.05334062551293983e-01 
5.09260268684951223e-01 
5.08417446220905567e-01 
5.03998585398111043e-01 
TIMING 7.532587

```
stderr:
```
W20260914 18:29:53.650217 140609835399616 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.002760 sec
W20260914 18:29:53.652074 140609835399616 tool.cpp:2693] HIP (compiler) version 7.2.0 initialized (instance=0)
W20260914 18:29:53.652202 140609835399616 simple_timer.cpp:55] [rocprofv3] './gate_harness3_roctx' ::     0.000000 sec
W20260914 18:29:53.653632 140609835399616 tool.cpp:2693] HIP (runtime) version 7.2.0 initialized (instance=0)
W20260914 18:29:53.695199 140609835399616 tool.cpp:2693] HSA version 8.20.0 initialized (instance=0)
W20260914 18:29:53.784754 140609835399616 tool.cpp:2693] MARKER (ROCTx) version 1.2.3 initialized (instance=0)
W20260914 18:29:53.792123 140609835399616 simple_timer.cpp:55] [rocprofv3] './gate_harness3_roctx' ::     0.139921 sec
E20260914 18:29:53.823831 140609835399616 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof/prof_hip_api_trace.csv
E20260914 18:29:53.845970 140609835399616 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof/prof_marker_api_trace.csv
E20260914 18:29:53.849494 140609835399616 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof/prof_agent_info.csv
W20260914 18:29:53.853504 140609835399616 simple_timer.cpp:55] [rocprofv3] output generation ::     0.058800 sec
W20260914 18:29:53.853620 140609835399616 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.059012 sec

```
hipMemcpy* calls inside the roctx-scoped infer_batch call, from hip_api_trace cut to marker_api_trace: 0

## result

PASS: every configuration matched.
