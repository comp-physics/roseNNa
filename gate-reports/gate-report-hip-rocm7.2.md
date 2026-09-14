
# rosenna gpu-gate report

- model: gemm_big
- backend: hip
- host-fallback: False
- cc: amdclang
- fc: amdflang
- flags: '-fopenmp --offload-arch=gfx90a'
- devcc: hipcc
- devflags: ''
- OMP_TARGET_OFFLOAD=MANDATORY is set for every omp-backend harness: a machine with no working offload device must fail here, loudly, rather than silently pass by falling back to the host.

### toolchain

platform: Linux-5.14.0-611.54.1.el9_7.x86_64-x86_64-with-glibc2.34
cc (amdclang) --version:
```
AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/rocm-7.2.0/lib/llvm/bin
Configuration file: /opt/rocm-7.2.0/lib/llvm/bin/clang.cfg

```
fc (amdflang) --version:
```
AMD flang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/rocm-7.2.0/lib/llvm/bin
Configuration file: /opt/rocm-7.2.0/lib/llvm/bin/flang.cfg

```
devcc (hipcc) --version:
```
HIP version: 7.2.26015-fc0010cf6a
AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/rocm-7.2.0/lib/llvm/bin
Configuration file: /opt/rocm-7.2.0/lib/llvm/bin/clang++.cfg

```

## gemm_big: embedded


**build c library (embedded, backend=omp, host compiler: serves the per-point harness)** (in gate-hip/embedded)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=omp CC=amdclang CFLAGS=-O2 'ROSENNA_OFFLOAD_FLAGS=-fopenmp --offload-arch=gfx90a'
```
exit status: 0
stdout:
```
amdclang -O2 -fopenmp --offload-arch=gfx90a -c gemm_big.c -o gemm_big.o
ar rcs libgemm_big.a gemm_big.o

```

**build c library (embedded, backend=hip, device compiler, in hip_lib/: serves the infer_batch harness)** (in gate-hip/embedded/hip_lib)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=hip DEVCC=hipcc
```
exit status: 0
stdout:
```
hipcc -O2 -x hip -c gemm_big.c -o gemm_big.o
hipcc -O2 -x hip -c gemm_big_kernel.cu -o gemm_big_kernel.o
ar rcs libgemm_big.a gemm_big.o gemm_big_kernel.o

```

**build fortran library (embedded)** (in gate-hip/embedded)

```
$ make -f gemm_big_fortran.mk FC=amdflang FFLAGS=-O2 'ROSENNA_OFFLOAD_FLAGS=-fopenmp --offload-arch=gfx90a'
```
exit status: 0
stdout:
```
amdflang -O2 -fopenmp --offload-arch=gfx90a -c gemm_big_model.F90 -o gemm_big_model.o
ar rcs libgemm_big_f.a gemm_big_model.o

```

### c harness: per-point infer via target teams distribute parallel for


**compile c per-point harness (host compiler, host offload flags)** (in gate-hip/embedded)

```
$ amdclang -O2 -fopenmp --offload-arch=gfx90a -c gate_harness1.c -o gate_harness1.o
```
exit status: 0

**link c per-point harness (host compiler, host offload flags, omp-backend archive)** (in gate-hip/embedded)

```
$ amdclang -fopenmp --offload-arch=gfx90a gate_harness1.o -lm -o gate_harness1
```
exit status: 0

**run c per-point harness** (in gate-hip/embedded)

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
TIMING 2.184868

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
2.185 ns per point

### fortran harness: per-point infer via target teams distribute parallel do


**compile fortran per-point harness** (in gate-hip/embedded)

```
$ amdflang -O2 -fopenmp --offload-arch=gfx90a gate_harness2.f90 libgemm_big_f.a -o gate_harness2
```
exit status: 0

**run fortran per-point harness** (in gate-hip/embedded)

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
TIMING   4.7760780000000000E+00

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
4.776 ns per point

### infer_batch harness: device-resident data


**compile and link hip infer_batch harness (device compiler)** (in gate-hip/embedded)

```
$ hipcc gate_harness3.cu -Lhip_lib -lgemm_big -o gate_harness3_dev
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

**run hip infer_batch harness** (in gate-hip/embedded)

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
TIMING 4.049538

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
4.050 ns per point

#### rocprof check: hipMemcpy count inside the roctx-scoped infer_batch call (ruling R15)


**probe for <rocprofiler-sdk-roctx/roctx.h>** (in gate-hip/embedded)

```
$ hipcc -c gate_marker_probe.cu -o gate_marker_probe.o
```
exit status: 0

**compile roctx-bracketed infer_batch harness** (in gate-hip/embedded)

```
$ hipcc -DROSENNA_GATE_MARKERS=1 gate_harness3.cu -Lhip_lib -lgemm_big -lrocprofiler-sdk-roctx -o gate_harness3_roctx
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

**rocprofv3 --hip-trace --marker-trace -f csv** (in gate-hip/embedded)

```
$ /opt/rocm-7.2.0/bin/rocprofv3 --hip-trace --marker-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof -o prof -- ./gate_harness3_roctx
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
TIMING 4.352649

```
stderr:
```
W20260914 18:27:29.475729 140497654486656 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.000779 sec
W20260914 18:27:29.476021 140497654486656 tool.cpp:2422] HIP (compiler) version 7.2.0 initialized (instance=0)
W20260914 18:27:29.476114 140497654486656 simple_timer.cpp:55] [rocprofv3] './gate_harness3_roctx' ::     0.000000 sec
W20260914 18:27:29.476706 140497654486656 tool.cpp:2422] HIP (runtime) version 7.2.0 initialized (instance=0)
W20260914 18:27:29.514906 140497654486656 tool.cpp:2422] HSA version 8.20.0 initialized (instance=0)
W20260914 18:27:29.611987 140497654486656 tool.cpp:2422] MARKER (ROCTx) version 1.1.0 initialized (instance=0)
W20260914 18:27:29.616394 140497654486656 simple_timer.cpp:55] [rocprofv3] './gate_harness3_roctx' ::     0.140279 sec
E20260914 18:27:29.649879 140497654486656 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof/prof_hip_api_trace.csv
E20260914 18:27:29.674079 140497654486656 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof/prof_marker_api_trace.csv
E20260914 18:27:29.677703 140497654486656 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof/prof_agent_info.csv
W20260914 18:27:29.682427 140497654486656 simple_timer.cpp:55] [rocprofv3] output generation ::     0.064169 sec
W20260914 18:27:29.682481 140497654486656 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.064301 sec

```
hipMemcpy* calls inside the roctx-scoped infer_batch call, from hip_api_trace cut to marker_api_trace: 0

## gemm_big: file-loaded


**build c library (file-loaded, backend=omp, host compiler: serves the per-point harness)** (in gate-hip/file_loaded)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=omp CC=amdclang CFLAGS=-O2 'ROSENNA_OFFLOAD_FLAGS=-fopenmp --offload-arch=gfx90a'
```
exit status: 0
stdout:
```
amdclang -O2 -fopenmp --offload-arch=gfx90a -c gemm_big.c -o gemm_big.o
ar rcs libgemm_big.a gemm_big.o

```

**build c library (file-loaded, backend=hip, device compiler, in hip_lib/: serves the infer_batch harness)** (in gate-hip/file_loaded/hip_lib)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=hip DEVCC=hipcc
```
exit status: 0
stdout:
```
hipcc -O2 -x hip -c gemm_big.c -o gemm_big.o
hipcc -O2 -x hip -c gemm_big_kernel.cu -o gemm_big_kernel.o
ar rcs libgemm_big.a gemm_big.o gemm_big_kernel.o

```

**build fortran library (file-loaded)** (in gate-hip/file_loaded)

```
$ make -f gemm_big_fortran.mk FC=amdflang FFLAGS=-O2 'ROSENNA_OFFLOAD_FLAGS=-fopenmp --offload-arch=gfx90a'
```
exit status: 0
stdout:
```
amdflang -O2 -fopenmp --offload-arch=gfx90a -c gemm_big_model.F90 -o gemm_big_model.o
ar rcs libgemm_big_f.a gemm_big_model.o

```

### c harness: per-point infer via target teams distribute parallel for


**compile c per-point harness (host compiler, host offload flags)** (in gate-hip/file_loaded)

```
$ amdclang -O2 -fopenmp --offload-arch=gfx90a -c gate_harness1.c -o gate_harness1.o
```
exit status: 0

**link c per-point harness (host compiler, host offload flags, omp-backend archive)** (in gate-hip/file_loaded)

```
$ amdclang -fopenmp --offload-arch=gfx90a gate_harness1.o libgemm_big.a -lm -o gate_harness1
```
exit status: 0

**run c per-point harness** (in gate-hip/file_loaded)

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
TIMING 5.700111

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
5.700 ns per point

### fortran harness: per-point infer via target teams distribute parallel do


**compile fortran per-point harness** (in gate-hip/file_loaded)

```
$ amdflang -O2 -fopenmp --offload-arch=gfx90a gate_harness2.f90 libgemm_big_f.a -o gate_harness2
```
exit status: 0

**run fortran per-point harness** (in gate-hip/file_loaded)

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
TIMING   4.7707980000000001E+00

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
4.771 ns per point

### infer_batch harness: device-resident data


**compile and link hip infer_batch harness (device compiler)** (in gate-hip/file_loaded)

```
$ hipcc gate_harness3.cu -Lhip_lib -lgemm_big -o gate_harness3_dev
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

**run hip infer_batch harness** (in gate-hip/file_loaded)

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
TIMING 6.690259

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
6.690 ns per point

#### rocprof check: hipMemcpy count inside the roctx-scoped infer_batch call (ruling R15)


**probe for <rocprofiler-sdk-roctx/roctx.h>** (in gate-hip/file_loaded)

```
$ hipcc -c gate_marker_probe.cu -o gate_marker_probe.o
```
exit status: 0

**compile roctx-bracketed infer_batch harness** (in gate-hip/file_loaded)

```
$ hipcc -DROSENNA_GATE_MARKERS=1 gate_harness3.cu -Lhip_lib -lgemm_big -lrocprofiler-sdk-roctx -o gate_harness3_roctx
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

**rocprofv3 --hip-trace --marker-trace -f csv** (in gate-hip/file_loaded)

```
$ /opt/rocm-7.2.0/bin/rocprofv3 --hip-trace --marker-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof -o prof -- ./gate_harness3_roctx
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
TIMING 7.257869

```
stderr:
```
W20260914 18:27:58.726209 140176495990400 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.000878 sec
W20260914 18:27:58.726620 140176495990400 tool.cpp:2422] HIP (compiler) version 7.2.0 initialized (instance=0)
W20260914 18:27:58.726771 140176495990400 simple_timer.cpp:55] [rocprofv3] './gate_harness3_roctx' ::     0.000000 sec
W20260914 18:27:58.728055 140176495990400 tool.cpp:2422] HIP (runtime) version 7.2.0 initialized (instance=0)
W20260914 18:27:58.764890 140176495990400 tool.cpp:2422] HSA version 8.20.0 initialized (instance=0)
W20260914 18:27:58.853597 140176495990400 tool.cpp:2422] MARKER (ROCTx) version 1.1.0 initialized (instance=0)
W20260914 18:27:58.860643 140176495990400 simple_timer.cpp:55] [rocprofv3] './gate_harness3_roctx' ::     0.133872 sec
E20260914 18:27:58.899511 140176495990400 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof/prof_hip_api_trace.csv
E20260914 18:27:58.923588 140176495990400 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof/prof_marker_api_trace.csv
E20260914 18:27:58.926814 140176495990400 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof/prof_agent_info.csv
W20260914 18:27:58.930969 140176495990400 simple_timer.cpp:55] [rocprofv3] output generation ::     0.068663 sec
W20260914 18:27:58.931049 140176495990400 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.068793 sec

```
hipMemcpy* calls inside the roctx-scoped infer_batch call, from hip_api_trace cut to marker_api_trace: 0

## result

PASS: every configuration matched.
