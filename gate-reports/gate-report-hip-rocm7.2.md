
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
TIMING 1.921535

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
1.922 ns per point

### fortran harness: per-point infer via target teams distribute parallel do


**compile fortran per-point harness** (in gate-hip/embedded)

```
$ amdflang -O2 -fopenmp --offload-arch=gfx90a gate_harness2.F90 libgemm_big_f.a -o gate_harness2
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
TIMING   4.4257867500000003E+00

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
4.426 ns per point

### infer_batch harness: device-resident data


**compile and link hip infer_batch harness (device compiler)** (in gate-hip/embedded)

```
$ hipcc gate_harness3.cu -Lhip_lib -lgemm_big -o gate_harness3_dev
```
exit status: 0
stderr:
```
gate_harness3.cu:62:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   62 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:70:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:70:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:70:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:70:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                                             ^~~~~~~~~~~~
5 warnings generated when compiling for gfx90a.
gate_harness3.cu:62:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   62 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:70:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:70:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:70:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:70:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
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
TIMING 3.983233

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
3.983 ns per point

#### rocprofv3 check: transfers inside the marker-scoped 4-step loop, every harness (ruling R15)


**probe for <rocprofiler-sdk-roctx/roctx.h>** (in gate-hip/embedded)

```
$ hipcc -c gate_marker_probe.cu -o gate_marker_probe.o
```
exit status: 0

**compile marker-bracketed c per-point harness** (in gate-hip/embedded)

```
$ amdclang -O2 -fopenmp --offload-arch=gfx90a -DROSENNA_GATE_MARKERS=1 gate_harness1.c -lm -L/opt/rocm-7.2.0/lib -I/opt/rocm-7.2.0/include -lrocprofiler-sdk-roctx -o gate_harness1_prof
```
exit status: 0

**rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv (c per-point)** (in gate-hip/embedded)

```
$ /opt/rocm-7.2.0/bin/rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof_gate_harness1_prof -o prof -- ./gate_harness1_prof
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
TIMING 1.974225

```
stderr:
```
W20260914 18:47:08.705681 139883954177600 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.001495 sec
W20260914 18:47:08.744062 139883954177600 tool.cpp:2422] HSA version 8.20.0 initialized (instance=0)
W20260914 18:47:08.760643 139883954177600 simple_timer.cpp:55] [rocprofv3] './gate_harness1_prof' ::     0.000000 sec
W20260914 18:47:08.801274 139883954177600 tool.cpp:2422] MARKER (ROCTx) version 1.1.0 initialized (instance=0)
W20260914 18:47:08.814045 139883954177600 simple_timer.cpp:55] [rocprofv3] './gate_harness1_prof' ::     0.053402 sec
E20260914 18:47:08.838254 139883954177600 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof_gate_harness1_prof/prof_memory_copy_trace.csv
E20260914 18:47:08.855646 139883954177600 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof_gate_harness1_prof/prof_marker_api_trace.csv
E20260914 18:47:08.859014 139883954177600 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof_gate_harness1_prof/prof_agent_info.csv
W20260914 18:47:08.862262 139883954177600 simple_timer.cpp:55] [rocprofv3] output generation ::     0.045898 sec
W20260914 18:47:08.862312 139883954177600 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.046176 sec

```
c per-point: hipMemcpy* API calls plus MEMORY_COPY operations inside the roctx-scoped 4-step loop: 0

**compile marker-bracketed fortran per-point harness** (in gate-hip/embedded)

```
$ amdflang -O2 -fopenmp --offload-arch=gfx90a -DROSENNA_GATE_MARKERS=1 gate_harness2.F90 libgemm_big_f.a -L/opt/rocm-7.2.0/lib -I/opt/rocm-7.2.0/include -lrocprofiler-sdk-roctx -o gate_harness2_prof
```
exit status: 0

**rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv (fortran per-point)** (in gate-hip/embedded)

```
$ /opt/rocm-7.2.0/bin/rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof_gate_harness2_prof -o prof -- ./gate_harness2_prof
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
TIMING   4.4295562500000001E+00

```
stderr:
```
W20260914 18:47:19.260406 140655708210752 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.001345 sec
W20260914 18:47:19.299042 140655708210752 tool.cpp:2422] HSA version 8.20.0 initialized (instance=0)
W20260914 18:47:19.315988 140655708210752 simple_timer.cpp:55] [rocprofv3] './gate_harness2_prof' ::     0.000000 sec
W20260914 18:47:19.357467 140655708210752 tool.cpp:2422] MARKER (ROCTx) version 1.1.0 initialized (instance=0)
E20260914 18:47:19.427758 140655708210752 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof_gate_harness2_prof/prof_memory_copy_trace.csv
E20260914 18:47:19.446586 140655708210752 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof_gate_harness2_prof/prof_marker_api_trace.csv
E20260914 18:47:19.450081 140655708210752 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof_gate_harness2_prof/prof_agent_info.csv
W20260914 18:47:19.453700 140655708210752 simple_timer.cpp:55] [rocprofv3] output generation ::     0.045266 sec
W20260914 18:47:19.453755 140655708210752 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.045641 sec

```
fortran per-point: hipMemcpy* API calls plus MEMORY_COPY operations inside the roctx-scoped 4-step loop: 0

**compile marker-bracketed hip infer_batch harness** (in gate-hip/embedded)

```
$ hipcc -DROSENNA_GATE_MARKERS=1 gate_harness3.cu -Lhip_lib -lgemm_big -L/opt/rocm-7.2.0/lib -I/opt/rocm-7.2.0/include -lrocprofiler-sdk-roctx -o gate_harness3_prof
```
exit status: 0
stderr:
```
gate_harness3.cu:62:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   62 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:70:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:70:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:70:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:70:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                                             ^~~~~~~~~~~~
5 warnings generated when compiling for gfx90a.
gate_harness3.cu:62:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   62 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:70:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:70:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:70:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:70:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                                             ^~~~~~~~~~~~
5 warnings generated when compiling for host.

```

**rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv (hip infer_batch)** (in gate-hip/embedded)

```
$ /opt/rocm-7.2.0/bin/rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof_gate_harness3_prof -o prof -- ./gate_harness3_prof
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
TIMING 4.367487

```
stderr:
```
W20260914 18:47:21.624639 140120713844352 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.001243 sec
W20260914 18:47:21.624919 140120713844352 tool.cpp:2422] HIP (compiler) version 7.2.0 initialized (instance=0)
W20260914 18:47:21.625014 140120713844352 simple_timer.cpp:55] [rocprofv3] './gate_harness3_prof' ::     0.000000 sec
W20260914 18:47:21.625517 140120713844352 tool.cpp:2422] HIP (runtime) version 7.2.0 initialized (instance=0)
W20260914 18:47:21.661552 140120713844352 tool.cpp:2422] HSA version 8.20.0 initialized (instance=0)
W20260914 18:47:21.742778 140120713844352 tool.cpp:2422] MARKER (ROCTx) version 1.1.0 initialized (instance=0)
W20260914 18:47:21.760195 140120713844352 simple_timer.cpp:55] [rocprofv3] './gate_harness3_prof' ::     0.135180 sec
E20260914 18:47:21.789335 140120713844352 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof_gate_harness3_prof/prof_hip_api_trace.csv
E20260914 18:47:21.815574 140120713844352 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof_gate_harness3_prof/prof_memory_copy_trace.csv
E20260914 18:47:21.830816 140120713844352 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof_gate_harness3_prof/prof_marker_api_trace.csv
E20260914 18:47:21.834091 140120713844352 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/embedded/gate_rocprof_gate_harness3_prof/prof_agent_info.csv
W20260914 18:47:21.839711 140120713844352 simple_timer.cpp:55] [rocprofv3] output generation ::     0.076088 sec
W20260914 18:47:21.839770 140120713844352 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.076268 sec

```
hip infer_batch: hipMemcpy* API calls plus MEMORY_COPY operations inside the roctx-scoped 4-step loop: 0

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
TIMING 5.702019

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
5.702 ns per point

### fortran harness: per-point infer via target teams distribute parallel do


**compile fortran per-point harness** (in gate-hip/file_loaded)

```
$ amdflang -O2 -fopenmp --offload-arch=gfx90a gate_harness2.F90 libgemm_big_f.a -o gate_harness2
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
TIMING   4.4202190000000003E+00

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
4.420 ns per point

### infer_batch harness: device-resident data


**compile and link hip infer_batch harness (device compiler)** (in gate-hip/file_loaded)

```
$ hipcc gate_harness3.cu -Lhip_lib -lgemm_big -o gate_harness3_dev
```
exit status: 0
stderr:
```
gate_harness3.cu:62:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   62 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:70:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:70:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:70:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:70:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                                             ^~~~~~~~~~~~
5 warnings generated when compiling for gfx90a.
gate_harness3.cu:62:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   62 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:70:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:70:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:70:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:70:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
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
TIMING 6.626196

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
6.626 ns per point

#### rocprofv3 check: transfers inside the marker-scoped 4-step loop, every harness (ruling R15)


**probe for <rocprofiler-sdk-roctx/roctx.h>** (in gate-hip/file_loaded)

```
$ hipcc -c gate_marker_probe.cu -o gate_marker_probe.o
```
exit status: 0

**compile marker-bracketed c per-point harness** (in gate-hip/file_loaded)

```
$ amdclang -O2 -fopenmp --offload-arch=gfx90a -DROSENNA_GATE_MARKERS=1 gate_harness1.c libgemm_big.a -lm -L/opt/rocm-7.2.0/lib -I/opt/rocm-7.2.0/include -lrocprofiler-sdk-roctx -o gate_harness1_prof
```
exit status: 0

**rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv (c per-point)** (in gate-hip/file_loaded)

```
$ /opt/rocm-7.2.0/bin/rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof_gate_harness1_prof -o prof -- ./gate_harness1_prof
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
TIMING 5.333483

```
stderr:
```
W20260914 18:47:50.816552 140068266169920 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.001315 sec
W20260914 18:47:50.855524 140068266169920 tool.cpp:2422] HSA version 8.20.0 initialized (instance=0)
W20260914 18:47:50.872213 140068266169920 simple_timer.cpp:55] [rocprofv3] './gate_harness1_prof' ::     0.000000 sec
W20260914 18:47:50.911687 140068266169920 tool.cpp:2422] MARKER (ROCTx) version 1.1.0 initialized (instance=0)
W20260914 18:47:50.938650 140068266169920 simple_timer.cpp:55] [rocprofv3] './gate_harness1_prof' ::     0.066437 sec
E20260914 18:47:50.962704 140068266169920 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof_gate_harness1_prof/prof_memory_copy_trace.csv
E20260914 18:47:50.979683 140068266169920 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof_gate_harness1_prof/prof_marker_api_trace.csv
E20260914 18:47:50.982919 140068266169920 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof_gate_harness1_prof/prof_agent_info.csv
W20260914 18:47:50.986467 140068266169920 simple_timer.cpp:55] [rocprofv3] output generation ::     0.045601 sec
W20260914 18:47:50.986536 140068266169920 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.045849 sec

```
c per-point: hipMemcpy* API calls plus MEMORY_COPY operations inside the roctx-scoped 4-step loop: 0

**compile marker-bracketed fortran per-point harness** (in gate-hip/file_loaded)

```
$ amdflang -O2 -fopenmp --offload-arch=gfx90a -DROSENNA_GATE_MARKERS=1 gate_harness2.F90 libgemm_big_f.a -L/opt/rocm-7.2.0/lib -I/opt/rocm-7.2.0/include -lrocprofiler-sdk-roctx -o gate_harness2_prof
```
exit status: 0

**rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv (fortran per-point)** (in gate-hip/file_loaded)

```
$ /opt/rocm-7.2.0/bin/rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof_gate_harness2_prof -o prof -- ./gate_harness2_prof
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
TIMING   4.4264029999999996E+00

```
stderr:
```
W20260914 18:48:02.118137 140237198599744 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.001734 sec
W20260914 18:48:02.157715 140237198599744 tool.cpp:2422] HSA version 8.20.0 initialized (instance=0)
W20260914 18:48:02.174755 140237198599744 simple_timer.cpp:55] [rocprofv3] './gate_harness2_prof' ::     0.000000 sec
W20260914 18:48:02.215388 140237198599744 tool.cpp:2422] MARKER (ROCTx) version 1.1.0 initialized (instance=0)
E20260914 18:48:02.287078 140237198599744 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof_gate_harness2_prof/prof_memory_copy_trace.csv
E20260914 18:48:02.306351 140237198599744 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof_gate_harness2_prof/prof_marker_api_trace.csv
E20260914 18:48:02.309929 140237198599744 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof_gate_harness2_prof/prof_agent_info.csv
W20260914 18:48:02.313380 140237198599744 simple_timer.cpp:55] [rocprofv3] output generation ::     0.045870 sec
W20260914 18:48:02.313435 140237198599744 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.046232 sec

```
fortran per-point: hipMemcpy* API calls plus MEMORY_COPY operations inside the roctx-scoped 4-step loop: 0

**compile marker-bracketed hip infer_batch harness** (in gate-hip/file_loaded)

```
$ hipcc -DROSENNA_GATE_MARKERS=1 gate_harness3.cu -Lhip_lib -lgemm_big -L/opt/rocm-7.2.0/lib -I/opt/rocm-7.2.0/include -lrocprofiler-sdk-roctx -o gate_harness3_prof
```
exit status: 0
stderr:
```
gate_harness3.cu:62:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   62 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:70:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:70:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:70:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:70:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                                             ^~~~~~~~~~~~
5 warnings generated when compiling for gfx90a.
gate_harness3.cu:62:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   62 |     hipDeviceSynchronize();
      |     ^~~~~~~~~~~~~~~~~~~~~~
gate_harness3.cu:70:5: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |     ^~~~~~~~~~~
gate_harness3.cu:70:18: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                  ^~~~~~~~~~~
gate_harness3.cu:70:31: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                               ^~~~~~~~~~~~
gate_harness3.cu:70:45: warning: ignoring return value of type 'hipError_t' declared with 'nodiscard' attribute [-Wunused-value]
   70 |     hipFree(dx); hipFree(dy); hipFree(dxt); hipFree(dyt);
      |                                             ^~~~~~~~~~~~
5 warnings generated when compiling for host.

```

**rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv (hip infer_batch)** (in gate-hip/file_loaded)

```
$ /opt/rocm-7.2.0/bin/rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof_gate_harness3_prof -o prof -- ./gate_harness3_prof
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
TIMING 7.058132

```
stderr:
```
W20260914 18:48:04.578806 139992698074752 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.001621 sec
W20260914 18:48:04.579079 139992698074752 tool.cpp:2422] HIP (compiler) version 7.2.0 initialized (instance=0)
W20260914 18:48:04.579182 139992698074752 simple_timer.cpp:55] [rocprofv3] './gate_harness3_prof' ::     0.000000 sec
W20260914 18:48:04.580276 139992698074752 tool.cpp:2422] HIP (runtime) version 7.2.0 initialized (instance=0)
W20260914 18:48:04.616776 139992698074752 tool.cpp:2422] HSA version 8.20.0 initialized (instance=0)
W20260914 18:48:04.701972 139992698074752 tool.cpp:2422] MARKER (ROCTx) version 1.1.0 initialized (instance=0)
W20260914 18:48:04.730144 139992698074752 simple_timer.cpp:55] [rocprofv3] './gate_harness3_prof' ::     0.150962 sec
E20260914 18:48:04.762466 139992698074752 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof_gate_harness3_prof/prof_hip_api_trace.csv
E20260914 18:48:04.792229 139992698074752 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof_gate_harness3_prof/prof_memory_copy_trace.csv
E20260914 18:48:04.808219 139992698074752 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof_gate_harness3_prof/prof_marker_api_trace.csv
E20260914 18:48:04.811686 139992698074752 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip/file_loaded/gate_rocprof_gate_harness3_prof/prof_agent_info.csv
W20260914 18:48:04.817535 139992698074752 simple_timer.cpp:55] [rocprofv3] output generation ::     0.083085 sec
W20260914 18:48:04.817625 139992698074752 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.083312 sec

```
hip infer_batch: hipMemcpy* API calls plus MEMORY_COPY operations inside the roctx-scoped 4-step loop: 0

## result

PASS: every configuration matched.
