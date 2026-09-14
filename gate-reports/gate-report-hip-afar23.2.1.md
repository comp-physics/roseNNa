
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
TIMING 3.117800

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
3.118 ns per point

### fortran harness: per-point infer via target teams distribute parallel do


**compile fortran per-point harness** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdflang -O2 -fopenmp --offload-arch=gfx90a gate_harness2.F90 libgemm_big_f.a -o gate_harness2
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
TIMING   4.3886992500000002E+00

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
4.389 ns per point

### infer_batch harness: device-resident data


**compile and link hip infer_batch harness (device compiler)** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc gate_harness3.cu -Lhip_lib -lgemm_big -o gate_harness3_dev
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
TIMING 3.991087

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
3.991 ns per point

#### rocprofv3 check: transfers inside the marker-scoped 4-step loop, every harness (ruling R15)


**probe for <rocprofiler-sdk-roctx/roctx.h>** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc -c gate_marker_probe.cu -o gate_marker_probe.o
```
exit status: 0

**compile marker-bracketed c per-point harness** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdclang -O2 -fopenmp --offload-arch=gfx90a -DROSENNA_GATE_MARKERS=1 gate_harness1.c -lm -L/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/lib -I/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/include -lrocprofiler-sdk-roctx -o gate_harness1_prof
```
exit status: 0

**rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv (c per-point)** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof_gate_harness1_prof -o prof -- ./gate_harness1_prof
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
TIMING 2.026796

```
stderr:
```
W20260914 18:49:03.729322 140227785853376 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.004049 sec
W20260914 18:49:03.782225 140227785853376 tool.cpp:2693] HSA version 8.20.0 initialized (instance=0)
W20260914 18:49:03.800850 140227785853376 simple_timer.cpp:55] [rocprofv3] './gate_harness1_prof' ::     0.000000 sec
W20260914 18:49:03.865925 140227785853376 tool.cpp:2693] MARKER (ROCTx) version 1.2.3 initialized (instance=0)
W20260914 18:49:03.882189 140227785853376 simple_timer.cpp:55] [rocprofv3] './gate_harness1_prof' ::     0.081339 sec
E20260914 18:49:03.906540 140227785853376 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof_gate_harness1_prof/prof_memory_copy_trace.csv
E20260914 18:49:03.924291 140227785853376 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof_gate_harness1_prof/prof_marker_api_trace.csv
E20260914 18:49:03.927697 140227785853376 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof_gate_harness1_prof/prof_agent_info.csv
W20260914 18:49:03.931128 140227785853376 simple_timer.cpp:55] [rocprofv3] output generation ::     0.046586 sec
W20260914 18:49:03.931184 140227785853376 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.046823 sec

```
c per-point: hipMemcpy* API calls plus MEMORY_COPY operations inside the roctx-scoped 4-step loop: 0

**compile marker-bracketed fortran per-point harness** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdflang -O2 -fopenmp --offload-arch=gfx90a -DROSENNA_GATE_MARKERS=1 gate_harness2.F90 libgemm_big_f.a -L/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/lib -I/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/include -lrocprofiler-sdk-roctx -o gate_harness2_prof
```
exit status: 0
stderr:
```
ld.lld: warning: <unknown>:0:0: in function __keep_alive void (): local memory global used by non-kernel function


```

**rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv (fortran per-point)** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof_gate_harness2_prof -o prof -- ./gate_harness2_prof
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
TIMING   4.5770320000000000E+00

```
stderr:
```
W20260914 18:49:32.976246 140508978219456 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.003968 sec
W20260914 18:49:33.049459 140508978219456 tool.cpp:2693] HSA version 8.20.0 initialized (instance=0)
W20260914 18:49:33.068030 140508978219456 simple_timer.cpp:55] [rocprofv3] './gate_harness2_prof' ::     0.000000 sec
W20260914 18:49:33.114637 140508978219456 tool.cpp:2693] MARKER (ROCTx) version 1.2.3 initialized (instance=0)
E20260914 18:49:33.189916 140508978219456 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof_gate_harness2_prof/prof_memory_copy_trace.csv
E20260914 18:49:33.207321 140508978219456 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof_gate_harness2_prof/prof_marker_api_trace.csv
E20260914 18:49:33.211642 140508978219456 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof_gate_harness2_prof/prof_agent_info.csv
W20260914 18:49:33.215269 140508978219456 simple_timer.cpp:55] [rocprofv3] output generation ::     0.045435 sec
W20260914 18:49:33.215329 140508978219456 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.045669 sec

```
fortran per-point: hipMemcpy* API calls plus MEMORY_COPY operations inside the roctx-scoped 4-step loop: 0

**compile marker-bracketed hip infer_batch harness** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc -DROSENNA_GATE_MARKERS=1 gate_harness3.cu -Lhip_lib -lgemm_big -L/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/lib -I/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/include -lrocprofiler-sdk-roctx -o gate_harness3_prof
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

**rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv (hip infer_batch)** (in gate-hip-afar/embedded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof_gate_harness3_prof -o prof -- ./gate_harness3_prof
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
TIMING 4.546332

```
stderr:
```
W20260914 18:49:35.681255 140472742382016 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.004997 sec
W20260914 18:49:35.683250 140472742382016 tool.cpp:2693] HIP (compiler) version 7.2.0 initialized (instance=0)
W20260914 18:49:35.683331 140472742382016 simple_timer.cpp:55] [rocprofv3] './gate_harness3_prof' ::     0.000000 sec
W20260914 18:49:35.684406 140472742382016 tool.cpp:2693] HIP (runtime) version 7.2.0 initialized (instance=0)
W20260914 18:49:35.722258 140472742382016 tool.cpp:2693] HSA version 8.20.0 initialized (instance=0)
W20260914 18:49:35.803864 140472742382016 tool.cpp:2693] MARKER (ROCTx) version 1.2.3 initialized (instance=0)
W20260914 18:49:35.822094 140472742382016 simple_timer.cpp:55] [rocprofv3] './gate_harness3_prof' ::     0.138763 sec
E20260914 18:49:35.851882 140472742382016 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof_gate_harness3_prof/prof_hip_api_trace.csv
E20260914 18:49:35.884449 140472742382016 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof_gate_harness3_prof/prof_memory_copy_trace.csv
E20260914 18:49:35.901833 140472742382016 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof_gate_harness3_prof/prof_marker_api_trace.csv
E20260914 18:49:35.905189 140472742382016 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/embedded/gate_rocprof_gate_harness3_prof/prof_agent_info.csv
W20260914 18:49:35.911456 140472742382016 simple_timer.cpp:55] [rocprofv3] output generation ::     0.085674 sec
W20260914 18:49:35.911515 140472742382016 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.085865 sec

```
hip infer_batch: hipMemcpy* API calls plus MEMORY_COPY operations inside the roctx-scoped 4-step loop: 0

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
TIMING 5.824268

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
5.824 ns per point

### fortran harness: per-point infer via target teams distribute parallel do


**compile fortran per-point harness** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdflang -O2 -fopenmp --offload-arch=gfx90a gate_harness2.F90 libgemm_big_f.a -o gate_harness2
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
TIMING   4.3751034999999998E+00

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
4.375 ns per point

### infer_batch harness: device-resident data


**compile and link hip infer_batch harness (device compiler)** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc gate_harness3.cu -Lhip_lib -lgemm_big -o gate_harness3_dev
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
TIMING 6.513516

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
6.514 ns per point

#### rocprofv3 check: transfers inside the marker-scoped 4-step loop, every harness (ruling R15)


**probe for <rocprofiler-sdk-roctx/roctx.h>** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc -c gate_marker_probe.cu -o gate_marker_probe.o
```
exit status: 0

**compile marker-bracketed c per-point harness** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdclang -O2 -fopenmp --offload-arch=gfx90a -DROSENNA_GATE_MARKERS=1 gate_harness1.c libgemm_big.a -lm -L/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/lib -I/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/include -lrocprofiler-sdk-roctx -o gate_harness1_prof
```
exit status: 0
stderr:
```
ld.lld: warning: <unknown>:0:0: in function __keep_alive void (): local memory global used by non-kernel function


```

**rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv (c per-point)** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof_gate_harness1_prof -o prof -- ./gate_harness1_prof
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
TIMING 3.804266

```
stderr:
```
W20260914 18:50:24.849485 139978296252864 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.003800 sec
W20260914 18:50:24.900050 139978296252864 tool.cpp:2693] HSA version 8.20.0 initialized (instance=0)
W20260914 18:50:24.919155 139978296252864 simple_timer.cpp:55] [rocprofv3] './gate_harness1_prof' ::     0.000000 sec
W20260914 18:50:24.967140 139978296252864 tool.cpp:2693] MARKER (ROCTx) version 1.2.3 initialized (instance=0)
W20260914 18:50:24.989294 139978296252864 simple_timer.cpp:55] [rocprofv3] './gate_harness1_prof' ::     0.070139 sec
E20260914 18:50:25.014390 139978296252864 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof_gate_harness1_prof/prof_memory_copy_trace.csv
E20260914 18:50:25.031902 139978296252864 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof_gate_harness1_prof/prof_marker_api_trace.csv
E20260914 18:50:25.035279 139978296252864 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof_gate_harness1_prof/prof_agent_info.csv
W20260914 18:50:25.038784 139978296252864 simple_timer.cpp:55] [rocprofv3] output generation ::     0.046761 sec
W20260914 18:50:25.038845 139978296252864 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.047033 sec

```
c per-point: hipMemcpy* API calls plus MEMORY_COPY operations inside the roctx-scoped 4-step loop: 0

**compile marker-bracketed fortran per-point harness** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/amdflang -O2 -fopenmp --offload-arch=gfx90a -DROSENNA_GATE_MARKERS=1 gate_harness2.F90 libgemm_big_f.a -L/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/lib -I/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/include -lrocprofiler-sdk-roctx -o gate_harness2_prof
```
exit status: 0
stderr:
```
ld.lld: warning: <unknown>:0:0: in function __keep_alive void (): local memory global used by non-kernel function


```

**rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv (fortran per-point)** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof_gate_harness2_prof -o prof -- ./gate_harness2_prof
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
TIMING   4.3886667499999996E+00

```
stderr:
```
W20260914 18:50:52.471899 139875084802496 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.003950 sec
W20260914 18:50:52.522429 139875084802496 tool.cpp:2693] HSA version 8.20.0 initialized (instance=0)
W20260914 18:50:52.540120 139875084802496 simple_timer.cpp:55] [rocprofv3] './gate_harness2_prof' ::     0.000000 sec
W20260914 18:50:52.588741 139875084802496 tool.cpp:2693] MARKER (ROCTx) version 1.2.3 initialized (instance=0)
E20260914 18:50:52.662956 139875084802496 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof_gate_harness2_prof/prof_memory_copy_trace.csv
E20260914 18:50:52.680791 139875084802496 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof_gate_harness2_prof/prof_marker_api_trace.csv
E20260914 18:50:52.685233 139875084802496 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof_gate_harness2_prof/prof_agent_info.csv
W20260914 18:50:52.689442 139875084802496 simple_timer.cpp:55] [rocprofv3] output generation ::     0.047139 sec
W20260914 18:50:52.689497 139875084802496 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.047491 sec

```
fortran per-point: hipMemcpy* API calls plus MEMORY_COPY operations inside the roctx-scoped 4-step loop: 0

**compile marker-bracketed hip infer_batch harness** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/hipcc -DROSENNA_GATE_MARKERS=1 gate_harness3.cu -Lhip_lib -lgemm_big -L/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/lib -I/work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/include -lrocprofiler-sdk-roctx -o gate_harness3_prof
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

**rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv (hip infer_batch)** (in gate-hip-afar/file_loaded)

```
$ /work1/spencerbryngelson/sbryngelson/software/therock-afar-23.2.1-gfx90a-7.13.0-7357b5084b/bin/rocprofv3 --hip-trace --marker-trace --memory-copy-trace -f csv -d /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof_gate_harness3_prof -o prof -- ./gate_harness3_prof
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
TIMING 7.551397

```
stderr:
```
W20260914 18:50:54.893868 139754316601792 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.004541 sec
W20260914 18:50:54.896050 139754316601792 tool.cpp:2693] HIP (compiler) version 7.2.0 initialized (instance=0)
W20260914 18:50:54.896139 139754316601792 simple_timer.cpp:55] [rocprofv3] './gate_harness3_prof' ::     0.000000 sec
W20260914 18:50:54.897376 139754316601792 tool.cpp:2693] HIP (runtime) version 7.2.0 initialized (instance=0)
W20260914 18:50:54.942224 139754316601792 tool.cpp:2693] HSA version 8.20.0 initialized (instance=0)
W20260914 18:50:55.028708 139754316601792 tool.cpp:2693] MARKER (ROCTx) version 1.2.3 initialized (instance=0)
W20260914 18:50:55.058906 139754316601792 simple_timer.cpp:55] [rocprofv3] './gate_harness3_prof' ::     0.162767 sec
E20260914 18:50:55.087826 139754316601792 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof_gate_harness3_prof/prof_hip_api_trace.csv
E20260914 18:50:55.137775 139754316601792 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof_gate_harness3_prof/prof_memory_copy_trace.csv
E20260914 18:50:55.153227 139754316601792 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof_gate_harness3_prof/prof_marker_api_trace.csv
E20260914 18:50:55.156646 139754316601792 output_stream.cpp:111] Opened result file: /work1/spencerbryngelson/sbryngelson/rosenna/gate-hip-afar/file_loaded/gate_rocprof_gate_harness3_prof/prof_agent_info.csv
W20260914 18:50:55.162279 139754316601792 simple_timer.cpp:55] [rocprofv3] output generation ::     0.099238 sec
W20260914 18:50:55.162337 139754316601792 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.099414 sec

```
hip infer_batch: hipMemcpy* API calls plus MEMORY_COPY operations inside the roctx-scoped 4-step loop: 0

## result

PASS: every configuration matched.
