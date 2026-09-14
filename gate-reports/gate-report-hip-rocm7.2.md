
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
TIMING 1.981974

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
1.982 ns per point

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
TIMING   4.7304709999999996E+00

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
4.730 ns per point

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
TIMING 4.035491

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
4.035 ns per point

#### nsys check: cudaMemcpy count inside the nvtx-scoped infer_batch call (ruling R15)

nsys check skipped: --backend hip (rocprof scoping of the call is a follow-up; nsys/nvtx are CUDA-only).

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
TIMING 5.674124

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
5.674 ns per point

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
TIMING   4.6185400000000003E+00

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
4.619 ns per point

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
TIMING 6.619985

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
6.620 ns per point

#### nsys check: cudaMemcpy count inside the nvtx-scoped infer_batch call (ruling R15)

nsys check skipped: --backend hip (rocprof scoping of the call is a follow-up; nsys/nvtx are CUDA-only).

## result

PASS: every configuration matched.
