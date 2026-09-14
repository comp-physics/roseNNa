
# rosenna gpu-gate report

- model: gemm_big
- backend: omp
- host-fallback: True
- cc: gcc
- fc: gfortran
- flags: '-fopenmp'
- OMP_TARGET_OFFLOAD=MANDATORY is NOT set (host-fallback mode): this run exercises the omp-backend contract end to end on a machine with no accelerator, the same host-fallback contract tests/test_device_c.py and tests/test_device_fortran.py already cover.

### toolchain

platform: Linux-6.8.0-134-generic-x86_64-with-glibc2.39
cc (gcc) --version:
```
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


```
fc (gfortran) --version:
```
GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


```

## gemm_big: embedded


**build c library (embedded, backend=omp, host compiler: serves the per-point harness)** (in gate-hfin/embedded)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=omp CC=gcc 'CFLAGS=-O2 -Wall -Wextra -std=c11' ROSENNA_OFFLOAD_FLAGS=-fopenmp
```
exit status: 0
stdout:
```
gcc -O2 -Wall -Wextra -std=c11 -fopenmp -c gemm_big.c -o gemm_big.o
ar rcs libgemm_big.a gemm_big.o

```

**build fortran library (embedded)** (in gate-hfin/embedded)

```
$ make -f gemm_big_fortran.mk FC=gfortran 'FFLAGS=-O2 -Wall -Wextra -std=f2008' ROSENNA_OFFLOAD_FLAGS=-fopenmp
```
exit status: 0
stdout:
```
gfortran -O2 -Wall -Wextra -std=f2008 -fopenmp -c gemm_big_model.F90 -o gemm_big_model.o
ar rcs libgemm_big_f.a gemm_big_model.o

```

### c harness: per-point infer via target teams distribute parallel for


**compile c per-point harness (host compiler, host offload flags)** (in gate-hfin/embedded)

```
$ gcc -O2 -Wall -Wextra -std=c11 -fopenmp -c gate_harness1.c -o gate_harness1.o
```
exit status: 0

**link c per-point harness (host compiler, host offload flags, omp-backend archive)** (in gate-hfin/embedded)

```
$ gcc -fopenmp gate_harness1.o -lm -o gate_harness1
```
exit status: 0

**run c per-point harness** (in gate-hfin/embedded)

```
$ ./gate_harness1
```
exit status: 0
stdout:
```
4.89140427905968345e-01 
4.89250434340573193e-01 
4.91835667410042032e-01 
4.89199836524719156e-01 
4.90754835217244723e-01 
4.91486185717204871e-01 
4.89628045762320718e-01 
4.89513920535771696e-01 
TIMING 76.058495

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
76.058 ns per point

### fortran harness: per-point infer via target teams distribute parallel do


**compile fortran per-point harness** (in gate-hfin/embedded)

```
$ gfortran -O2 -Wall -Wextra -std=f2008 -fopenmp gate_harness2.f90 libgemm_big_f.a -o gate_harness2
```
exit status: 0

**run fortran per-point harness** (in gate-hfin/embedded)

```
$ ./gate_harness2
```
exit status: 0
stdout:
```
  4.8914042790596834E-01
  4.8925043434057319E-01
  4.9183566741004203E-01
  4.8919983652471916E-01
  4.9075483521724472E-01
  4.9148618571720487E-01
  4.8962804576232072E-01
  4.8951392053577170E-01
TIMING   6.6311374000000001E+01

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
66.311 ns per point

### infer_batch harness: device-resident data


**compile c infer_batch harness (omp)** (in gate-hfin/embedded)

```
$ gcc -O2 -Wall -Wextra -std=c11 -fopenmp gate_harness3.c libgemm_big.a -lm -o gate_harness3
```
exit status: 0

**run c infer_batch harness (omp)** (in gate-hfin/embedded)

```
$ ./gate_harness3
```
exit status: 0
stdout:
```
4.89140427905968345e-01 
4.89250434340573193e-01 
4.91835667410042032e-01 
4.89199836524719156e-01 
4.90754835217244723e-01 
4.91486185717204871e-01 
4.89628045762320718e-01 
4.89513920535771696e-01 
TIMING 66.210914

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
66.211 ns per point

**compile fortran infer_batch harness (omp)** (in gate-hfin/embedded)

```
$ gfortran -O2 -Wall -Wextra -std=f2008 -fopenmp gate_harness3.f90 libgemm_big_f.a -o gate_harness3_f
```
exit status: 0

**run fortran infer_batch harness (omp)** (in gate-hfin/embedded)

```
$ ./gate_harness3_f
```
exit status: 0
stdout:
```
  4.8914042790596834E-01
  4.8925043434057319E-01
  4.9183566741004203E-01
  4.8919983652471916E-01
  4.9075483521724472E-01
  4.9148618571720487E-01
  4.8962804576232072E-01
  4.8951392053577170E-01
TIMING   4.4482532999999997E+01

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
44.483 ns per point

## gemm_big: file-loaded


**build c library (file-loaded, backend=omp, host compiler: serves the per-point harness)** (in gate-hfin/file_loaded)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=omp CC=gcc 'CFLAGS=-O2 -Wall -Wextra -std=c11' ROSENNA_OFFLOAD_FLAGS=-fopenmp
```
exit status: 0
stdout:
```
gcc -O2 -Wall -Wextra -std=c11 -fopenmp -c gemm_big.c -o gemm_big.o
ar rcs libgemm_big.a gemm_big.o

```

**build fortran library (file-loaded)** (in gate-hfin/file_loaded)

```
$ make -f gemm_big_fortran.mk FC=gfortran 'FFLAGS=-O2 -Wall -Wextra -std=f2008' ROSENNA_OFFLOAD_FLAGS=-fopenmp
```
exit status: 0
stdout:
```
gfortran -O2 -Wall -Wextra -std=f2008 -fopenmp -c gemm_big_model.F90 -o gemm_big_model.o
ar rcs libgemm_big_f.a gemm_big_model.o

```

### c harness: per-point infer via target teams distribute parallel for


**compile c per-point harness (host compiler, host offload flags)** (in gate-hfin/file_loaded)

```
$ gcc -O2 -Wall -Wextra -std=c11 -fopenmp -c gate_harness1.c -o gate_harness1.o
```
exit status: 0

**link c per-point harness (host compiler, host offload flags, omp-backend archive)** (in gate-hfin/file_loaded)

```
$ gcc -fopenmp gate_harness1.o libgemm_big.a -lm -o gate_harness1
```
exit status: 0

**run c per-point harness** (in gate-hfin/file_loaded)

```
$ ./gate_harness1
```
exit status: 0
stdout:
```
4.89140427905968345e-01 
4.89250434340573193e-01 
4.91835667410042032e-01 
4.89199836524719156e-01 
4.90754835217244723e-01 
4.91486185717204871e-01 
4.89628045762320718e-01 
4.89513920535771696e-01 
TIMING 60.614542

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
60.615 ns per point

### fortran harness: per-point infer via target teams distribute parallel do


**compile fortran per-point harness** (in gate-hfin/file_loaded)

```
$ gfortran -O2 -Wall -Wextra -std=f2008 -fopenmp gate_harness2.f90 libgemm_big_f.a -o gate_harness2
```
exit status: 0

**run fortran per-point harness** (in gate-hfin/file_loaded)

```
$ ./gate_harness2
```
exit status: 0
stdout:
```
  4.8914042790596834E-01
  4.8925043434057319E-01
  4.9183566741004203E-01
  4.8919983652471916E-01
  4.9075483521724472E-01
  4.9148618571720487E-01
  4.8962804576232072E-01
  4.8951392053577170E-01
TIMING   5.7342905000000002E+01

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
57.343 ns per point

### infer_batch harness: device-resident data


**compile c infer_batch harness (omp)** (in gate-hfin/file_loaded)

```
$ gcc -O2 -Wall -Wextra -std=c11 -fopenmp gate_harness3.c libgemm_big.a -lm -o gate_harness3
```
exit status: 0

**run c infer_batch harness (omp)** (in gate-hfin/file_loaded)

```
$ ./gate_harness3
```
exit status: 0
stdout:
```
4.89140427905968345e-01 
4.89250434340573193e-01 
4.91835667410042032e-01 
4.89199836524719156e-01 
4.90754835217244723e-01 
4.91486185717204871e-01 
4.89628045762320718e-01 
4.89513920535771696e-01 
TIMING 49.503203

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
49.503 ns per point

**compile fortran infer_batch harness (omp)** (in gate-hfin/file_loaded)

```
$ gfortran -O2 -Wall -Wextra -std=f2008 -fopenmp gate_harness3.f90 libgemm_big_f.a -o gate_harness3_f
```
exit status: 0

**run fortran infer_batch harness (omp)** (in gate-hfin/file_loaded)

```
$ ./gate_harness3_f
```
exit status: 0
stdout:
```
  4.8914042790596834E-01
  4.8925043434057319E-01
  4.9183566741004203E-01
  4.8919983652471916E-01
  4.9075483521724472E-01
  4.9148618571720487E-01
  4.8962804576232072E-01
  4.8951392053577170E-01
TIMING   4.5717359999999999E+01

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
45.717 ns per point

## result

PASS: every configuration matched.
