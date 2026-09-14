
# rosenna gpu-gate report

- model: gemm_big
- backend: cuda
- host-fallback: False
- cc: nvc
- fc: nvfortran
- flags: '-mp=gpu -gpu=cc80'
- devcc: nvcc
- devflags: '-arch=sm_80'
- OMP_TARGET_OFFLOAD=MANDATORY is set for every omp-backend harness: a machine with no working offload device must fail here, loudly, rather than silently pass by falling back to the host.

### toolchain

platform: Linux-6.8.0-134-generic-x86_64-with-glibc2.39
cc (nvc) --version:
```

nvc 25.11-0 64-bit target on x86-64 Linux -tp icelake-server 
NVIDIA Compilers and Tools
Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

```
fc (nvfortran) --version:
```

nvfortran 25.11-0 64-bit target on x86-64 Linux -tp icelake-server 
NVIDIA Compilers and Tools
Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

```
devcc (nvcc) --version:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0

```

## gemm_big: embedded


**build c library (embedded, backend=omp, host compiler: serves the per-point harness)** (in gate-fin/embedded)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=omp CC=nvc CFLAGS=-O2 'ROSENNA_OFFLOAD_FLAGS=-mp=gpu -gpu=cc80'
```
exit status: 0
stdout:
```
nvc -O2 -mp=gpu -gpu=cc80 -c gemm_big.c -o gemm_big.o
ar rcs libgemm_big.a gemm_big.o

```

**build c library (embedded, backend=cuda, device compiler, in cuda_lib/: serves the infer_batch harness)** (in gate-fin/embedded/cuda_lib)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=cuda DEVCC=nvcc DEVFLAGS=-arch=sm_80
```
exit status: 0
stdout:
```
nvcc -arch=sm_80 -x cu -c gemm_big.c -o gemm_big.o
nvcc -arch=sm_80 -c gemm_big_kernel.cu -o gemm_big_kernel.o
ar rcs libgemm_big.a gemm_big.o gemm_big_kernel.o

```

**build fortran library (embedded)** (in gate-fin/embedded)

```
$ make -f gemm_big_fortran.mk FC=nvfortran FFLAGS=-O2 'ROSENNA_OFFLOAD_FLAGS=-mp=gpu -gpu=cc80'
```
exit status: 0
stdout:
```
nvfortran -O2 -mp=gpu -gpu=cc80 -c gemm_big_model.F90 -o gemm_big_model.o
ar rcs libgemm_big_f.a gemm_big_model.o

```

### c harness: per-point infer via target teams distribute parallel for


**compile c per-point harness (host compiler, host offload flags)** (in gate-fin/embedded)

```
$ nvc -O2 -mp=gpu -gpu=cc80 -c gate_harness1.c -o gate_harness1.o
```
exit status: 0

**link c per-point harness (host compiler, host offload flags, omp-backend archive)** (in gate-fin/embedded)

```
$ nvc -mp=gpu -gpu=cc80 gate_harness1.o -lm -o gate_harness1
```
exit status: 0

**run c per-point harness** (in gate-fin/embedded)

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
TIMING 1.726866

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
1.727 ns per point

### fortran harness: per-point infer via target teams distribute parallel do


**compile fortran per-point harness** (in gate-fin/embedded)

```
$ nvfortran -O2 -mp=gpu -gpu=cc80 gate_harness2.f90 libgemm_big_f.a -o gate_harness2
```
exit status: 0
stderr:
```
gate_harness2.f90:

```

**run fortran per-point harness** (in gate-fin/embedded)

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
TIMING   1.7527999999999999E+00

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
1.753 ns per point

### infer_batch harness: device-resident data


**compile and link cuda infer_batch harness (device compiler)** (in gate-fin/embedded)

```
$ nvcc -arch=sm_80 gate_harness3.cu cuda_lib/libgemm_big.a -o gate_harness3_dev
```
exit status: 0

**run cuda infer_batch harness** (in gate-fin/embedded)

```
$ ./gate_harness3_dev
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
TIMING 1.467631

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
1.468 ns per point

#### nsys check: cudaMemcpy count inside the nvtx-scoped infer_batch call (ruling R15)


**probe for <nvtx3/nvToolsExt.h>** (in gate-fin/embedded)

```
$ nvcc -arch=sm_80 -c gate_nvtx_probe.cu -o gate_nvtx_probe.o
```
exit status: 0

**compile nvtx-bracketed infer_batch harness (no explicit -lnvToolsExt)** (in gate-fin/embedded)

```
$ nvcc -arch=sm_80 -DROSENNA_GATE_NVTX=1 gate_harness3.cu cuda_lib/libgemm_big.a -o gate_harness3_nvtx
```
exit status: 0

**nsys profile --capture-range=nvtx --nvtx-capture=rosenna_timed --capture-range-end=stop --stats=true** (in gate-fin/embedded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys profile -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range=nvtx --nvtx-capture=rosenna_timed --capture-range-end=stop --stats=true --force-overwrite=true -o /fastscratch/sbryngelson3/roseNNa/gate-fin/embedded/gate_nsys_profile ./gate_harness3_nvtx
```
exit status: 0
stdout:
```
Capture range started in the application.
Capture range ended in the application.
Generating '/tmp/nsys-report-7df7.qdstrm'

[1/8] [0%                          ] gate_nsys_profile.nsys-repProcessing events...

[1/8] [0%                          ] gate_nsys_profile.nsys-rep
[1/8] [==========49%               ] gate_nsys_profile.nsys-rep
[1/8] [========================98% ] gate_nsys_profile.nsys-rep
[1/8] [========================100%] gate_nsys_profile.nsys-rep
[1/8] [========================100%] gate_nsys_profile.nsys-rep

[2/8] [0%                          ] gate_nsys_profile.sqlite
[2KProcessing 1036 events: 

[2/8] [1%                          ] gate_nsys_profile.sqlite
[2/8] [2%                          ] gate_nsys_profile.sqlite
[2/8] [3%                          ] gate_nsys_profile.sqlite
[2/8] [4%                          ] gate_nsys_profile.sqlite
[2/8] [5%                          ] gate_nsys_profile.sqlite
[2/8] [6%                          ] gate_nsys_profile.sqlite
[2/8] [7%                          ] gate_nsys_profile.sqlite
[2/8] [8%                          ] gate_nsys_profile.sqlite
[2/8] [9%                          ] gate_nsys_profile.sqlite
[2/8] [10%                         ] gate_nsys_profile.sqlite
[2/8] [11%                         ] gate_nsys_profile.sqlite
[2/8] [12%                         ] gate_nsys_profile.sqlite
[2/8] [13%                         ] gate_nsys_profile.sqlite
[2/8] [14%                         ] gate_nsys_profile.sqlite
[2/8] [=15%                        ] gate_nsys_profile.sqlite
[2/8] [=16%                        ] gate_nsys_profile.sqlite
[2/8] [=17%                        ] gate_nsys_profile.sqlite
[2/8] [==18%                       ] gate_nsys_profile.sqlite
[2/8] [==19%                       ] gate_nsys_profile.sqlite
[2/8] [==20%                       ] gate_nsys_profile.sqlite
[2/8] [==21%                       ] gate_nsys_profile.sqlite
[2/8] [===22%                      ] gate_nsys_profile.sqlite
[2/8] [===23%                      ] gate_nsys_profile.sqlite
[2/8] [===24%                      ] gate_nsys_profile.sqlite
[2/8] [====25%                     ] gate_nsys_profile.sqlite
[2/8] [====26%                     ] gate_nsys_profile.sqlite
[2/8] [====27%                     ] gate_nsys_profile.sqlite
[2/8] [====28%                     ] gate_nsys_profile.sqlite
[2/8] [=====29%                    ] gate_nsys_profile.sqlite
[2/8] [=====30%                    ] gate_nsys_profile.sqlite
[2/8] [=====31%                    ] gate_nsys_profile.sqlite
[2/8] [=====32%                    ] gate_nsys_profile.sqlite
[2/8] [======33%                   ] gate_nsys_profile.sqlite
[2/8] [======34%                   ] gate_nsys_profile.sqlite
[2/8] [======35%                   ] gate_nsys_profile.sqlite
[2/8] [=======36%                  ] gate_nsys_profile.sqlite
[2/8] [=======37%                  ] gate_nsys_profile.sqlite
[2/8] [=======38%                  ] gate_nsys_profile.sqlite
[2/8] [=======39%                  ] gate_nsys_profile.sqlite
[2/8] [========40%                 ] gate_nsys_profile.sqlite
[2/8] [========41%                 ] gate_nsys_profile.sqlite
[2/8] [========42%                 ] gate_nsys_profile.sqlite
[2/8] [=========43%                ] gate_nsys_profile.sqlite
[2/8] [=========44%                ] gate_nsys_profile.sqlite
[2/8] [=========45%                ] gate_nsys_profile.sqlite
[2/8] [=========46%                ] gate_nsys_profile.sqlite
[2/8] [==========47%               ] gate_nsys_profile.sqlite
[2/8] [==========48%               ] gate_nsys_profile.sqlite
[2/8] [==========49%               ] gate_nsys_profile.sqlite
[2/8] [===========50%              ] gate_nsys_profile.sqlite
[2/8] [===========51%              ] gate_nsys_profile.sqlite
[2/8] [===========52%              ] gate_nsys_profile.sqlite
[2/8] [===========53%              ] gate_nsys_profile.sqlite
[2/8] [============54%             ] gate_nsys_profile.sqlite
[2/8] [============55%             ] gate_nsys_profile.sqlite
[2/8] [============56%             ] gate_nsys_profile.sqlite
[2/8] [============57%             ] gate_nsys_profile.sqlite
[2/8] [=============58%            ] gate_nsys_profile.sqlite
[2/8] [=============59%            ] gate_nsys_profile.sqlite
[2/8] [=============60%            ] gate_nsys_profile.sqlite
[2/8] [==============61%           ] gate_nsys_profile.sqlite
[2/8] [==============62%           ] gate_nsys_profile.sqlite
[2/8] [==============63%           ] gate_nsys_profile.sqlite
[2/8] [==============64%           ] gate_nsys_profile.sqlite
[2/8] [===============65%          ] gate_nsys_profile.sqlite
[2/8] [===============66%          ] gate_nsys_profile.sqlite
[2/8] [===============67%          ] gate_nsys_profile.sqlite
[2/8] [================68%         ] gate_nsys_profile.sqlite
[2/8] [================69%         ] gate_nsys_profile.sqlite
[2/8] [================70%         ] gate_nsys_profile.sqlite
[2/8] [================71%         ] gate_nsys_profile.sqlite
[2/8] [=================72%        ] gate_nsys_profile.sqlite
[2/8] [=================73%        ] gate_nsys_profile.sqlite
[2/8] [=================74%        ] gate_nsys_profile.sqlite
[2/8] [==================75%       ] gate_nsys_profile.sqlite
[2/8] [==================76%       ] gate_nsys_profile.sqlite
[2/8] [==================77%       ] gate_nsys_profile.sqlite
[2/8] [==================78%       ] gate_nsys_profile.sqlite
[2/8] [===================79%      ] gate_nsys_profile.sqlite
[2/8] [===================80%      ] gate_nsys_profile.sqlite
[2/8] [===================81%      ] gate_nsys_profile.sqlite
[2/8] [===================82%      ] gate_nsys_profile.sqlite
[2/8] [====================83%     ] gate_nsys_profile.sqlite
[2/8] [====================84%     ] gate_nsys_profile.sqlite
[2/8] [====================85%     ] gate_nsys_profile.sqlite
[2/8] [=====================86%    ] gate_nsys_profile.sqlite
[2/8] [=====================87%    ] gate_nsys_profile.sqlite
[2/8] [=====================88%    ] gate_nsys_profile.sqlite
[2/8] [=====================89%    ] gate_nsys_profile.sqlite
[2/8] [======================90%   ] gate_nsys_profile.sqlite
[2/8] [======================91%   ] gate_nsys_profile.sqlite
[2/8] [======================92%   ] gate_nsys_profile.sqlite
[2/8] [=======================93%  ] gate_nsys_profile.sqlite
[2/8] [=======================94%  ] gate_nsys_profile.sqlite
[2/8] [=======================95%  ] gate_nsys_profile.sqlite
[2/8] [=======================96%  ] gate_nsys_profile.sqlite
[2/8] [========================97% ] gate_nsys_profile.sqlite
[2/8] [========================98% ] gate_nsys_profile.sqlite
[2/8] [========================99% ] gate_nsys_profile.sqlite
[2/8] [========================100%] gate_nsys_profile.sqlite
[2/8] [========================100%] gate_nsys_profile.sqlite
4.89140427905968345e-01 
4.89250434340573193e-01 
4.91835667410042032e-01 
4.89199836524719156e-01 
4.90754835217244723e-01 
4.91486185717204871e-01 
4.89628045762320718e-01 
4.89513920535771696e-01 
TIMING 6016.651713
[3/8] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)   Style       Range     
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------  --------------
    100.0          143,045          1  143,045.0  143,045.0   143,045   143,045          0.0  PushPop  :rosenna_timed

[4/8] Executing 'osrt_sum' stats report
[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)        Name      
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  ----------------
     95.5           88,330          1  88,330.0  88,330.0    88,330    88,330          0.0  cudaLaunchKernel
      4.5            4,115          1   4,115.0   4,115.0     4,115     4,115          0.0  cuKernelGetName 

[6/8] Executing 'cuda_gpu_kern_sum' stats report
[7/8] Executing 'cuda_gpu_mem_time_sum' stats report
[8/8] Executing 'cuda_gpu_mem_size_sum' stats report
Generated:
	/fastscratch/sbryngelson3/roseNNa/gate-fin/embedded/gate_nsys_profile.nsys-rep
	/fastscratch/sbryngelson3/roseNNa/gate-fin/embedded/gate_nsys_profile.sqlite

```
stderr:
```
WARNING: CPU IP/backtrace sampling not supported, disabling.
Try the 'nsys status --environment' command to learn more.

WARNING: CPU context switch tracing not supported, disabling.
Try the 'nsys status --environment' command to learn more.

SKIPPED: No data available.
SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-fin/embedded/gate_nsys_profile.sqlite does not contain CUDA kernel data.
SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-fin/embedded/gate_nsys_profile.sqlite does not contain GPU memory data.
SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-fin/embedded/gate_nsys_profile.sqlite does not contain GPU memory data.

```

**nsys stats -q --force-export=true --report cuda_api_sum --format csv** (in gate-fin/embedded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys stats -q --force-export=true --report cuda_api_sum --format csv /fastscratch/sbryngelson3/roseNNa/gate-fin/embedded/gate_nsys_profile.nsys-rep
```
exit status: 0
stdout:
```
Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
95.5,88330,1,88330.0,88330.0,88330,88330,0.0,cudaLaunchKernel
4.5,4115,1,4115.0,4115.0,4115,4115,0.0,cuKernelGetName


```
cudaMemcpy* Num Calls inside the nvtx-scoped infer_batch call, from cuda_api_sum: 0

## gemm_big: file-loaded


**build c library (file-loaded, backend=omp, host compiler: serves the per-point harness)** (in gate-fin/file_loaded)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=omp CC=nvc CFLAGS=-O2 'ROSENNA_OFFLOAD_FLAGS=-mp=gpu -gpu=cc80'
```
exit status: 0
stdout:
```
nvc -O2 -mp=gpu -gpu=cc80 -c gemm_big.c -o gemm_big.o
ar rcs libgemm_big.a gemm_big.o

```

**build c library (file-loaded, backend=cuda, device compiler, in cuda_lib/: serves the infer_batch harness)** (in gate-fin/file_loaded/cuda_lib)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=cuda DEVCC=nvcc DEVFLAGS=-arch=sm_80
```
exit status: 0
stdout:
```
nvcc -arch=sm_80 -x cu -c gemm_big.c -o gemm_big.o
nvcc -arch=sm_80 -c gemm_big_kernel.cu -o gemm_big_kernel.o
ar rcs libgemm_big.a gemm_big.o gemm_big_kernel.o

```

**build fortran library (file-loaded)** (in gate-fin/file_loaded)

```
$ make -f gemm_big_fortran.mk FC=nvfortran FFLAGS=-O2 'ROSENNA_OFFLOAD_FLAGS=-mp=gpu -gpu=cc80'
```
exit status: 0
stdout:
```
nvfortran -O2 -mp=gpu -gpu=cc80 -c gemm_big_model.F90 -o gemm_big_model.o
ar rcs libgemm_big_f.a gemm_big_model.o

```

### c harness: per-point infer via target teams distribute parallel for


**compile c per-point harness (host compiler, host offload flags)** (in gate-fin/file_loaded)

```
$ nvc -O2 -mp=gpu -gpu=cc80 -c gate_harness1.c -o gate_harness1.o
```
exit status: 0

**link c per-point harness (host compiler, host offload flags, omp-backend archive)** (in gate-fin/file_loaded)

```
$ nvc -mp=gpu -gpu=cc80 gate_harness1.o libgemm_big.a -lm -o gate_harness1
```
exit status: 0

**run c per-point harness** (in gate-fin/file_loaded)

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
TIMING 1.732826

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
1.733 ns per point

### fortran harness: per-point infer via target teams distribute parallel do


**compile fortran per-point harness** (in gate-fin/file_loaded)

```
$ nvfortran -O2 -mp=gpu -gpu=cc80 gate_harness2.f90 libgemm_big_f.a -o gate_harness2
```
exit status: 0
stderr:
```
gate_harness2.f90:

```

**run fortran per-point harness** (in gate-fin/file_loaded)

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
TIMING   1.7408999999999999E+00

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
1.741 ns per point

### infer_batch harness: device-resident data


**compile and link cuda infer_batch harness (device compiler)** (in gate-fin/file_loaded)

```
$ nvcc -arch=sm_80 gate_harness3.cu cuda_lib/libgemm_big.a -o gate_harness3_dev
```
exit status: 0

**run cuda infer_batch harness** (in gate-fin/file_loaded)

```
$ ./gate_harness3_dev
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
TIMING 1.514821

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
1.515 ns per point

#### nsys check: cudaMemcpy count inside the nvtx-scoped infer_batch call (ruling R15)


**probe for <nvtx3/nvToolsExt.h>** (in gate-fin/file_loaded)

```
$ nvcc -arch=sm_80 -c gate_nvtx_probe.cu -o gate_nvtx_probe.o
```
exit status: 0

**compile nvtx-bracketed infer_batch harness (no explicit -lnvToolsExt)** (in gate-fin/file_loaded)

```
$ nvcc -arch=sm_80 -DROSENNA_GATE_NVTX=1 gate_harness3.cu cuda_lib/libgemm_big.a -o gate_harness3_nvtx
```
exit status: 0

**nsys profile --capture-range=nvtx --nvtx-capture=rosenna_timed --capture-range-end=stop --stats=true** (in gate-fin/file_loaded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys profile -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range=nvtx --nvtx-capture=rosenna_timed --capture-range-end=stop --stats=true --force-overwrite=true -o /fastscratch/sbryngelson3/roseNNa/gate-fin/file_loaded/gate_nsys_profile ./gate_harness3_nvtx
```
exit status: 0
stdout:
```
Capture range started in the application.
Capture range ended in the application.
Generating '/tmp/nsys-report-9e4d.qdstrm'

[1/8] [0%                          ] gate_nsys_profile.nsys-repProcessing events...

[1/8] [0%                          ] gate_nsys_profile.nsys-rep
[1/8] [==========49%               ] gate_nsys_profile.nsys-rep
[1/8] [========================98% ] gate_nsys_profile.nsys-rep
[1/8] [========================100%] gate_nsys_profile.nsys-rep
[1/8] [========================100%] gate_nsys_profile.nsys-rep

[2/8] [0%                          ] gate_nsys_profile.sqlite
[2KProcessing 1034 events: 

[2/8] [1%                          ] gate_nsys_profile.sqlite
[2/8] [2%                          ] gate_nsys_profile.sqlite
[2/8] [3%                          ] gate_nsys_profile.sqlite
[2/8] [4%                          ] gate_nsys_profile.sqlite
[2/8] [5%                          ] gate_nsys_profile.sqlite
[2/8] [6%                          ] gate_nsys_profile.sqlite
[2/8] [7%                          ] gate_nsys_profile.sqlite
[2/8] [8%                          ] gate_nsys_profile.sqlite
[2/8] [9%                          ] gate_nsys_profile.sqlite
[2/8] [10%                         ] gate_nsys_profile.sqlite
[2/8] [11%                         ] gate_nsys_profile.sqlite
[2/8] [12%                         ] gate_nsys_profile.sqlite
[2/8] [13%                         ] gate_nsys_profile.sqlite
[2/8] [14%                         ] gate_nsys_profile.sqlite
[2/8] [=15%                        ] gate_nsys_profile.sqlite
[2/8] [=16%                        ] gate_nsys_profile.sqlite
[2/8] [=17%                        ] gate_nsys_profile.sqlite
[2/8] [==18%                       ] gate_nsys_profile.sqlite
[2/8] [==19%                       ] gate_nsys_profile.sqlite
[2/8] [==20%                       ] gate_nsys_profile.sqlite
[2/8] [==21%                       ] gate_nsys_profile.sqlite
[2/8] [===22%                      ] gate_nsys_profile.sqlite
[2/8] [===23%                      ] gate_nsys_profile.sqlite
[2/8] [===24%                      ] gate_nsys_profile.sqlite
[2/8] [====25%                     ] gate_nsys_profile.sqlite
[2/8] [====26%                     ] gate_nsys_profile.sqlite
[2/8] [====27%                     ] gate_nsys_profile.sqlite
[2/8] [====28%                     ] gate_nsys_profile.sqlite
[2/8] [=====29%                    ] gate_nsys_profile.sqlite
[2/8] [=====30%                    ] gate_nsys_profile.sqlite
[2/8] [=====31%                    ] gate_nsys_profile.sqlite
[2/8] [=====32%                    ] gate_nsys_profile.sqlite
[2/8] [======33%                   ] gate_nsys_profile.sqlite
[2/8] [======34%                   ] gate_nsys_profile.sqlite
[2/8] [======35%                   ] gate_nsys_profile.sqlite
[2/8] [=======36%                  ] gate_nsys_profile.sqlite
[2/8] [=======37%                  ] gate_nsys_profile.sqlite
[2/8] [=======38%                  ] gate_nsys_profile.sqlite
[2/8] [=======39%                  ] gate_nsys_profile.sqlite
[2/8] [========40%                 ] gate_nsys_profile.sqlite
[2/8] [========41%                 ] gate_nsys_profile.sqlite
[2/8] [========42%                 ] gate_nsys_profile.sqlite
[2/8] [=========43%                ] gate_nsys_profile.sqlite
[2/8] [=========44%                ] gate_nsys_profile.sqlite
[2/8] [=========45%                ] gate_nsys_profile.sqlite
[2/8] [=========46%                ] gate_nsys_profile.sqlite
[2/8] [==========47%               ] gate_nsys_profile.sqlite
[2/8] [==========48%               ] gate_nsys_profile.sqlite
[2/8] [==========49%               ] gate_nsys_profile.sqlite
[2/8] [===========50%              ] gate_nsys_profile.sqlite
[2/8] [===========51%              ] gate_nsys_profile.sqlite
[2/8] [===========52%              ] gate_nsys_profile.sqlite
[2/8] [===========53%              ] gate_nsys_profile.sqlite
[2/8] [============54%             ] gate_nsys_profile.sqlite
[2/8] [============55%             ] gate_nsys_profile.sqlite
[2/8] [============56%             ] gate_nsys_profile.sqlite
[2/8] [============57%             ] gate_nsys_profile.sqlite
[2/8] [=============58%            ] gate_nsys_profile.sqlite
[2/8] [=============59%            ] gate_nsys_profile.sqlite
[2/8] [=============60%            ] gate_nsys_profile.sqlite
[2/8] [==============61%           ] gate_nsys_profile.sqlite
[2/8] [==============62%           ] gate_nsys_profile.sqlite
[2/8] [==============63%           ] gate_nsys_profile.sqlite
[2/8] [==============64%           ] gate_nsys_profile.sqlite
[2/8] [===============65%          ] gate_nsys_profile.sqlite
[2/8] [===============66%          ] gate_nsys_profile.sqlite
[2/8] [===============67%          ] gate_nsys_profile.sqlite
[2/8] [================68%         ] gate_nsys_profile.sqlite
[2/8] [================69%         ] gate_nsys_profile.sqlite
[2/8] [================70%         ] gate_nsys_profile.sqlite
[2/8] [================71%         ] gate_nsys_profile.sqlite
[2/8] [=================72%        ] gate_nsys_profile.sqlite
[2/8] [=================73%        ] gate_nsys_profile.sqlite
[2/8] [=================74%        ] gate_nsys_profile.sqlite
[2/8] [==================75%       ] gate_nsys_profile.sqlite
[2/8] [==================76%       ] gate_nsys_profile.sqlite
[2/8] [==================77%       ] gate_nsys_profile.sqlite
[2/8] [==================78%       ] gate_nsys_profile.sqlite
[2/8] [===================79%      ] gate_nsys_profile.sqlite
[2/8] [===================80%      ] gate_nsys_profile.sqlite
[2/8] [===================81%      ] gate_nsys_profile.sqlite
[2/8] [===================82%      ] gate_nsys_profile.sqlite
[2/8] [====================83%     ] gate_nsys_profile.sqlite
[2/8] [====================84%     ] gate_nsys_profile.sqlite
[2/8] [====================85%     ] gate_nsys_profile.sqlite
[2/8] [=====================86%    ] gate_nsys_profile.sqlite
[2/8] [=====================87%    ] gate_nsys_profile.sqlite
[2/8] [=====================88%    ] gate_nsys_profile.sqlite
[2/8] [=====================89%    ] gate_nsys_profile.sqlite
[2/8] [======================90%   ] gate_nsys_profile.sqlite
[2/8] [======================91%   ] gate_nsys_profile.sqlite
[2/8] [======================92%   ] gate_nsys_profile.sqlite
[2/8] [=======================93%  ] gate_nsys_profile.sqlite
[2/8] [=======================94%  ] gate_nsys_profile.sqlite
[2/8] [=======================95%  ] gate_nsys_profile.sqlite
[2/8] [=======================96%  ] gate_nsys_profile.sqlite
[2/8] [========================97% ] gate_nsys_profile.sqlite
[2/8] [========================98% ] gate_nsys_profile.sqlite
[2/8] [========================99% ] gate_nsys_profile.sqlite
[2/8] [========================100%] gate_nsys_profile.sqlite
[2/8] [========================100%] gate_nsys_profile.sqlite
4.89140427905968345e-01 
4.89250434340573193e-01 
4.91835667410042032e-01 
4.89199836524719156e-01 
4.90754835217244723e-01 
4.91486185717204871e-01 
4.89628045762320718e-01 
4.89513920535771696e-01 
TIMING 6219.770709
[3/8] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)   Style       Range     
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------  --------------
    100.0          183,168          1  183,168.0  183,168.0   183,168   183,168          0.0  PushPop  :rosenna_timed

[4/8] Executing 'osrt_sum' stats report
[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)        Name      
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------
     96.9          127,214          1  127,214.0  127,214.0   127,214   127,214          0.0  cudaLaunchKernel
      3.1            4,116          1    4,116.0    4,116.0     4,116     4,116          0.0  cuKernelGetName 

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                       Name                     
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------
    100.0        1,445,578          1  1,445,578.0  1,445,578.0  1,445,578  1,445,578          0.0  gemm_big_kernel(int, const double *, double *)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report
[8/8] Executing 'cuda_gpu_mem_size_sum' stats report
Generated:
	/fastscratch/sbryngelson3/roseNNa/gate-fin/file_loaded/gate_nsys_profile.nsys-rep
	/fastscratch/sbryngelson3/roseNNa/gate-fin/file_loaded/gate_nsys_profile.sqlite

```
stderr:
```
WARNING: CPU IP/backtrace sampling not supported, disabling.
Try the 'nsys status --environment' command to learn more.

WARNING: CPU context switch tracing not supported, disabling.
Try the 'nsys status --environment' command to learn more.

SKIPPED: No data available.
SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-fin/file_loaded/gate_nsys_profile.sqlite does not contain GPU memory data.
SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-fin/file_loaded/gate_nsys_profile.sqlite does not contain GPU memory data.

```

**nsys stats -q --force-export=true --report cuda_api_sum --format csv** (in gate-fin/file_loaded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys stats -q --force-export=true --report cuda_api_sum --format csv /fastscratch/sbryngelson3/roseNNa/gate-fin/file_loaded/gate_nsys_profile.nsys-rep
```
exit status: 0
stdout:
```
Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
96.9,127214,1,127214.0,127214.0,127214,127214,0.0,cudaLaunchKernel
3.1,4116,1,4116.0,4116.0,4116,4116,0.0,cuKernelGetName


```
cudaMemcpy* Num Calls inside the nvtx-scoped infer_batch call, from cuda_api_sum: 0

## result

PASS: every configuration matched.
