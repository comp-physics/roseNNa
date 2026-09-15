
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


**build c library (embedded, backend=omp, host compiler: serves the per-point harness)** (in gate-cur/embedded)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=omp CC=nvc CFLAGS=-O2 'ROSENNA_OFFLOAD_FLAGS=-mp=gpu -gpu=cc80'
```
exit status: 0
stdout:
```
nvc -O2 -mp=gpu -gpu=cc80 -c gemm_big.c -o gemm_big.o
ar rcs libgemm_big.a gemm_big.o

```

**build c library (embedded, backend=cuda, device compiler, in cuda_lib/: serves the infer_batch harness)** (in gate-cur/embedded/cuda_lib)

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

**build fortran library (embedded)** (in gate-cur/embedded)

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


**compile c per-point harness (host compiler, host offload flags)** (in gate-cur/embedded)

```
$ nvc -O2 -mp=gpu -gpu=cc80 -c gate_harness1.c -o gate_harness1.o
```
exit status: 0

**link c per-point harness (host compiler, host offload flags, omp-backend archive)** (in gate-cur/embedded)

```
$ nvc -mp=gpu -gpu=cc80 gate_harness1.o -lm -o gate_harness1
```
exit status: 0

**run c per-point harness** (in gate-cur/embedded)

```
$ ./gate_harness1
```
exit status: 0
stdout:
```
5.36466048695884545e-01 
5.36039895567142044e-01 
5.40751061319241666e-01 
5.36212892349821391e-01 
5.39204062915815352e-01 
5.38379088865937105e-01 
5.36743434583053403e-01 
5.38699451026247611e-01 
TIMING 1.703262

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
1.703 ns per point

### fortran harness: per-point infer via target teams distribute parallel do


**compile fortran per-point harness** (in gate-cur/embedded)

```
$ nvfortran -O2 -mp=gpu -gpu=cc80 gate_harness2.F90 libgemm_big_f.a -o gate_harness2
```
exit status: 0
stderr:
```
gate_harness2.F90:

```

**run fortran per-point harness** (in gate-cur/embedded)

```
$ ./gate_harness2
```
exit status: 0
stdout:
```
  5.3646604869588455E-01
  5.3603989556714204E-01
  5.4075106131924167E-01
  5.3621289234982139E-01
  5.3920406291581535E-01
  5.3837908886593711E-01
  5.3674343458305340E-01
  5.3869945102624761E-01
TIMING   1.7359750000000000E+00

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
1.736 ns per point

### infer_batch harness: device-resident data


**compile and link cuda infer_batch harness (device compiler)** (in gate-cur/embedded)

```
$ nvcc -arch=sm_80 gate_harness3.cu -Lcuda_lib -lgemm_big -o gate_harness3_dev
```
exit status: 0

**run cuda infer_batch harness** (in gate-cur/embedded)

```
$ ./gate_harness3_dev
```
exit status: 0
stdout:
```
5.36466048695884545e-01 
5.36039895567142044e-01 
5.40751061319241666e-01 
5.36212892349821391e-01 
5.39204062915815352e-01 
5.38379088865937105e-01 
5.36743434583053403e-01 
5.38699451026247611e-01 
TIMING 1.428440

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
1.428 ns per point

#### nsys check: transfers inside the marker-scoped 4-step loop, every harness (ruling R15)


**probe for <nvtx3/nvToolsExt.h>** (in gate-cur/embedded)

```
$ nvcc -arch=sm_80 -c gate_marker_probe.cu -o gate_marker_probe.o
```
exit status: 0

**compile marker-bracketed c per-point harness** (in gate-cur/embedded)

```
$ nvc -O2 -mp=gpu -gpu=cc80 -DROSENNA_GATE_MARKERS=1 gate_harness1.c -lm -o gate_harness1_prof
```
exit status: 0

**nsys profile --capture-range=nvtx --nvtx-capture=rosenna_timed (c per-point)** (in gate-cur/embedded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys profile -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range=nvtx --nvtx-capture=rosenna_timed --capture-range-end=stop --stats=true --force-overwrite=true -o /fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness1_prof ./gate_harness1_prof
```
exit status: 0
stdout:
```
Capture range started in the application.
Capture range ended in the application.
Generating '/tmp/nsys-report-8ead.qdstrm'

[1/8] [0%                          ] gate_nsys_gate_harness1_prof.nsys-repProcessing events...

[1/8] [0%                          ] gate_nsys_gate_harness1_prof.nsys-rep
[1/8] [==========49%               ] gate_nsys_gate_harness1_prof.nsys-rep
[1/8] [========================98% ] gate_nsys_gate_harness1_prof.nsys-rep
[1/8] [========================100%] gate_nsys_gate_harness1_prof.nsys-rep
[1/8] [========================100%] gate_nsys_gate_harness1_prof.nsys-rep

[2/8] [0%                          ] gate_nsys_gate_harness1_prof.sqlite
[2KProcessing 1026 events: 

[2/8] [1%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [2%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [3%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [4%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [5%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [6%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [7%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [8%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [9%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [10%                         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [11%                         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [12%                         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [13%                         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [14%                         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=15%                        ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=16%                        ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=17%                        ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==18%                       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==19%                       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==20%                       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==21%                       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===22%                      ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===23%                      ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===24%                      ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [====25%                     ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [====26%                     ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [====27%                     ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [====28%                     ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====29%                    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====30%                    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====31%                    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====32%                    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [======33%                   ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [======34%                   ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [======35%                   ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======36%                  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======37%                  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======38%                  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======39%                  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========40%                 ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========41%                 ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========42%                 ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=========43%                ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=========44%                ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=========45%                ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=========46%                ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==========47%               ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==========48%               ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==========49%               ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===========50%              ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===========51%              ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===========52%              ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===========53%              ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [============54%             ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [============55%             ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [============56%             ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [============57%             ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=============58%            ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=============59%            ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=============60%            ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==============61%           ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==============62%           ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==============63%           ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==============64%           ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===============65%          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===============66%          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===============67%          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [================68%         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [================69%         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [================70%         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [================71%         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=================72%        ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=================73%        ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=================74%        ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==================75%       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==================76%       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==================77%       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==================78%       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===================79%      ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===================80%      ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===================81%      ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===================82%      ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [====================83%     ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [====================84%     ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [====================85%     ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====================86%    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====================87%    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====================88%    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====================89%    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [======================90%   ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [======================91%   ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [======================92%   ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======================93%  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======================94%  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======================95%  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======================96%  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========================97% ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========================98% ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========================99% ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========================100%] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========================100%] gate_nsys_gate_harness1_prof.sqlite
5.36466048695884545e-01 
5.36039895567142044e-01 
5.40751061319241666e-01 
5.36212892349821391e-01 
5.39204062915815352e-01 
5.38379088865937105e-01 
5.36743434583053403e-01 
5.38699451026247611e-01 
TIMING 1.726985
[3/8] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Style       Range     
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------  --------------
    100.0        6,922,657          1  6,922,657.0  6,922,657.0  6,922,657  6,922,657          0.0  PushPop  :rosenna_timed

[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)  Name
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----
    100.0       10,070,499          1  10,070,499.0  10,070,499.0  10,070,499  10,070,499          0.0  poll

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)         Name        
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------------------
     99.1        6,780,408          4  1,695,102.0  1,695,219.0  1,694,500  1,695,470        441.1  cuStreamSynchronize
      0.9           64,403          4     16,100.8      7,472.5      6,205     43,253     18,111.6  cuLaunchKernel     

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)          Name         
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ---------------------
    100.0        6,767,700          4  1,691,925.0  1,690,909.0  1,689,869  1,696,013      2,778.0  nvkernel_main_F1L50_4

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report
[8/8] Executing 'cuda_gpu_mem_size_sum' stats report
Generated:
	/fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness1_prof.nsys-rep
	/fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness1_prof.sqlite

```
stderr:
```
WARNING: CPU IP/backtrace sampling not supported, disabling.
Try the 'nsys status --environment' command to learn more.

WARNING: CPU context switch tracing not supported, disabling.
Try the 'nsys status --environment' command to learn more.

SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness1_prof.sqlite does not contain GPU memory data.
SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness1_prof.sqlite does not contain GPU memory data.

```

**nsys stats -q --force-export=true --report cuda_api_sum --format csv** (in gate-cur/embedded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys stats -q --force-export=true --report cuda_api_sum --format csv /fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness1_prof.nsys-rep
```
exit status: 0
stdout:
```
Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
99.1,6780408,4,1695102.0,1695219.0,1694500,1695470,441.1,cuStreamSynchronize
0.9,64403,4,16100.8,7472.5,6205,43253,18111.6,cuLaunchKernel


```
c per-point: Memcpy API calls (cudaMemcpy*, cuMemcpy*) inside the nvtx-scoped 4-step loop, from cuda_api_sum: 0

**compile marker-bracketed fortran per-point harness** (in gate-cur/embedded)

```
$ nvfortran -O2 -mp=gpu -gpu=cc80 -DROSENNA_GATE_MARKERS=1 gate_harness2.F90 libgemm_big_f.a -o gate_harness2_prof
```
exit status: 2
stderr:
```
gate_harness2.F90:
/usr/bin/ld: /tmp/nvfortran13LikFliIy9N_.o: in function `MAIN_':
/fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_harness2.F90:58:(.text+0x15d4): undefined reference to `nvtxRangePushA'
/usr/bin/ld: /fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_harness2.F90:64:(.text+0x163d): undefined reference to `nvtxRangePop'
pgacclnk: child process exit status 1: /usr/bin/ld

```

**compile marker-bracketed fortran per-point harness (retry: -lnvToolsExt)** (in gate-cur/embedded)

```
$ nvfortran -O2 -mp=gpu -gpu=cc80 -DROSENNA_GATE_MARKERS=1 gate_harness2.F90 libgemm_big_f.a -lnvToolsExt -o gate_harness2_prof
```
exit status: 2
stderr:
```
gate_harness2.F90:
/usr/bin/ld: cannot find -lnvToolsExt: No such file or directory
pgacclnk: child process exit status 1: /usr/bin/ld

```

**compile marker-bracketed fortran per-point harness (retry: -cudalib=nvtx)** (in gate-cur/embedded)

```
$ nvfortran -O2 -mp=gpu -gpu=cc80 -DROSENNA_GATE_MARKERS=1 gate_harness2.F90 libgemm_big_f.a -cudalib=nvtx -o gate_harness2_prof
```
exit status: 0
stderr:
```
gate_harness2.F90:

```

**nsys profile --capture-range=nvtx --nvtx-capture=rosenna_timed (fortran per-point)** (in gate-cur/embedded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys profile -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range=nvtx --nvtx-capture=rosenna_timed --capture-range-end=stop --stats=true --force-overwrite=true -o /fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness2_prof ./gate_harness2_prof
```
exit status: 0
stdout:
```
Capture range started in the application.
Capture range ended in the application.
Generating '/tmp/nsys-report-47c6.qdstrm'

[1/8] [0%                          ] gate_nsys_gate_harness2_prof.nsys-repProcessing events...

[1/8] [0%                          ] gate_nsys_gate_harness2_prof.nsys-rep
[1/8] [==========49%               ] gate_nsys_gate_harness2_prof.nsys-rep
[1/8] [========================98% ] gate_nsys_gate_harness2_prof.nsys-rep
[1/8] [========================100%] gate_nsys_gate_harness2_prof.nsys-rep
[1/8] [========================100%] gate_nsys_gate_harness2_prof.nsys-rep

[2/8] [0%                          ] gate_nsys_gate_harness2_prof.sqlite
[2KProcessing 1030 events: 

[2/8] [1%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [2%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [3%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [4%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [5%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [6%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [7%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [8%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [9%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [10%                         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [11%                         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [12%                         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [13%                         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [14%                         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=15%                        ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=16%                        ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=17%                        ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==18%                       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==19%                       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==20%                       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==21%                       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===22%                      ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===23%                      ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===24%                      ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [====25%                     ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [====26%                     ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [====27%                     ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [====28%                     ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====29%                    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====30%                    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====31%                    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====32%                    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [======33%                   ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [======34%                   ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [======35%                   ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======36%                  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======37%                  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======38%                  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======39%                  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========40%                 ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========41%                 ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========42%                 ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=========43%                ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=========44%                ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=========45%                ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=========46%                ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==========47%               ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==========48%               ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==========49%               ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===========50%              ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===========51%              ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===========52%              ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===========53%              ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [============54%             ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [============55%             ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [============56%             ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [============57%             ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=============58%            ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=============59%            ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=============60%            ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==============61%           ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==============62%           ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==============63%           ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==============64%           ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===============65%          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===============66%          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===============67%          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [================68%         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [================69%         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [================70%         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [================71%         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=================72%        ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=================73%        ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=================74%        ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==================75%       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==================76%       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==================77%       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==================78%       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===================79%      ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===================80%      ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===================81%      ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===================82%      ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [====================83%     ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [====================84%     ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [====================85%     ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====================86%    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====================87%    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====================88%    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====================89%    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [======================90%   ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [======================91%   ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [======================92%   ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======================93%  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======================94%  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======================95%  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======================96%  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========================97% ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========================98% ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========================99% ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========================100%] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========================100%] gate_nsys_gate_harness2_prof.sqlite
  5.3646604869588455E-01
  5.3603989556714204E-01
  5.4075106131924167E-01
  5.3621289234982139E-01
  5.3920406291581535E-01
  5.3837908886593711E-01
  5.3674343458305340E-01
  5.3869945102624761E-01
TIMING   1.7829999999999999E+00
[3/8] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Style       Range     
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------  --------------
    100.0        7,161,120          1  7,161,120.0  7,161,120.0  7,161,120  7,161,120          0.0  PushPop  :rosenna_timed

[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)  Name
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----
    100.0       10,067,784          1  10,067,784.0  10,067,784.0  10,067,784  10,067,784          0.0  poll

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)         Name        
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------------------
     98.8        6,909,218          4  1,727,304.5  1,730,659.0  1,713,709  1,734,191      9,273.1  cuStreamSynchronize
      1.2           86,093          4     21,523.3      7,190.0      4,799     66,914     30,287.3  cuLaunchKernel     

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                Name              
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  --------------------------------
    100.0        6,927,795          4  1,731,948.8  1,732,637.0  1,728,716  1,733,805      2,238.2  nvkernel_host_step_loop_F1L98_6_

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report
[8/8] Executing 'cuda_gpu_mem_size_sum' stats report
Generated:
	/fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness2_prof.nsys-rep
	/fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness2_prof.sqlite

```
stderr:
```
WARNING: CPU IP/backtrace sampling not supported, disabling.
Try the 'nsys status --environment' command to learn more.

WARNING: CPU context switch tracing not supported, disabling.
Try the 'nsys status --environment' command to learn more.

SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness2_prof.sqlite does not contain GPU memory data.
SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness2_prof.sqlite does not contain GPU memory data.

```

**nsys stats -q --force-export=true --report cuda_api_sum --format csv** (in gate-cur/embedded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys stats -q --force-export=true --report cuda_api_sum --format csv /fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness2_prof.nsys-rep
```
exit status: 0
stdout:
```
Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
98.8,6909218,4,1727304.5,1730659.0,1713709,1734191,9273.1,cuStreamSynchronize
1.2,86093,4,21523.3,7190.0,4799,66914,30287.3,cuLaunchKernel


```
fortran per-point: Memcpy API calls (cudaMemcpy*, cuMemcpy*) inside the nvtx-scoped 4-step loop, from cuda_api_sum: 0

**compile marker-bracketed cuda infer_batch harness** (in gate-cur/embedded)

```
$ nvcc -arch=sm_80 -DROSENNA_GATE_MARKERS=1 gate_harness3.cu -Lcuda_lib -lgemm_big -o gate_harness3_prof
```
exit status: 0

**nsys profile --capture-range=nvtx --nvtx-capture=rosenna_timed (cuda infer_batch)** (in gate-cur/embedded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys profile -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range=nvtx --nvtx-capture=rosenna_timed --capture-range-end=stop --stats=true --force-overwrite=true -o /fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness3_prof ./gate_harness3_prof
```
exit status: 0
stdout:
```
Capture range started in the application.
Capture range ended in the application.
Generating '/tmp/nsys-report-128e.qdstrm'

[1/8] [0%                          ] gate_nsys_gate_harness3_prof.nsys-repProcessing events...

[1/8] [0%                          ] gate_nsys_gate_harness3_prof.nsys-rep
[1/8] [==========48%               ] gate_nsys_gate_harness3_prof.nsys-rep
[1/8] [========================97% ] gate_nsys_gate_harness3_prof.nsys-rep
[1/8] [========================98% ] gate_nsys_gate_harness3_prof.nsys-rep
[1/8] [========================100%] gate_nsys_gate_harness3_prof.nsys-rep
[1/8] [========================100%] gate_nsys_gate_harness3_prof.nsys-rep

[2/8] [0%                          ] gate_nsys_gate_harness3_prof.sqlite
[2KProcessing 1014 events: 

[2/8] [1%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [2%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [3%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [4%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [5%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [6%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [7%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [8%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [9%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [10%                         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [11%                         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [12%                         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [13%                         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [14%                         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=15%                        ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=16%                        ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=17%                        ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==18%                       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==19%                       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==20%                       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==21%                       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===22%                      ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===23%                      ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===24%                      ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [====25%                     ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [====26%                     ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [====27%                     ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [====28%                     ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====29%                    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====30%                    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====31%                    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====32%                    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [======33%                   ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [======34%                   ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [======35%                   ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======36%                  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======37%                  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======38%                  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======39%                  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========40%                 ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========41%                 ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========42%                 ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=========43%                ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=========44%                ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=========45%                ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=========46%                ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==========47%               ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==========48%               ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==========49%               ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===========50%              ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===========51%              ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===========52%              ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===========53%              ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [============54%             ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [============55%             ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [============56%             ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [============57%             ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=============58%            ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=============59%            ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=============60%            ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==============61%           ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==============62%           ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==============63%           ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==============64%           ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===============65%          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===============66%          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===============67%          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [================68%         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [================69%         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [================70%         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [================71%         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=================72%        ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=================73%        ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=================74%        ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==================75%       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==================76%       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==================77%       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==================78%       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===================79%      ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===================80%      ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===================81%      ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===================82%      ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [====================83%     ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [====================84%     ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [====================85%     ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====================86%    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====================87%    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====================88%    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====================89%    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [======================90%   ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [======================91%   ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [======================92%   ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======================93%  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======================94%  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======================95%  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======================96%  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========================97% ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========================98% ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========================99% ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========================100%] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========================100%] gate_nsys_gate_harness3_prof.sqlite
5.36466048695884545e-01 
5.36039895567142044e-01 
5.40751061319241666e-01 
5.36212892349821391e-01 
5.39204062915815352e-01 
5.38379088865937105e-01 
5.36743434583053403e-01 
5.38699451026247611e-01 
TIMING 1587.285999
[3/8] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Style       Range     
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------  --------------
    100.0        5,860,370          1  5,860,370.0  5,860,370.0  5,860,370  5,860,370          0.0  PushPop  :rosenna_timed

[4/8] Executing 'osrt_sum' stats report
[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)          Name         
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ---------------------
     97.5        5,661,389          1  5,661,389.0  5,661,389.0  5,661,389  5,661,389          0.0  cudaDeviceSynchronize
      2.4          139,969          4     34,992.3     10,161.0      8,493    111,154     50,786.7  cudaLaunchKernel     
      0.1            5,336          4      1,334.0        194.5        126      4,821      2,325.3  cuKernelGetName      

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                       Name                     
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------
    100.0        5,697,579          4  1,424,394.8  1,424,538.5  1,421,259  1,427,243      2,470.2  gemm_big_kernel(int, const double *, double *)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report
[8/8] Executing 'cuda_gpu_mem_size_sum' stats report
Generated:
	/fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness3_prof.nsys-rep
	/fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness3_prof.sqlite

```
stderr:
```
WARNING: CPU IP/backtrace sampling not supported, disabling.
Try the 'nsys status --environment' command to learn more.

WARNING: CPU context switch tracing not supported, disabling.
Try the 'nsys status --environment' command to learn more.

SKIPPED: No data available.
SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness3_prof.sqlite does not contain GPU memory data.
SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness3_prof.sqlite does not contain GPU memory data.

```

**nsys stats -q --force-export=true --report cuda_api_sum --format csv** (in gate-cur/embedded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys stats -q --force-export=true --report cuda_api_sum --format csv /fastscratch/sbryngelson3/roseNNa/gate-cur/embedded/gate_nsys_gate_harness3_prof.nsys-rep
```
exit status: 0
stdout:
```
Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
97.5,5661389,1,5661389.0,5661389.0,5661389,5661389,0.0,cudaDeviceSynchronize
2.4,139969,4,34992.3,10161.0,8493,111154,50786.7,cudaLaunchKernel
0.1,5336,4,1334.0,194.5,126,4821,2325.3,cuKernelGetName


```
cuda infer_batch: Memcpy API calls (cudaMemcpy*, cuMemcpy*) inside the nvtx-scoped 4-step loop, from cuda_api_sum: 0

## gemm_big: file-loaded


**build c library (file-loaded, backend=omp, host compiler: serves the per-point harness)** (in gate-cur/file_loaded)

```
$ make -f gemm_big.mk ROSENNA_BACKEND=omp CC=nvc CFLAGS=-O2 'ROSENNA_OFFLOAD_FLAGS=-mp=gpu -gpu=cc80'
```
exit status: 0
stdout:
```
nvc -O2 -mp=gpu -gpu=cc80 -c gemm_big.c -o gemm_big.o
ar rcs libgemm_big.a gemm_big.o

```

**build c library (file-loaded, backend=cuda, device compiler, in cuda_lib/: serves the infer_batch harness)** (in gate-cur/file_loaded/cuda_lib)

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

**build fortran library (file-loaded)** (in gate-cur/file_loaded)

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


**compile c per-point harness (host compiler, host offload flags)** (in gate-cur/file_loaded)

```
$ nvc -O2 -mp=gpu -gpu=cc80 -c gate_harness1.c -o gate_harness1.o
```
exit status: 0

**link c per-point harness (host compiler, host offload flags, omp-backend archive)** (in gate-cur/file_loaded)

```
$ nvc -mp=gpu -gpu=cc80 gate_harness1.o libgemm_big.a -lm -o gate_harness1
```
exit status: 0

**run c per-point harness** (in gate-cur/file_loaded)

```
$ ./gate_harness1
```
exit status: 0
stdout:
```
5.36466048695884545e-01 
5.36039895567142044e-01 
5.40751061319241666e-01 
5.36212892349821391e-01 
5.39204062915815352e-01 
5.38379088865937105e-01 
5.36743434583053403e-01 
5.38699451026247611e-01 
TIMING 1.728773

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
1.729 ns per point

### fortran harness: per-point infer via target teams distribute parallel do


**compile fortran per-point harness** (in gate-cur/file_loaded)

```
$ nvfortran -O2 -mp=gpu -gpu=cc80 gate_harness2.F90 libgemm_big_f.a -o gate_harness2
```
exit status: 0
stderr:
```
gate_harness2.F90:

```

**run fortran per-point harness** (in gate-cur/file_loaded)

```
$ ./gate_harness2
```
exit status: 0
stdout:
```
  5.3646604869588455E-01
  5.3603989556714204E-01
  5.4075106131924167E-01
  5.3621289234982139E-01
  5.3920406291581535E-01
  5.3837908886593711E-01
  5.3674343458305340E-01
  5.3869945102624761E-01
TIMING   1.7415250000000000E+00

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
1.742 ns per point

### infer_batch harness: device-resident data


**compile and link cuda infer_batch harness (device compiler)** (in gate-cur/file_loaded)

```
$ nvcc -arch=sm_80 gate_harness3.cu -Lcuda_lib -lgemm_big -o gate_harness3_dev
```
exit status: 0

**run cuda infer_batch harness** (in gate-cur/file_loaded)

```
$ ./gate_harness3_dev
```
exit status: 0
stdout:
```
5.36466048695884545e-01 
5.36039895567142044e-01 
5.40751061319241666e-01 
5.36212892349821391e-01 
5.39204062915815352e-01 
5.38379088865937105e-01 
5.36743434583053403e-01 
5.38699451026247611e-01 
TIMING 1.469135

```
matches the onnxruntime reference (rtol=1e-05, atol=1e-06)
1.469 ns per point

#### nsys check: transfers inside the marker-scoped 4-step loop, every harness (ruling R15)


**probe for <nvtx3/nvToolsExt.h>** (in gate-cur/file_loaded)

```
$ nvcc -arch=sm_80 -c gate_marker_probe.cu -o gate_marker_probe.o
```
exit status: 0

**compile marker-bracketed c per-point harness** (in gate-cur/file_loaded)

```
$ nvc -O2 -mp=gpu -gpu=cc80 -DROSENNA_GATE_MARKERS=1 gate_harness1.c libgemm_big.a -lm -o gate_harness1_prof
```
exit status: 0
stderr:
```
gate_harness1.c:

```

**nsys profile --capture-range=nvtx --nvtx-capture=rosenna_timed (c per-point)** (in gate-cur/file_loaded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys profile -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range=nvtx --nvtx-capture=rosenna_timed --capture-range-end=stop --stats=true --force-overwrite=true -o /fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness1_prof ./gate_harness1_prof
```
exit status: 0
stdout:
```
Capture range started in the application.
Capture range ended in the application.
Generating '/tmp/nsys-report-f121.qdstrm'

[1/8] [0%                          ] gate_nsys_gate_harness1_prof.nsys-repProcessing events...

[1/8] [0%                          ] gate_nsys_gate_harness1_prof.nsys-rep
[1/8] [==========49%               ] gate_nsys_gate_harness1_prof.nsys-rep
[1/8] [========================97% ] gate_nsys_gate_harness1_prof.nsys-rep
[1/8] [========================100%] gate_nsys_gate_harness1_prof.nsys-rep
[1/8] [========================100%] gate_nsys_gate_harness1_prof.nsys-rep

[2/8] [0%                          ] gate_nsys_gate_harness1_prof.sqlite
[2KProcessing 1018 events: 

[2/8] [1%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [2%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [3%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [4%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [5%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [6%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [7%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [8%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [9%                          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [10%                         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [11%                         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [12%                         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [13%                         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [14%                         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=15%                        ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=16%                        ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=17%                        ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==18%                       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==19%                       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==20%                       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==21%                       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===22%                      ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===23%                      ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===24%                      ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [====25%                     ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [====26%                     ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [====27%                     ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [====28%                     ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====29%                    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====30%                    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====31%                    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====32%                    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [======33%                   ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [======34%                   ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [======35%                   ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======36%                  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======37%                  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======38%                  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======39%                  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========40%                 ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========41%                 ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========42%                 ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=========43%                ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=========44%                ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=========45%                ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=========46%                ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==========47%               ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==========48%               ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==========49%               ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===========50%              ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===========51%              ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===========52%              ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===========53%              ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [============54%             ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [============55%             ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [============56%             ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [============57%             ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=============58%            ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=============59%            ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=============60%            ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==============61%           ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==============62%           ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==============63%           ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==============64%           ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===============65%          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===============66%          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===============67%          ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [================68%         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [================69%         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [================70%         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [================71%         ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=================72%        ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=================73%        ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=================74%        ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==================75%       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==================76%       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==================77%       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [==================78%       ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===================79%      ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===================80%      ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===================81%      ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [===================82%      ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [====================83%     ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [====================84%     ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [====================85%     ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====================86%    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====================87%    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====================88%    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=====================89%    ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [======================90%   ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [======================91%   ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [======================92%   ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======================93%  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======================94%  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======================95%  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [=======================96%  ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========================97% ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========================98% ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========================99% ] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========================100%] gate_nsys_gate_harness1_prof.sqlite
[2/8] [========================100%] gate_nsys_gate_harness1_prof.sqlite
5.36466048695884545e-01 
5.36039895567142044e-01 
5.40751061319241666e-01 
5.36212892349821391e-01 
5.39204062915815352e-01 
5.38379088865937105e-01 
5.36743434583053403e-01 
5.38699451026247611e-01 
TIMING 1.790047
[3/8] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Style       Range     
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------  --------------
    100.0        7,195,623          1  7,195,623.0  7,195,623.0  7,195,623  7,195,623          0.0  PushPop  :rosenna_timed

[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)  Name
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----
    100.0       10,072,358          1  10,072,358.0  10,072,358.0  10,072,358  10,072,358          0.0  poll

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)         Name        
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------------------
     98.2        6,868,834          4  1,717,208.5  1,719,297.0  1,707,731  1,722,509      6,519.7  cuStreamSynchronize
      1.8          122,666          4     30,666.5      8,697.0      6,508     98,764     45,410.1  cuLaunchKernel     

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)          Name         
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ---------------------
    100.0        6,866,865          4  1,716,716.3  1,716,316.0  1,713,837  1,720,396      2,966.0  nvkernel_main_F1L50_4

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report
[8/8] Executing 'cuda_gpu_mem_size_sum' stats report
Generated:
	/fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness1_prof.nsys-rep
	/fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness1_prof.sqlite

```
stderr:
```
WARNING: CPU IP/backtrace sampling not supported, disabling.
Try the 'nsys status --environment' command to learn more.

WARNING: CPU context switch tracing not supported, disabling.
Try the 'nsys status --environment' command to learn more.

SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness1_prof.sqlite does not contain GPU memory data.
SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness1_prof.sqlite does not contain GPU memory data.

```

**nsys stats -q --force-export=true --report cuda_api_sum --format csv** (in gate-cur/file_loaded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys stats -q --force-export=true --report cuda_api_sum --format csv /fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness1_prof.nsys-rep
```
exit status: 0
stdout:
```
Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
98.2,6868834,4,1717208.5,1719297.0,1707731,1722509,6519.7,cuStreamSynchronize
1.8,122666,4,30666.5,8697.0,6508,98764,45410.1,cuLaunchKernel


```
c per-point: Memcpy API calls (cudaMemcpy*, cuMemcpy*) inside the nvtx-scoped 4-step loop, from cuda_api_sum: 0

**compile marker-bracketed fortran per-point harness** (in gate-cur/file_loaded)

```
$ nvfortran -O2 -mp=gpu -gpu=cc80 -DROSENNA_GATE_MARKERS=1 gate_harness2.F90 libgemm_big_f.a -o gate_harness2_prof
```
exit status: 2
stderr:
```
gate_harness2.F90:
/usr/bin/ld: /tmp/nvfortranax9ika_2yf1yy.o: in function `MAIN_':
/fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_harness2.F90:58:(.text+0x1614): undefined reference to `nvtxRangePushA'
/usr/bin/ld: /fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_harness2.F90:64:(.text+0x167d): undefined reference to `nvtxRangePop'
pgacclnk: child process exit status 1: /usr/bin/ld

```

**compile marker-bracketed fortran per-point harness (retry: -lnvToolsExt)** (in gate-cur/file_loaded)

```
$ nvfortran -O2 -mp=gpu -gpu=cc80 -DROSENNA_GATE_MARKERS=1 gate_harness2.F90 libgemm_big_f.a -lnvToolsExt -o gate_harness2_prof
```
exit status: 2
stderr:
```
gate_harness2.F90:
/usr/bin/ld: cannot find -lnvToolsExt: No such file or directory
pgacclnk: child process exit status 1: /usr/bin/ld

```

**compile marker-bracketed fortran per-point harness (retry: -cudalib=nvtx)** (in gate-cur/file_loaded)

```
$ nvfortran -O2 -mp=gpu -gpu=cc80 -DROSENNA_GATE_MARKERS=1 gate_harness2.F90 libgemm_big_f.a -cudalib=nvtx -o gate_harness2_prof
```
exit status: 0
stderr:
```
gate_harness2.F90:

```

**nsys profile --capture-range=nvtx --nvtx-capture=rosenna_timed (fortran per-point)** (in gate-cur/file_loaded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys profile -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range=nvtx --nvtx-capture=rosenna_timed --capture-range-end=stop --stats=true --force-overwrite=true -o /fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness2_prof ./gate_harness2_prof
```
exit status: 0
stdout:
```
Capture range started in the application.
Capture range ended in the application.
Generating '/tmp/nsys-report-b741.qdstrm'

[1/8] [0%                          ] gate_nsys_gate_harness2_prof.nsys-repProcessing events...

[1/8] [0%                          ] gate_nsys_gate_harness2_prof.nsys-rep
[1/8] [==========48%               ] gate_nsys_gate_harness2_prof.nsys-rep
[1/8] [========================97% ] gate_nsys_gate_harness2_prof.nsys-rep
[1/8] [========================98% ] gate_nsys_gate_harness2_prof.nsys-rep
[1/8] [========================100%] gate_nsys_gate_harness2_prof.nsys-rep
[1/8] [========================100%] gate_nsys_gate_harness2_prof.nsys-rep

[2/8] [0%                          ] gate_nsys_gate_harness2_prof.sqlite
[2KProcessing 1027 events: 

[2/8] [1%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [2%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [3%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [4%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [5%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [6%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [7%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [8%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [9%                          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [10%                         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [11%                         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [12%                         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [13%                         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [14%                         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=15%                        ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=16%                        ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=17%                        ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==18%                       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==19%                       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==20%                       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==21%                       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===22%                      ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===23%                      ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===24%                      ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [====25%                     ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [====26%                     ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [====27%                     ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [====28%                     ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====29%                    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====30%                    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====31%                    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====32%                    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [======33%                   ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [======34%                   ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [======35%                   ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======36%                  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======37%                  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======38%                  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======39%                  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========40%                 ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========41%                 ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========42%                 ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=========43%                ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=========44%                ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=========45%                ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=========46%                ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==========47%               ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==========48%               ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==========49%               ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===========50%              ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===========51%              ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===========52%              ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===========53%              ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [============54%             ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [============55%             ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [============56%             ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [============57%             ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=============58%            ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=============59%            ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=============60%            ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==============61%           ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==============62%           ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==============63%           ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==============64%           ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===============65%          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===============66%          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===============67%          ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [================68%         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [================69%         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [================70%         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [================71%         ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=================72%        ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=================73%        ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=================74%        ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==================75%       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==================76%       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==================77%       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [==================78%       ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===================79%      ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===================80%      ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===================81%      ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [===================82%      ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [====================83%     ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [====================84%     ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [====================85%     ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====================86%    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====================87%    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====================88%    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=====================89%    ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [======================90%   ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [======================91%   ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [======================92%   ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======================93%  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======================94%  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======================95%  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [=======================96%  ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========================97% ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========================98% ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========================99% ] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========================100%] gate_nsys_gate_harness2_prof.sqlite
[2/8] [========================100%] gate_nsys_gate_harness2_prof.sqlite
  5.3646604869588455E-01
  5.3603989556714204E-01
  5.4075106131924167E-01
  5.3621289234982139E-01
  5.3920406291581535E-01
  5.3837908886593711E-01
  5.3674343458305340E-01
  5.3869945102624761E-01
TIMING   1.7912250000000001E+00
[3/8] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Style       Range     
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------  --------------
    100.0        7,200,609          1  7,200,609.0  7,200,609.0  7,200,609  7,200,609          0.0  PushPop  :rosenna_timed

[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)  Name
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----
    100.0       10,121,964          1  10,121,964.0  10,121,964.0  10,121,964  10,121,964          0.0  poll

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)         Name        
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------------------
     98.4        6,900,108          4  1,725,027.0  1,726,824.5  1,712,946  1,733,513      8,654.8  cuStreamSynchronize
      1.6          111,663          4     27,915.8     11,433.0      8,347     80,450     35,059.6  cuLaunchKernel     

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                Name              
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  --------------------------------
    100.0        6,915,638          4  1,728,909.5  1,729,117.0  1,724,366  1,733,038      4,761.2  nvkernel_host_step_loop_F1L98_6_

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report
[8/8] Executing 'cuda_gpu_mem_size_sum' stats report
Generated:
	/fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness2_prof.nsys-rep
	/fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness2_prof.sqlite

```
stderr:
```
WARNING: CPU IP/backtrace sampling not supported, disabling.
Try the 'nsys status --environment' command to learn more.

WARNING: CPU context switch tracing not supported, disabling.
Try the 'nsys status --environment' command to learn more.

SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness2_prof.sqlite does not contain GPU memory data.
SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness2_prof.sqlite does not contain GPU memory data.

```

**nsys stats -q --force-export=true --report cuda_api_sum --format csv** (in gate-cur/file_loaded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys stats -q --force-export=true --report cuda_api_sum --format csv /fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness2_prof.nsys-rep
```
exit status: 0
stdout:
```
Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
98.4,6900108,4,1725027.0,1726824.5,1712946,1733513,8654.8,cuStreamSynchronize
1.6,111663,4,27915.8,11433.0,8347,80450,35059.6,cuLaunchKernel


```
fortran per-point: Memcpy API calls (cudaMemcpy*, cuMemcpy*) inside the nvtx-scoped 4-step loop, from cuda_api_sum: 0

**compile marker-bracketed cuda infer_batch harness** (in gate-cur/file_loaded)

```
$ nvcc -arch=sm_80 -DROSENNA_GATE_MARKERS=1 gate_harness3.cu -Lcuda_lib -lgemm_big -o gate_harness3_prof
```
exit status: 0

**nsys profile --capture-range=nvtx --nvtx-capture=rosenna_timed (cuda infer_batch)** (in gate-cur/file_loaded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys profile -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 --capture-range=nvtx --nvtx-capture=rosenna_timed --capture-range-end=stop --stats=true --force-overwrite=true -o /fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness3_prof ./gate_harness3_prof
```
exit status: 0
stdout:
```
Capture range started in the application.
Capture range ended in the application.
Generating '/tmp/nsys-report-9905.qdstrm'

[1/8] [0%                          ] gate_nsys_gate_harness3_prof.nsys-repProcessing events...

[1/8] [0%                          ] gate_nsys_gate_harness3_prof.nsys-rep
[1/8] [==========49%               ] gate_nsys_gate_harness3_prof.nsys-rep
[1/8] [========================98% ] gate_nsys_gate_harness3_prof.nsys-rep
[1/8] [========================100%] gate_nsys_gate_harness3_prof.nsys-rep
[1/8] [========================100%] gate_nsys_gate_harness3_prof.nsys-rep

[2/8] [0%                          ] gate_nsys_gate_harness3_prof.sqlite
[2KProcessing 1012 events: 

[2/8] [1%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [2%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [3%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [4%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [5%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [6%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [7%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [8%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [9%                          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [10%                         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [11%                         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [12%                         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [13%                         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [14%                         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=15%                        ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=16%                        ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=17%                        ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==18%                       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==19%                       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==20%                       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==21%                       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===22%                      ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===23%                      ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===24%                      ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [====25%                     ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [====26%                     ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [====27%                     ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [====28%                     ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====29%                    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====30%                    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====31%                    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====32%                    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [======33%                   ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [======34%                   ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [======35%                   ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======36%                  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======37%                  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======38%                  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======39%                  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========40%                 ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========41%                 ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========42%                 ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=========43%                ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=========44%                ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=========45%                ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=========46%                ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==========47%               ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==========48%               ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==========49%               ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===========50%              ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===========51%              ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===========52%              ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===========53%              ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [============54%             ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [============55%             ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [============56%             ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [============57%             ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=============58%            ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=============59%            ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=============60%            ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==============61%           ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==============62%           ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==============63%           ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==============64%           ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===============65%          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===============66%          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===============67%          ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [================68%         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [================69%         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [================70%         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [================71%         ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=================72%        ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=================73%        ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=================74%        ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==================75%       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==================76%       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==================77%       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [==================78%       ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===================79%      ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===================80%      ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===================81%      ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [===================82%      ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [====================83%     ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [====================84%     ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [====================85%     ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====================86%    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====================87%    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====================88%    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=====================89%    ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [======================90%   ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [======================91%   ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [======================92%   ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======================93%  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======================94%  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======================95%  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [=======================96%  ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========================97% ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========================98% ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========================99% ] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========================100%] gate_nsys_gate_harness3_prof.sqlite
[2/8] [========================100%] gate_nsys_gate_harness3_prof.sqlite
5.36466048695884545e-01 
5.36039895567142044e-01 
5.40751061319241666e-01 
5.36212892349821391e-01 
5.39204062915815352e-01 
5.38379088865937105e-01 
5.36743434583053403e-01 
5.38699451026247611e-01 
TIMING 1622.259224
[3/8] Executing 'nvtx_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Style       Range     
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------  --------------
    100.0        5,971,598          1  5,971,598.0  5,971,598.0  5,971,598  5,971,598          0.0  PushPop  :rosenna_timed

[4/8] Executing 'osrt_sum' stats report
[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)          Name         
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ---------------------
     97.4        5,751,389          1  5,751,389.0  5,751,389.0  5,751,389  5,751,389          0.0  cudaDeviceSynchronize
      2.5          150,310          4     37,577.5     13,048.0      8,469    115,745     52,279.8  cudaLaunchKernel     
      0.1            5,281          4      1,320.3        314.5        127      4,525      2,141.9  cuKernelGetName      

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)                       Name                     
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  ----------------------------------------------
    100.0        5,799,310          4  1,449,827.5  1,449,947.5  1,446,347  1,453,068      3,223.2  gemm_big_kernel(int, const double *, double *)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report
[8/8] Executing 'cuda_gpu_mem_size_sum' stats report
Generated:
	/fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness3_prof.nsys-rep
	/fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness3_prof.sqlite

```
stderr:
```
WARNING: CPU IP/backtrace sampling not supported, disabling.
Try the 'nsys status --environment' command to learn more.

WARNING: CPU context switch tracing not supported, disabling.
Try the 'nsys status --environment' command to learn more.

SKIPPED: No data available.
SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness3_prof.sqlite does not contain GPU memory data.
SKIPPED: /fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness3_prof.sqlite does not contain GPU memory data.

```

**nsys stats -q --force-export=true --report cuda_api_sum --format csv** (in gate-cur/file_loaded)

```
$ /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nsys stats -q --force-export=true --report cuda_api_sum --format csv /fastscratch/sbryngelson3/roseNNa/gate-cur/file_loaded/gate_nsys_gate_harness3_prof.nsys-rep
```
exit status: 0
stdout:
```
Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
97.4,5751389,1,5751389.0,5751389.0,5751389,5751389,0.0,cudaDeviceSynchronize
2.5,150310,4,37577.5,13048.0,8469,115745,52279.8,cudaLaunchKernel
0.1,5281,4,1320.3,314.5,127,4525,2141.9,cuKernelGetName


```
cuda infer_batch: Memcpy API calls (cudaMemcpy*, cuMemcpy*) inside the nvtx-scoped 4-step loop, from cuda_api_sum: 0

## result

PASS: every configuration matched.
