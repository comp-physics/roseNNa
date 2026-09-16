/* Emulate nvc's teams-loop mapping with a hand-written CUDA kernel:
   one BLOCK per point, 32 threads, only lane 0 doing the work. If the
   32x gap is the idle-lane mapping, this reproduces the nvc timing. */
#include <stdio.h>
#include <stdlib.h>
#include "gemm_big.h"
extern "C" int gemm_big_device_bind(void);

static __global__ void k_one_thread_per_point(int n, const double *__restrict__ x, double *__restrict__ y) {
    const int p = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= n) return;
    gemm_big_infer(x + (size_t)p * 2, y + (size_t)p * 1);
}
static __global__ void k_one_block_per_point(int n, const double *__restrict__ x, double *__restrict__ y) {
    if (threadIdx.x != 0) return;              /* nvc: 31 of 32 lanes idle */
    const int p = (int)blockIdx.x;
    if (p >= n) return;
    gemm_big_infer(x + (size_t)p * 2, y + (size_t)p * 1);
}
int main(void) {
    if (gemm_big_init("gemm_big.rwt") != 0) { printf("init failed\n"); return 1; }
    if (gemm_big_device_bind_here() != 0) { printf("bind failed\n"); return 1; }
    const long n = 1000000L;
    double *hx = (double*)malloc(sizeof(double)*n*2);
    for (long i = 0; i < n*2; ++i) hx[i] = 0.5;
    double *dx, *dy;
    cudaMalloc(&dx, sizeof(double)*n*2); cudaMalloc(&dy, sizeof(double)*n);
    cudaMemcpy(dx, hx, sizeof(double)*n*2, cudaMemcpyHostToDevice);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float ms;
    for (int rep = 0; rep < 2; ++rep) {
        k_one_thread_per_point<<<(n+127)/128, 128>>>(n, dx, dy);
        cudaDeviceSynchronize();
        cudaEventRecord(a);
        k_one_thread_per_point<<<(n+127)/128, 128>>>(n, dx, dy);
        cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b); { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { printf("CUDA ERR: %s\n", cudaGetErrorString(e)); return 2; } }
        if (rep) printf("128 thr/block, all lanes active : %7.2f ns/point\n", ms*1e6/n);
        cudaEventRecord(a);
        k_one_block_per_point<<<n, 32>>>(n, dx, dy);
        cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b); { cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) { printf("CUDA ERR: %s\n", cudaGetErrorString(e)); return 2; } }
        if (rep) printf("32 thr/block, lane 0 only (nvc) : %7.2f ns/point\n", ms*1e6/n);
    }
    return 0;
}
