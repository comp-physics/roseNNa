/* 2-D FitzHugh-Nagumo with a learned time-stepper on 3x3 patches, C.
 *
 * Pattern: a batched surrogate. Every big step, the solver gathers each
 * cell's 3x3 patch of (u, v) into one feature array, makes ONE call to
 * stepper_infer_batch over the whole field, and scatters the result back
 * into u and v. Everything stays on the device: feat and out are mapped
 * once and handed to infer_batch as device pointers (use_device_ptr), and
 * the model was generated file-loaded (--no-embed), so this is also the
 * example with an init: stepper_init reads stepper.rwt and uploads the
 * weights ONCE, before the loop.
 *
 * The program runs the fine scheme (K small steps per big step) as the
 * reference and the surrogate for NBIG big steps, prints the relative L2
 * error of the surrogate at the end and the time of both, and exits 0 if
 * the error is under TOL. Same scheme as train.py: explicit Euler, 5-point
 * Laplacian, periodic. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "stepper.h"

#ifndef NX
#define NX 256             /* grid; -DNX=64 for a quick host run */
#endif
#define NCELL (NX * NX)
#define DU 1.0
#define DV 0.05
#define PA 0.7
#define PB 0.8
#define EPS 0.08
#define DT 0.1
#define K 10               /* fine steps per surrogate step */
#define NBIG 100
#define TOL 0.05
#define NFEAT 18

static inline int wrap(int i) { return (i + NX) % NX; }

/* --- the fine scheme: one explicit step of the whole field --- */
static void fine_step(const double *u, const double *v, double *un, double *vn) {
#pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NX; ++j) {
            const int c = i * NX + j;
            const int n = wrap(i - 1) * NX + j, s = wrap(i + 1) * NX + j;
            const int w = i * NX + wrap(j - 1), e = i * NX + wrap(j + 1);
            const double lu = u[n] + u[s] + u[w] + u[e] - 4.0 * u[c];
            const double lv = v[n] + v[s] + v[w] + v[e] - 4.0 * v[c];
            un[c] = u[c] + DT * (DU * lu + u[c] - u[c] * u[c] * u[c] / 3.0 - v[c]);
            vn[c] = v[c] + DT * (DV * lv + EPS * (u[c] + PA - PB * v[c]));
        }
}

/* --- the surrogate: gather, one batched call, scatter --- */
static void gather(const double *u, const double *v, double *feat) {
#pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NX; ++j) {
            double *f = feat + (size_t)(i * NX + j) * NFEAT;
            /* Same order as train.py's patches(): u's 3x3 row-major, then v's. */
            for (int di = -1; di <= 1; ++di)
                for (int dj = -1; dj <= 1; ++dj) {
                    const int p = wrap(i + di) * NX + wrap(j + dj);
                    f[(di + 1) * 3 + (dj + 1)] = u[p];
                    f[9 + (di + 1) * 3 + (dj + 1)] = v[p];
                }
        }
}

static void scatter(const double *out, double *u, double *v) {
#pragma omp target teams distribute parallel for
    for (int c = 0; c < NCELL; ++c) {
        u[c] = out[2 * c];
        v[c] = out[2 * c + 1];
    }
}

static int big_step(double *u, double *v, double *feat, double *out) {
    gather(u, v, feat);
    int status;
    /* feat and out are already on the device; infer_batch gets their device
       addresses and launches over them. Nothing is transferred here. */
#pragma omp target data use_device_ptr(feat, out)
    status = stepper_infer_batch(NCELL, feat, out, 0);
    if (status) return status;
    /* With a cuda/hip libstepper.a the launch is asynchronous on the null
       stream and nothing orders it against this program's next target
       region (which runs on the OpenMP runtime's own queue): without this
       wait the scatter read stale output on an MI210. With an omp-backend
       archive the loop was synchronous and this is a no-op. */
    status = stepper_sync(0);
    if (status) return status;
    scatter(out, u, v);
    return 0;
}

/* A deterministic held-out initial condition: a few Fourier modes with
   hashed amplitudes and phases, then a tanh, as train.py's fields are. */
static double hash01(long long a, long long b) {
    return (double)(((a * 40503LL + b) * 2654435761LL) % 4294967296LL) / 4294967296.0;
}

static void initial_fields(double *u, double *v) {
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NX; ++j) {
            const double x = (double)i / NX, y = (double)j / NX;
            double su = 0.0, sv = 0.0;
            for (int m = 0; m < 4; ++m) {
                const int p = 1 + (int)(3 * hash01(7, m)), q = 1 + (int)(3 * hash01(8, m));
                const double ph = 2.0 * M_PI * (p * x + q * y);
                su += (2.0 * hash01(9, m) - 1.0) * sin(ph + 2.0 * M_PI * hash01(10, m));
                sv += (hash01(11, m) - 0.5) * sin(ph + 2.0 * M_PI * hash01(12, m));
            }
            u[i * NX + j] = 2.0 * tanh(su);
            v[i * NX + j] = 0.5 * tanh(sv);
        }
}

int main(void) {
    /* Plan step, once, before anything is mapped or timed: the weights go
       to the device here and never again. */
    const int st = stepper_init("stepper.rwt");
    if (st != 0) { printf("stepper_init failed with status %d\n", st); return 2; }

    double *u = malloc(sizeof(double) * NCELL), *v = malloc(sizeof(double) * NCELL);
    double *ur = malloc(sizeof(double) * NCELL), *vr = malloc(sizeof(double) * NCELL);
    double *ut = malloc(sizeof(double) * NCELL), *vt = malloc(sizeof(double) * NCELL);
    double *feat = malloc(sizeof(double) * NCELL * NFEAT), *out = malloc(sizeof(double) * NCELL * 2);
    initial_fields(u, v);
    for (int c = 0; c < NCELL; ++c) { ur[c] = u[c]; vr[c] = v[c]; }

    /* Reference: NBIG * K fine steps, ping-ponging two mapped buffers. */
#pragma omp target enter data map(to: ur[0:NCELL], vr[0:NCELL]) map(alloc: ut[0:NCELL], vt[0:NCELL])
    double t0 = omp_get_wtime();
    for (int s = 0; s < NBIG * K / 2; ++s) {
        fine_step(ur, vr, ut, vt);
        fine_step(ut, vt, ur, vr);
    }
    const double t_ref = omp_get_wtime() - t0;
#pragma omp target exit data map(from: ur[0:NCELL], vr[0:NCELL]) map(delete: ut[0:NCELL], vt[0:NCELL])

    /* Surrogate: map once, NBIG big steps, nothing moves inside the loop. */
#pragma omp target enter data map(to: u[0:NCELL], v[0:NCELL]) map(alloc: feat[0:NCELL * NFEAT], out[0:NCELL * 2])
    t0 = omp_get_wtime();
    int status = 0;
    for (int s = 0; s < NBIG && status == 0; ++s) status = big_step(u, v, feat, out);
    const double t_nn = omp_get_wtime() - t0;
#pragma omp target exit data map(from: u[0:NCELL], v[0:NCELL]) map(delete: feat[0:NCELL * NFEAT], out[0:NCELL * 2])
    if (status) { printf("stepper_infer_batch failed with status %d\n", status); return 3; }

    double e = 0.0, r = 0.0;
    for (int c = 0; c < NCELL; ++c) {
        e += (u[c] - ur[c]) * (u[c] - ur[c]) + (v[c] - vr[c]) * (v[c] - vr[c]);
        r += ur[c] * ur[c] + vr[c] * vr[c];
    }
    const double err = sqrt(e / r);
    printf("%dx%d grid, %d surrogate steps of %d fine steps each:\n", NX, NX, NBIG, K);
    printf("  fine reference   %6.1f ms  (%.2f ms per big step)\n", 1e3 * t_ref, 1e3 * t_ref / NBIG);
    printf("  surrogate        %6.1f ms  (%.2f ms per big step: gather, infer_batch, scatter)\n",
           1e3 * t_nn, 1e3 * t_nn / NBIG);
    printf("  relative L2 error of the surrogate vs the reference: %.3e\n", err);
    free(u); free(v); free(ur); free(vr); free(ut); free(vt); free(feat); free(out);
    if (!(err < TOL)) { printf("FAIL: error above %.2f\n", TOL); return 1; }
    printf("OK\n");
    return 0;
}
