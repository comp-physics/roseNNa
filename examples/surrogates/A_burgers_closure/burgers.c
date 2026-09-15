/* Coarse-grid viscous Burgers with a learned subgrid closure, C.
 *
 * Pattern: a per-cell closure called from the solver's own offload loop.
 * closure_infer is header-inline with its weights embedded, so there is no
 * init and nothing to link; the only device data are the solver's own
 * arrays, mapped once before the time loop. Nothing inside the loop
 * allocates or transfers.
 *
 * The problem is an ensemble of NB independent 1-D realizations (a UQ or
 * parameter sweep has this shape), stored as u[NB][n]; one realization is
 * far too small to occupy a GPU, and a kernel that small is latency-bound
 * whatever it computes. The program runs three things and compares them:
 *   1. the fine-grid reference (NF cells), box-filtered to the coarse grid
 *   2. the coarse scheme alone (NC cells)
 *   3. the coarse scheme plus NN(stencil) per cell
 * and exits 0 if the closure brings the coarse run closer to the reference
 * than the coarse scheme alone, on average over the ensemble. Same scheme
 * as train.py: Godunov flux, central viscous term, forward Euler. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "closure.h"

#ifndef NB
#define NB 64              /* ensemble size; -DNB=4 for a quick host run */
#endif
#define NF 2048
#define FACTOR 16          /* spatial coarsening */
#define NC (NF / FACTOR)
#define NU 0.02
#define DT_C 0.01
#define N_SUB 64           /* fine sub-steps per coarse step */
#define NSTEPS 200
static const double L = 2.0 * M_PI;

static double godunov(double ul, double ur) {
    const double a = 0.5 * ul * ul, b = 0.5 * ur * ur;
    return (ul <= ur) ? ((ul <= 0.0 && ur >= 0.0) ? 0.0 : fmin(a, b)) : fmax(a, b);
}

/* One forward-Euler step of the scheme on every realization, into unew.
   With the closure on, NN(stencil) is added to each cell's right-hand side:
   this is the call a solver makes inside its step loop. */
static void step(const double *u, double *unew, int n, double dx, double dt, int use_nn) {
#pragma omp target teams distribute parallel for collapse(2)
    for (int b = 0; b < NB; ++b) {
        for (int i = 0; i < n; ++i) {
            const double *r = u + (size_t)b * n;
            const int im3 = (i - 3 + n) % n, im2 = (i - 2 + n) % n, im1 = (i - 1 + n) % n;
            const int ip1 = (i + 1) % n, ip2 = (i + 2) % n, ip3 = (i + 3) % n;
            double rhs = -(godunov(r[i], r[ip1]) - godunov(r[im1], r[i])) / dx
                       + NU * (r[ip1] - 2.0 * r[i] + r[im1]) / (dx * dx);
            if (use_nn) {
                const double stencil[7] = {r[im3], r[im2], r[im1], r[i], r[ip1], r[ip2], r[ip3]};
                double corr[1];
                closure_infer(stencil, corr);      /* the surrogate, per cell, on the device */
                rhs += corr[0];
            }
            unew[(size_t)b * n + i] = r[i] + dt * rhs;
        }
    }
}

static void run(double *u, double *tmp, int n, double dx, double dt, int nsteps, int use_nn) {
    /* Map once; the step loop below moves nothing. cur/nxt swap the two
       mapped buffers on the device; the result is copied back into u. */
    const size_t m = (size_t)NB * n;
    double *cur = u, *nxt = tmp;
#pragma omp target enter data map(to: cur[0:m]) map(alloc: nxt[0:m])
    for (int s = 0; s < nsteps; ++s) {
        step(cur, nxt, n, dx, dt, use_nn);
        double *t = cur; cur = nxt; nxt = t;
    }
#pragma omp target exit data map(from: cur[0:m]) map(delete: nxt[0:m])
    if (cur != u)
        for (size_t i = 0; i < m; ++i) u[i] = cur[i];
}

static void box_filter(const double *fine, double *coarse) {
    for (int b = 0; b < NB; ++b)
        for (int i = 0; i < NC; ++i) {
            double s = 0.0;
            for (int k = 0; k < FACTOR; ++k) s += fine[(size_t)b * NF + i * FACTOR + k];
            coarse[(size_t)b * NC + i] = s / FACTOR;
        }
}

/* Mean over the ensemble of the relative L2 error of each realization. */
static double mean_rel_l2(const double *a, const double *ref) {
    double total = 0.0;
    for (int b = 0; b < NB; ++b) {
        double e = 0.0, r = 0.0;
        for (int i = 0; i < NC; ++i) {
            const double d = a[(size_t)b * NC + i] - ref[(size_t)b * NC + i];
            e += d * d; r += ref[(size_t)b * NC + i] * ref[(size_t)b * NC + i];
        }
        total += sqrt(e / r);
    }
    return total / NB;
}

/* A deterministic ensemble of initial conditions (three sine modes with
   pseudo-random amplitude and phase) from an integer hash that stays inside
   a signed 64-bit range, so burgers.F90 reproduces it exactly. */
static double hash01(long long b, long long k) {
    return (double)(((b * 40503LL + k) * 2654435761LL) % 4294967296LL) / 4294967296.0;
}

int main(void) {
    double *uf = malloc(sizeof(double) * NB * NF), *tf = malloc(sizeof(double) * NB * NF);
    double *ref = malloc(sizeof(double) * NB * NC), *uc = malloc(sizeof(double) * NB * NC);
    double *tc = malloc(sizeof(double) * NB * NC), *un = malloc(sizeof(double) * NB * NC);
    double *tn = malloc(sizeof(double) * NB * NC);
    const double dxf = L / NF, dxc = L / NC;
    for (int b = 0; b < NB; ++b) {
        double amp[3], ph[3];
        for (int k = 0; k < 3; ++k) {
            amp[k] = (2.0 * hash01(b + 1, 2 * k + 1) - 1.0) / (k + 1);
            ph[k] = 2.0 * M_PI * hash01(b + 1, 2 * k + 2);
        }
        for (int i = 0; i < NF; ++i) {
            const double x = i * dxf;
            double v = 0.0;
            for (int k = 0; k < 3; ++k) v += amp[k] * sin((k + 1) * x + ph[k]);
            uf[(size_t)b * NF + i] = v;
        }
    }
    box_filter(uf, uc);
    for (size_t i = 0; i < (size_t)NB * NC; ++i) un[i] = uc[i];

    /* 1. Fine reference: N_SUB sub-steps per coarse step, then box-filter. */
    double t0 = omp_get_wtime();
    run(uf, tf, NF, dxf, DT_C / N_SUB, NSTEPS * N_SUB, 0);
    const double t_ref = omp_get_wtime() - t0;
    box_filter(uf, ref);
    /* 2. Coarse alone.  3. Coarse + closure. */
    t0 = omp_get_wtime();
    run(uc, tc, NC, dxc, DT_C, NSTEPS, 0);
    const double t_coarse = omp_get_wtime() - t0;
    t0 = omp_get_wtime();
    run(un, tn, NC, dxc, DT_C, NSTEPS, 1);
    const double t_nn = omp_get_wtime() - t0;

    const double e_coarse = mean_rel_l2(uc, ref), e_nn = mean_rel_l2(un, ref);
    printf("ensemble of %d realizations, %d coarse cells, %d steps; mean relative L2 error "
           "vs the filtered fine reference:\n", NB, NC, NSTEPS);
    printf("  coarse scheme alone     %.4e   (%.1f ms)\n", e_coarse, 1e3 * t_coarse);
    printf("  coarse + NN closure     %.4e   (%.1f ms, %.2f ns per cell-step for the closure)\n",
           e_nn, 1e3 * t_nn, 1e9 * (t_nn - t_coarse) / ((double)NB * NC * NSTEPS));
    printf("  fine reference, %d cells               (%.1f ms)\n", NF, 1e3 * t_ref);
    free(uf); free(tf); free(ref); free(uc); free(tc); free(un); free(tn);
    if (!(e_nn < e_coarse)) {
        printf("FAIL: the closure did not improve on the coarse scheme\n");
        return 1;
    }
    printf("OK: closure reduces the error by %.1fx\n", e_coarse / e_nn);
    return 0;
}
