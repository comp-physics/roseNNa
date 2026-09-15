/* Coarse-grid Burgers with a learned subgrid closure: closure_infer is called
   per cell from the solver's own offload loop. Embedded model, no init.
   An ensemble of NB realizations, u[NB][n]; fine reference vs coarse vs
   coarse + closure. Exits 0 if the closure reduces the coarse error. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "closure.h"

#define NF 2048
#define FACTOR 16
#define NC (NF / FACTOR)
#define NU 0.02
#define DT 0.01
#define N_SUB 64            /* fine sub-steps per coarse step */
#define NSTEPS 200
static const double L = 2.0 * M_PI;

static double godunov(double ul, double ur) {
    const double a = 0.5 * ul * ul, b = 0.5 * ur * ur;
    return (ul <= ur) ? ((ul <= 0.0 && ur >= 0.0) ? 0.0 : fmin(a, b)) : fmax(a, b);
}

/* Forward Euler, Godunov flux, central viscous term; NN(stencil) added when use_nn. */
static void step(const double *u, double *unew, int n, double dx, double dt, int use_nn) {
#pragma omp target teams distribute parallel for collapse(2)
    for (int b = 0; b < NB; ++b)
        for (int i = 0; i < n; ++i) {
            const double *r = u + (size_t)b * n;
            const int im3 = (i - 3 + n) % n, im2 = (i - 2 + n) % n, im1 = (i - 1 + n) % n;
            const int ip1 = (i + 1) % n, ip2 = (i + 2) % n, ip3 = (i + 3) % n;
            double rhs = -(godunov(r[i], r[ip1]) - godunov(r[im1], r[i])) / dx
                       + NU * (r[ip1] - 2.0 * r[i] + r[im1]) / (dx * dx);
            if (use_nn) {
                const double stencil[7] = {r[im3], r[im2], r[im1], r[i], r[ip1], r[ip2], r[ip3]};
                double corr[1];
                closure_infer(stencil, corr);
                rhs += corr[0];
            }
            unew[(size_t)b * n + i] = r[i] + dt * rhs;
        }
}

/* Map once, step nsteps times swapping device buffers, result back in u. */
static void run(double *u, double *tmp, int n, double dx, double dt, int nsteps, int use_nn) {
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

/* Integer hash in [0, 1); every intermediate fits int64, so burgers.F90 reproduces it. */
static double hash01(long long a, long long b) {
    return (double)(((a * 40503LL + b) * 2654435761LL) % 4294967296LL) / 4294967296.0;
}

int main(void) {
    double *uf = malloc(sizeof(double) * NB * NF), *tf = malloc(sizeof(double) * NB * NF);
    double *ref = malloc(sizeof(double) * NB * NC), *uc = malloc(sizeof(double) * NB * NC);
    double *tc = malloc(sizeof(double) * NB * NC), *un = malloc(sizeof(double) * NB * NC);
    double *tn = malloc(sizeof(double) * NB * NC);
    const double dxf = L / NF, dxc = L / NC;
    for (int b = 0; b < NB; ++b) {                    /* three sine modes, hashed amplitude and phase */
        double amp[3], ph[3];
        for (int k = 0; k < 3; ++k) {
            amp[k] = (2.0 * hash01(b + 1, 2 * k + 1) - 1.0) / (k + 1);
            ph[k] = 2.0 * M_PI * hash01(b + 1, 2 * k + 2);
        }
        for (int i = 0; i < NF; ++i) {
            double v = 0.0;
            for (int k = 0; k < 3; ++k) v += amp[k] * sin((k + 1) * i * dxf + ph[k]);
            uf[(size_t)b * NF + i] = v;
        }
    }
    box_filter(uf, uc);
    for (size_t i = 0; i < (size_t)NB * NC; ++i) un[i] = uc[i];

    double t0 = omp_get_wtime();
    run(uf, tf, NF, dxf, DT / N_SUB, NSTEPS * N_SUB, 0);
    const double t_ref = omp_get_wtime() - t0;
    box_filter(uf, ref);
    t0 = omp_get_wtime();
    run(uc, tc, NC, dxc, DT, NSTEPS, 0);
    const double t_coarse = omp_get_wtime() - t0;
    t0 = omp_get_wtime();
    run(un, tn, NC, dxc, DT, NSTEPS, 1);
    const double t_nn = omp_get_wtime() - t0;

    const double e_coarse = mean_rel_l2(uc, ref), e_nn = mean_rel_l2(un, ref);
    printf("%d realizations, %d coarse cells, %d steps; mean relative L2 error vs filtered fine:\n",
           NB, NC, NSTEPS);
    printf("  coarse             %.4e  (%.1f ms)\n", e_coarse, 1e3 * t_coarse);
    printf("  coarse + closure   %.4e  (%.1f ms, %.1f ns per cell-step for the closure)\n",
           e_nn, 1e3 * t_nn, 1e9 * (t_nn - t_coarse) / ((double)NB * NC * NSTEPS));
    printf("  fine, %d cells   (%.1f ms)\n", NF, 1e3 * t_ref);
    free(uf); free(tf); free(ref); free(uc); free(tc); free(un); free(tn);
    if (!(e_nn < e_coarse)) { printf("FAIL: closure did not help\n"); return 1; }
    printf("OK: error reduced %.1fx\n", e_coarse / e_nn);
    return 0;
}
