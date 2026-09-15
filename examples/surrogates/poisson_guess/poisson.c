/* Periodic Poisson solves with a conv-net initial guess. The whole right-hand
   side, with a 6-cell periodic halo the solver builds, is one model input
   (NCHW 1 x 1 x 76 x 76); the guess (1 x 1 x 64 x 64) starts a Jacobi solve on
   the device. Iterations are counted against a zero start and a warm start.
   The guess runs on the host: the generated infer holds a whole-field model's
   activations (660 KB here) as locals, more than a device thread's stack, so
   the solver uploads the 32 KB guess each step. Exits 0 if the NN start takes
   fewer iterations than the zero start. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "poisson_guess.h"

#define N 64
#define HALO 6
#define NP (N + 2 * HALO)
#define NSTEPS 20
#define TOL 1e-3
#define MAX_IT 20000
#define CHECK_EVERY 20

static inline int wrap(int i) { return (i + N) % N; }

/* Six Fourier modes whose phases advance with the step; mean zero, unit rms. */
static void rhs(double *f, int s) {
    static const int P[6] = {1, 2, -3, 4, 5, -7}, Q[6] = {2, -1, 3, 1, -5, 2};
    static const double A[6] = {0.9, -0.7, 0.5, 0.6, -0.4, 0.3};
    double mean = 0.0, ss = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double v = 0.0;
            for (int m = 0; m < 6; ++m)
                v += A[m] * cos(2.0 * M_PI * (P[m] * i + Q[m] * j) / N + 0.15 * s * (m + 1));
            f[i * N + j] = v; mean += v;
        }
    mean /= N * N;
    for (int c = 0; c < N * N; ++c) { f[c] -= mean; ss += f[c] * f[c]; }
    const double rms = sqrt(ss / (N * N));
    for (int c = 0; c < N * N; ++c) f[c] /= rms;
}

/* Jacobi in place until |lap(phi) - f| / |f| < TOL; returns iterations. */
static int jacobi(double *phi, double *tmp, const double *f, double fnorm) {
    int it;
    for (it = 0; it < MAX_IT; it += 2) {
#pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                tmp[i * N + j] = 0.25 * (phi[wrap(i - 1) * N + j] + phi[wrap(i + 1) * N + j]
                                         + phi[i * N + wrap(j - 1)] + phi[i * N + wrap(j + 1)] - f[i * N + j]);
#pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                phi[i * N + j] = 0.25 * (tmp[wrap(i - 1) * N + j] + tmp[wrap(i + 1) * N + j]
                                         + tmp[i * N + wrap(j - 1)] + tmp[i * N + wrap(j + 1)] - f[i * N + j]);
        if ((it + 2) % CHECK_EVERY == 0) {
            double r2 = 0.0;
#pragma omp target teams distribute parallel for collapse(2) reduction(+: r2)
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    const double lap = phi[wrap(i - 1) * N + j] + phi[wrap(i + 1) * N + j]
                                     + phi[i * N + wrap(j - 1)] + phi[i * N + wrap(j + 1)] - 4.0 * phi[i * N + j];
                    r2 += (lap - f[i * N + j]) * (lap - f[i * N + j]);
                }
            if (sqrt(r2) / fnorm < TOL) return it + 2;
        }
    }
    return MAX_IT;
}

static void zero_mean(double *phi) {
    double m = 0.0;
#pragma omp target teams distribute parallel for reduction(+: m)
    for (int c = 0; c < N * N; ++c) m += phi[c];
    m /= N * N;
#pragma omp target teams distribute parallel for
    for (int c = 0; c < N * N; ++c) phi[c] -= m;
}

int main(void) {
    double *f = malloc(sizeof(double) * N * N), *tmp = malloc(sizeof(double) * N * N);
    double *fp = malloc(sizeof(double) * NP * NP);
    double *phi_nn = calloc(N * N, sizeof(double)), *phi_zero = calloc(N * N, sizeof(double));
    double *phi_warm = calloc(N * N, sizeof(double));
    long it_nn = 0, it_zero = 0, it_warm = 0;
    double t_guess = 0.0;

#pragma omp target enter data map(alloc: f[0:N * N], tmp[0:N * N]) \
                              map(to: phi_nn[0:N * N], phi_zero[0:N * N], phi_warm[0:N * N])
    for (int s = 0; s < NSTEPS; ++s) {
        rhs(f, s);
        const double fnorm = sqrt((double)(N * N));
#pragma omp target update to(f[0:N * N])

        const double t0 = omp_get_wtime();
        for (int i = 0; i < NP; ++i)                           /* periodic halo */
            for (int j = 0; j < NP; ++j)
                fp[i * NP + j] = f[wrap(i - HALO) * N + wrap(j - HALO)];
        poisson_guess_infer(fp, phi_nn);                       /* the whole field, on the host */
#pragma omp target update to(phi_nn[0:N * N])
        zero_mean(phi_nn);
        t_guess += omp_get_wtime() - t0;

        it_nn += jacobi(phi_nn, tmp, f, fnorm);
#pragma omp target teams distribute parallel for
        for (int c = 0; c < N * N; ++c) phi_zero[c] = 0.0;
        it_zero += jacobi(phi_zero, tmp, f, fnorm);
        it_warm += jacobi(phi_warm, tmp, f, fnorm);
    }
#pragma omp target exit data map(delete: f[0:N * N], tmp[0:N * N], phi_nn[0:N * N], \
                                        phi_zero[0:N * N], phi_warm[0:N * N])

    printf("%dx%d periodic Poisson, %d steps of a rotating right-hand side, Jacobi to %.0e:\n",
           N, N, NSTEPS, TOL);
    printf("  iterations per step from zero               %6.0f\n", (double)it_zero / NSTEPS);
    printf("  iterations per step from the last solution  %6.0f\n", (double)it_warm / NSTEPS);
    printf("  iterations per step from the NN guess       %6.0f   (guess: %.1f ms per step)\n",
           (double)it_nn / NSTEPS, 1e3 * t_guess / NSTEPS);
    free(f); free(tmp); free(fp); free(phi_nn); free(phi_zero); free(phi_warm);
    if (it_nn >= (long)MAX_IT * NSTEPS || it_nn >= it_zero) {
        printf("FAIL: NN guess did not help\n");
        return 1;
    }
    printf("OK: NN guess saves %.0f%% of the zero-start iterations\n", 100.0 * (1.0 - (double)it_nn / it_zero));
    return 0;
}
