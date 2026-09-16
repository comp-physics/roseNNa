/* 1-D acoustics through a bubbly region with a recurrent per-cell surrogate.
   Each cell owns an LSTM state (h, c) kept on the device; per step the model
   gets x = [p, h, c] and returns y = [s, h', c'] (layout from `rosenna info`:
   x: p[0:1] h[1:33] c[33:65], y: s[0:1] h_next[1:33] c_next[33:65]), and s,
   the bubble population's volume-fraction rate, is the acoustic source.
   Reference: NBIN Rayleigh-Plesset bins per cell, RK4 with N_SUB sub-steps.
   Characteristic upwind at CFL = 1 (exact transport). Embedded model.
   Ensemble of NB lines with hashed pulses. Exits 0 if the surrogate's
   pressure field is within TOL of the reference after NSTEPS steps. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "bubbles.h"

#define NXC 512
#define NCELL (NB * NXC)
#define DX 0.05
#define DT 0.05
#ifndef NSTEPS
#define NSTEPS 400
#endif
#define BETA 0.1            /* coupling in the bubbly region; unstable above ~0.2 */
#define X_LO 10.0
#define X_HI 20.0
#define TOL 0.10
#define NBIN 8
#define GAMMA 1.4
#define MU 0.05
#define N_SUB 10
#define HID 32
#define NIN (1 + 2 * HID)
#define NOUT (1 + 2 * HID)

static double r0_of(int k) { return 0.5 * pow(4.0, (double)k / (NBIN - 1)); }   /* geomspace(0.5, 2) */

static void rp_rhs(double R, double V, double p, double r0, double *dR, double *dV) {
    *dR = V;
    *dV = (pow(r0 / R, 3.0 * GAMMA) - 1.0 - p - 4.0 * MU * V / R - 1.5 * V * V) / R;
}

/* One acoustic step of one cell's bins; returns s = sum_k w_k 3 R_k^2 V_k / R0_k^3. */
static double population_step(double *R, double *V, double p) {
    const double h = DT / N_SUB;
    double s = 0.0;
    for (int k = 0; k < NBIN; ++k) {
        const double r0 = r0_of(k);
        double r = R[k], v = V[k];
        for (int sub = 0; sub < N_SUB; ++sub) {
            double k1r, k1v, k2r, k2v, k3r, k3v, k4r, k4v;
            rp_rhs(r, v, p, r0, &k1r, &k1v);
            rp_rhs(r + 0.5 * h * k1r, v + 0.5 * h * k1v, p, r0, &k2r, &k2v);
            rp_rhs(r + 0.5 * h * k2r, v + 0.5 * h * k2v, p, r0, &k3r, &k3v);
            rp_rhs(r + h * k3r, v + h * k3v, p, r0, &k4r, &k4v);
            r += h / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
            v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        }
        R[k] = r; V[k] = v;
        s += (1.0 / NBIN) * 3.0 * r * r * v / (r0 * r0 * r0);
    }
    return s;
}

static void step(const double *wp, const double *wm, double *wp_new, double *wm_new, const double *beta,
                 double *R, double *V, double *H, double *C, double *src, int use_nn) {
#pragma omp target teams distribute parallel for
    for (int c = 0; c < NCELL; ++c) {
        const double p = 0.5 * (wp[c] + wm[c]);
        if (use_nn) {
            double x[NIN], y[NOUT];
            x[0] = p;
            for (int i = 0; i < HID; ++i) { x[1 + i] = H[(size_t)c * HID + i]; x[1 + HID + i] = C[(size_t)c * HID + i]; }
            bubbles_infer(x, y);
            src[c] = y[0];
            for (int i = 0; i < HID; ++i) { H[(size_t)c * HID + i] = y[1 + i]; C[(size_t)c * HID + i] = y[1 + HID + i]; }
        } else {
            src[c] = population_step(R + (size_t)c * NBIN, V + (size_t)c * NBIN, p);
        }
    }
#pragma omp target teams distribute parallel for collapse(2)
    for (int b = 0; b < NB; ++b)
        for (int i = 0; i < NXC; ++i) {
            const int c = b * NXC + i;
            const int l = b * NXC + (i > 0 ? i - 1 : 0), r = b * NXC + (i < NXC - 1 ? i + 1 : NXC - 1);
            wp_new[c] = wp[l] - DT * beta[c] * src[c];
            wm_new[c] = wm[r] - DT * beta[c] * src[c];
        }
}

static void run(double *wp, double *wm, double *wp2, double *wm2, const double *beta,
                double *R, double *V, double *H, double *C, double *src, int use_nn) {
    for (int s = 0; s < NSTEPS / 2; ++s) {
        step(wp, wm, wp2, wm2, beta, R, V, H, C, src, use_nn);
        step(wp2, wm2, wp, wm, beta, R, V, H, C, src, use_nn);
    }
}

static double hash01(long long a, long long b) {
    return (double)(((a * 40503LL + b) * 2654435761LL) % 4294967296LL) / 4294967296.0;
}

/* A right-going pulse per line, hashed amplitude and width. */
static void set_pulse(double *wp, double *wm) {
    for (int b = 0; b < NB; ++b) {
        const double amp = 0.1 + 0.25 * hash01(b + 1, 1), sig = 0.5 + 1.0 * hash01(b + 1, 2);
        for (int i = 0; i < NXC; ++i) {
            const double x = (i + 0.5) * DX;
            wp[b * NXC + i] = 2.0 * amp * exp(-0.5 * ((x - 4.0) / sig) * ((x - 4.0) / sig));
            wm[b * NXC + i] = 0.0;
        }
    }
}

int main(void) {
    double *wp = malloc(sizeof(double) * NCELL), *wm = malloc(sizeof(double) * NCELL);
    double *wp2 = malloc(sizeof(double) * NCELL), *wm2 = malloc(sizeof(double) * NCELL);
    double *pref = malloc(sizeof(double) * NCELL), *beta = malloc(sizeof(double) * NCELL);
    double *src = malloc(sizeof(double) * NCELL);
    double *R = malloc(sizeof(double) * NCELL * NBIN), *V = malloc(sizeof(double) * NCELL * NBIN);
    double *H = calloc((size_t)NCELL * HID, sizeof(double)), *C = calloc((size_t)NCELL * HID, sizeof(double));
    const size_t nst = (size_t)NCELL * NBIN, nh = (size_t)NCELL * HID;
    for (int c = 0; c < NCELL; ++c) {
        const double x = (c % NXC + 0.5) * DX;
        beta[c] = (x >= X_LO && x <= X_HI) ? BETA : 0.0;
        for (int k = 0; k < NBIN; ++k) { R[(size_t)c * NBIN + k] = r0_of(k); V[(size_t)c * NBIN + k] = 0.0; }
    }

    set_pulse(wp, wm);
#pragma omp target enter data map(to: wp[0:NCELL], wm[0:NCELL], beta[0:NCELL], R[0:nst], V[0:nst]) \
                              map(alloc: wp2[0:NCELL], wm2[0:NCELL], src[0:NCELL])
    double t0 = omp_get_wtime();
    run(wp, wm, wp2, wm2, beta, R, V, H, C, src, 0);
    const double t_ref = omp_get_wtime() - t0;
#pragma omp target exit data map(from: wp[0:NCELL], wm[0:NCELL]) \
                             map(delete: wp2[0:NCELL], wm2[0:NCELL], src[0:NCELL], beta[0:NCELL], R[0:nst], V[0:nst])
    for (int c = 0; c < NCELL; ++c) pref[c] = 0.5 * (wp[c] + wm[c]);

    set_pulse(wp, wm);
#pragma omp target enter data map(to: wp[0:NCELL], wm[0:NCELL], beta[0:NCELL], H[0:nh], C[0:nh]) \
                              map(alloc: wp2[0:NCELL], wm2[0:NCELL], src[0:NCELL])
    t0 = omp_get_wtime();
    run(wp, wm, wp2, wm2, beta, R, V, H, C, src, 1);
    const double t_nn = omp_get_wtime() - t0;
#pragma omp target exit data map(from: wp[0:NCELL], wm[0:NCELL]) \
                             map(delete: wp2[0:NCELL], wm2[0:NCELL], src[0:NCELL], beta[0:NCELL], H[0:nh], C[0:nh])

    double e = 0.0, r = 0.0;
    for (int c = 0; c < NCELL; ++c) {
        const double p = 0.5 * (wp[c] + wm[c]);
        e += (p - pref[c]) * (p - pref[c]); r += pref[c] * pref[c];
    }
    const double err = sqrt(e / r);
    printf("%d lines x %d cells, %d steps; %d bins x %d RK4 sub-steps per cell-step in the reference:\n",
           NB, NXC, NSTEPS, NBIN, N_SUB);
    printf("  reference population  %7.1f ms  (%.1f ns per cell-step)\n", 1e3 * t_ref, 1e9 * t_ref / ((double)NCELL * NSTEPS));
    printf("  LSTM surrogate        %7.1f ms  (%.1f ns per cell-step)\n", 1e3 * t_nn, 1e9 * t_nn / ((double)NCELL * NSTEPS));
    printf("  relative L2 error of the surrogate's pressure field: %.3e\n", err);
    free(wp); free(wm); free(wp2); free(wm2); free(pref); free(beta); free(src); free(R); free(V); free(H); free(C);
    if (!(err < TOL)) { printf("FAIL: error above %.2f\n", TOL); return 1; }
    printf("OK\n");
    return 0;
}
