/* 1-D acoustics through a bubbly region, with a recurrent per-cell surrogate, C.
 *
 * Pattern: a stateful surrogate. Every cell owns an LSTM state (h, c) that
 * lives on the device for the whole run; each step the solver hands the
 * model x = [p', h, c] and gets y = [s, h', c'] back, copies the state
 * slices of y into its own state arrays, and uses s -- the population's
 * volume-fraction rate -- as the acoustic source. The model has three graph
 * inputs and three graph outputs; the generator lays them out concatenated
 * (`rosenna info bubbles.onnx` prints x: p[0:1] h[1:33] c[33:65] and
 * y: s[0:1] h_next[1:33] c_next[33:65]), which is what the offsets below
 * are. Embedded model, header-inline, no init.
 *
 * Physics (nondimensional, rho = c = p_ambient = 1): linear acoustics in
 * characteristic form, w+ = p + u advected right, w- = p - u left, each
 * with the source -beta s; first-order upwind at CFL = 1, which is exact
 * transport. beta > 0 only inside the bubbly region. The reference
 * integrates NBIN Rayleigh-Plesset bins per cell with RK4 in N_SUB
 * sub-steps per acoustic step (the model train.py fitted); the surrogate
 * replaces that population with one LSTM step per cell.
 *
 * The problem is an ensemble of NB lines with different pulse amplitudes
 * and widths (one line is too small for a GPU); the program prints the
 * relative L2 error of the surrogate's pressure field against the
 * reference after NSTEPS steps, both timings, and exits 0 if the error is
 * under TOL. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "bubbles.h"

#ifndef NB
#define NB 64              /* ensemble size; -DNB=4 for a quick host run */
#endif
#define NX 512
#define NCELL (NB * NX)
#define DX 0.05
#define DT 0.05            /* CFL = 1 */
#define NSTEPS 400
#define BETA 0.1           /* bubble volume-fraction coupling inside the region; the explicit
                              coupling goes unstable above ~0.2 at this dt */
#define X_BUBBLY_LO 10.0
#define X_BUBBLY_HI 20.0
#define TOL 0.10
/* The population train.py integrated: NBIN bins, log-spaced rest radii. */
#define NBIN 8
#define GAMMA 1.4
#define MU 0.05
#define N_SUB 10
/* The model's x / y layouts, from `rosenna info bubbles.onnx`. */
#define HID 32
#define NIN (1 + 2 * HID)
#define NOUT (1 + 2 * HID)

static double r0_of(int k) { return 0.5 * pow(4.0, (double)k / (NBIN - 1)); }   /* geomspace(0.5, 2, NBIN) */

/* --- the reference: one acoustic step's worth of RK4 sub-steps on every bin --- */
static void rp_rhs(double R, double V, double p, double r0, double *dR, double *dV) {
    const double pg = pow(r0 / R, 3.0 * GAMMA);
    *dR = V;
    *dV = (pg - 1.0 - p - 4.0 * MU * V / R - 1.5 * V * V) / R;
}

static double population_step(double *R, double *V, double p) {
    /* Advance the NBIN bins of one cell through N_SUB RK4 sub-steps under
       pressure p; return s = sum_k w_k 3 R_k^2 V_k / R0_k^3 afterwards. */
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

/* --- one acoustic step: sources from the bubbles (reference or surrogate), then transport --- */
static void step(const double *wp, const double *wm, double *wp_new, double *wm_new, const double *beta,
                 double *R, double *V, double *H, double *C, double *src, int use_nn) {
    /* 1. Every cell's source from its own bubbles, given its pressure. */
#pragma omp target teams distribute parallel for
    for (int c = 0; c < NCELL; ++c) {
        const double p = 0.5 * (wp[c] + wm[c]);
        if (use_nn) {
            double x[NIN], y[NOUT];
            x[0] = p;
            for (int i = 0; i < HID; ++i) { x[1 + i] = H[(size_t)c * HID + i]; x[1 + HID + i] = C[(size_t)c * HID + i]; }
            bubbles_infer(x, y);                          /* the surrogate: one LSTM step */
            src[c] = y[0];
            for (int i = 0; i < HID; ++i) { H[(size_t)c * HID + i] = y[1 + i]; C[(size_t)c * HID + i] = y[1 + HID + i]; }
        } else {
            src[c] = population_step(R + (size_t)c * NBIN, V + (size_t)c * NBIN, p);
        }
    }
    /* 2. Exact upwind transport at CFL = 1, with the source; outflow ends. */
#pragma omp target teams distribute parallel for collapse(2)
    for (int b = 0; b < NB; ++b)
        for (int i = 0; i < NX; ++i) {
            const int c = b * NX + i;
            const int l = b * NX + (i > 0 ? i - 1 : 0), r = b * NX + (i < NX - 1 ? i + 1 : NX - 1);
            wp_new[c] = wp[l] - DT * beta[c] * src[c];
            wm_new[c] = wm[r] - DT * beta[c] * src[c];
        }
}

static double hash01(long long a, long long b) {
    return (double)(((a * 40503LL + b) * 2654435761LL) % 4294967296LL) / 4294967296.0;
}

int main(void) {
    double *wp = malloc(sizeof(double) * NCELL), *wm = malloc(sizeof(double) * NCELL);
    double *wp2 = malloc(sizeof(double) * NCELL), *wm2 = malloc(sizeof(double) * NCELL);
    double *pref = malloc(sizeof(double) * NCELL), *beta = malloc(sizeof(double) * NCELL);
    double *src = malloc(sizeof(double) * NCELL);
    double *R = malloc(sizeof(double) * NCELL * NBIN), *V = malloc(sizeof(double) * NCELL * NBIN);
    double *H = malloc(sizeof(double) * NCELL * HID), *C = malloc(sizeof(double) * NCELL * HID);

    /* A right-going pulse per line, amplitude and width hashed; bubbles at rest. */
    for (int b = 0; b < NB; ++b) {
        const double amp = 0.1 + 0.25 * hash01(b + 1, 1), sig = 0.5 + 1.0 * hash01(b + 1, 2);
        for (int i = 0; i < NX; ++i) {
            const int c = b * NX + i;
            const double x = (i + 0.5) * DX;
            const double p = amp * exp(-0.5 * ((x - 4.0) / sig) * ((x - 4.0) / sig));
            wp[c] = 2.0 * p; wm[c] = 0.0;                /* p = u = pulse: purely right-going */
            beta[c] = (x >= X_BUBBLY_LO && x <= X_BUBBLY_HI) ? BETA : 0.0;
            for (int k = 0; k < NBIN; ++k) { R[(size_t)c * NBIN + k] = r0_of(k); V[(size_t)c * NBIN + k] = 0.0; }
            for (int i2 = 0; i2 < HID; ++i2) { H[(size_t)c * HID + i2] = 0.0; C[(size_t)c * HID + i2] = 0.0; }
        }
    }
    const size_t nst = (size_t)NCELL * NBIN, nh = (size_t)NCELL * HID;
    double *pw = wp, *pm = wm, *qw = wp2, *qm = wm2;

    /* Reference: the polydisperse population per cell, RK4-substepped. */
#pragma omp target enter data map(to: wp[0:NCELL], wm[0:NCELL], beta[0:NCELL], R[0:nst], V[0:nst]) \
                              map(alloc: wp2[0:NCELL], wm2[0:NCELL], src[0:NCELL])
    double t0 = omp_get_wtime();
    for (int s = 0; s < NSTEPS; ++s) {
        step(pw, pm, qw, qm, beta, R, V, H, C, src, 0);
        double *t = pw; pw = qw; qw = t; t = pm; pm = qm; qm = t;
    }
    const double t_ref = omp_get_wtime() - t0;
#pragma omp target exit data map(from: wp[0:NCELL], wm[0:NCELL], wp2[0:NCELL], wm2[0:NCELL]) \
                             map(delete: R[0:nst], V[0:nst], src[0:NCELL], beta[0:NCELL])
    for (int c = 0; c < NCELL; ++c) pref[c] = 0.5 * (pw[c] + pm[c]);

    /* Surrogate: same pulse, LSTM state per cell resident on the device. */
    for (int b = 0; b < NB; ++b) {
        const double amp = 0.1 + 0.25 * hash01(b + 1, 1), sig = 0.5 + 1.0 * hash01(b + 1, 2);
        for (int i = 0; i < NX; ++i) {
            const double x = (i + 0.5) * DX;
            wp[b * NX + i] = 2.0 * amp * exp(-0.5 * ((x - 4.0) / sig) * ((x - 4.0) / sig));
            wm[b * NX + i] = 0.0;
        }
    }
    pw = wp; pm = wm; qw = wp2; qm = wm2;
#pragma omp target enter data map(to: wp[0:NCELL], wm[0:NCELL], beta[0:NCELL], H[0:nh], C[0:nh]) \
                              map(alloc: wp2[0:NCELL], wm2[0:NCELL], src[0:NCELL])
    t0 = omp_get_wtime();
    for (int s = 0; s < NSTEPS; ++s) {
        step(pw, pm, qw, qm, beta, R, V, H, C, src, 1);
        double *t = pw; pw = qw; qw = t; t = pm; pm = qm; qm = t;
    }
    const double t_nn = omp_get_wtime() - t0;
#pragma omp target exit data map(from: wp[0:NCELL], wm[0:NCELL], wp2[0:NCELL], wm2[0:NCELL]) \
                             map(delete: H[0:nh], C[0:nh], src[0:NCELL], beta[0:NCELL])

    double e = 0.0, r = 0.0;
    for (int c = 0; c < NCELL; ++c) {
        const double p = 0.5 * (pw[c] + pm[c]);
        e += (p - pref[c]) * (p - pref[c]); r += pref[c] * pref[c];
    }
    const double err = sqrt(e / r);
    printf("%d lines x %d cells, %d steps through a bubbly region (%d bins, %d RK4 sub-steps per step):\n",
           NB, NX, NSTEPS, NBIN, N_SUB);
    printf("  reference population   %7.1f ms  (%.1f ns per cell-step)\n", 1e3 * t_ref, 1e9 * t_ref / ((double)NCELL * NSTEPS));
    printf("  LSTM surrogate         %7.1f ms  (%.1f ns per cell-step)\n", 1e3 * t_nn, 1e9 * t_nn / ((double)NCELL * NSTEPS));
    printf("  relative L2 error of the surrogate's pressure field: %.3e\n", err);
    free(wp); free(wm); free(wp2); free(wm2); free(pref); free(beta); free(src); free(R); free(V); free(H); free(C);
    if (!(err < TOL)) { printf("FAIL: error above %.2f\n", TOL); return 1; }
    printf("OK\n");
    return 0;
}
