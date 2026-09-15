/* cns.c: a compact 3D compressible Navier-Stokes solver with a learned
 * per-cell closure, in one file.
 *
 * Finite volume on a uniform periodic box: MUSCL reconstruction with a minmod
 * limiter, HLLC flux, full Newtonian viscous stress plus Fourier conduction,
 * SSP-RK3 in time. Everything resident on the device: the state arrays are
 * mapped once before the time loop and nothing is transferred inside it.
 *
 * The closure is a rosenna-generated MLP that maps the nine components of the
 * local velocity-gradient tensor to a turbulent viscosity, evaluated once per
 * cell inside the solver's own offloaded loop (`closure()` below) and added to
 * the molecular viscosity at every face. That is the point of the example: the
 * network is called from inside a `target teams distribute parallel for`, with
 * no allocation, transfer or synchronisation anywhere in the loop path.
 *
 * The network is DETERMINISTICALLY INITIALIZED, NOT TRAINED (see closure.py).
 * Its output is a fixed, arbitrary function of the gradients -- this example
 * demonstrates the plumbing and the numerics around it, not a physical model.
 * So the closure's output is floored at zero and scaled to stay comparable to
 * the molecular viscosity; an actual trained closure would need neither.
 *
 * What the run asserts (it prints OK and exits 0 only if all of it holds):
 *   1. mass and total energy are conserved to round-off, which is what a
 *      conservative flux-difference update in a periodic box owes you;
 *   2. density and pressure stay positive and nothing goes non-finite;
 *   3. the nut field the offloaded solver computed equals a host-side
 *      evaluation of the same model on the same primitives. This is the
 *      roseNNa claim: the generated code gives the same answer on the device
 *      as on the host, inside a real solver rather than a harness.
 *   4. with -DBATCHED, that the batched path (one `closure_infer_batch` call
 *      over the whole field, linking lib<model>.a) agrees with the per-point
 *      path (`closure_infer`, header-inline) as well.
 *
 * The solver's shape -- padded blocks, one array per quantity, a `g` struct of
 * pointers, LOCALS/IDX macros -- follows microfd, a compact compressible solver
 * this example was originally written as a documented patch against. A patch
 * against a file we do not control cannot be compiled or tested, so the solver
 * is now its own implementation. No microfd source is used here.
 */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "closure.h"

#ifndef NX
#define NX 32                  /* interior cells per direction */
#endif
#ifndef NSTEPS
#define NSTEPS 20
#endif

#define NG 2                   /* ghost layers; MUSCL needs two */
#define NV 5                   /* rho, rho u, rho v, rho w, E */
#define PI 3.14159265358979323846
#define NUT_SCALE 1e-3         /* see the note above: the network is untrained */

static struct {
    int n[3];
    double h[3], L[3];
    double gamma, mu, pr, cfl, dt, t;
    double t_closure;                  /* seconds spent in the closure */
    long n_closure;                    /* closure kernel launches */
    double *q, *q1, *qs, *w, *F, *nut;
} g;

/* Padded-block conventions: one block of nc cells per quantity, index
 * c = i + j*sx + k*sy. So c+1 steps +x, c+sx steps +y, c+sy steps +z. */
#define LOCALS                                                    \
    const int nx = g.n[0], ny = g.n[1], nz = g.n[2];              \
    const long sx = nx + 2 * NG;                                  \
    const long sy = (long)(nx + 2 * NG) * (ny + 2 * NG);          \
    const long nc = sy * (long)(nz + 2 * NG);                     \
    (void)nx; (void)ny; (void)nz; (void)sx; (void)sy; (void)nc

#define IDX(i, j, k) ((long)(i) + (long)(j) * sx + (long)(k) * sy)
#define STRIDE(d) ((d) == 0 ? 1L : (d) == 1 ? sx : sy)

/* ------------------------------------------------------------------ halo */

/* Periodic ghost layers, every quantity. Runs on the device over the state
 * array in place; the three directions are separate loops because the corner
 * ghosts must be filled from already-copied faces. */
static void halo(double *q) {
    LOCALS;
    /* x, then y, then z: each pass reads what the previous one filled, which
     * is what gets the edge and corner ghosts right. */
    #pragma omp target teams distribute parallel for collapse(3)
    for (int v = 0; v < NV; v++)
        for (int k = 0; k < nz + 2 * NG; k++)
            for (int j = 0; j < ny + 2 * NG; j++)
                for (int i = 0; i < NG; i++) {
                    q[v * nc + IDX(i, j, k)] = q[v * nc + IDX(i + nx, j, k)];
                    q[v * nc + IDX(nx + NG + i, j, k)] = q[v * nc + IDX(NG + i, j, k)];
                }
    #pragma omp target teams distribute parallel for collapse(3)
    for (int v = 0; v < NV; v++)
        for (int k = 0; k < nz + 2 * NG; k++)
            for (int i = 0; i < nx + 2 * NG; i++)
                for (int j = 0; j < NG; j++) {
                    q[v * nc + IDX(i, j, k)] = q[v * nc + IDX(i, j + ny, k)];
                    q[v * nc + IDX(i, ny + NG + j, k)] = q[v * nc + IDX(i, NG + j, k)];
                }
    #pragma omp target teams distribute parallel for collapse(3)
    for (int v = 0; v < NV; v++)
        for (int j = 0; j < ny + 2 * NG; j++)
            for (int i = 0; i < nx + 2 * NG; i++)
                for (int k = 0; k < NG; k++) {
                    q[v * nc + IDX(i, j, k)] = q[v * nc + IDX(i, j, k + nz)];
                    q[v * nc + IDX(i, j, nz + NG + k)] = q[v * nc + IDX(i, j, NG + k)];
                }
}

/* ------------------------------------------------------------------ prim */

static void prim(const double *q) {
    LOCALS;
    double *w = g.w;
    const double gm = g.gamma - 1;
    #pragma omp target teams distribute parallel for
    for (long c = 0; c < nc; c++) {
        const double r = q[c], u = q[nc + c] / r, v = q[2 * nc + c] / r, s = q[3 * nc + c] / r;
        w[c] = r;
        w[nc + c] = u;
        w[2 * nc + c] = v;
        w[3 * nc + c] = s;
        w[4 * nc + c] = gm * (q[4 * nc + c] - .5 * r * (u * u + v * v + s * s));
    }
}

/* --------------------------------------------------------------- closure */

/* The per-cell closure: nine velocity gradients in, one turbulent viscosity
 * out, once per cell, inside the solver's own offload region. `closure_infer`
 * needs no pragma of its own -- closure.h wraps it in a guarded
 * `omp declare target` region, and with an embedded plan it is `static inline`
 * with the weights baked in, so there is nothing to allocate, transfer or
 * link for this path.
 *
 * The range is the padded block minus one layer on each side: the central
 * differences below read index +/-1 in every direction, halo() has already
 * filled every ghost layer, and face() reads nut one ghost cell into the low
 * boundary of each direction. */
static void closure(void) {
    LOCALS;
    const double *w = g.w;
    double *nut = g.nut;
    const double h0 = g.h[0], h1 = g.h[1], h2 = g.h[2];
    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = 1; k < nz + 2 * NG - 1; k++)
        for (int j = 1; j < ny + 2 * NG - 1; j++)
            for (int i = 1; i < nx + 2 * NG - 1; i++) {
                const long c = IDX(i, j, k);
                const double *u = w + nc + c, *v = w + 2 * nc + c, *s = w + 3 * nc + c;
                double feat[9] = {(u[1] - u[-1]) / (2 * h0), (u[sx] - u[-sx]) / (2 * h1),
                                  (u[sy] - u[-sy]) / (2 * h2), (v[1] - v[-1]) / (2 * h0),
                                  (v[sx] - v[-sx]) / (2 * h1), (v[sy] - v[-sy]) / (2 * h2),
                                  (s[1] - s[-1]) / (2 * h0), (s[sx] - s[-sx]) / (2 * h1),
                                  (s[sy] - s[-sy]) / (2 * h2)};
                double out;
                closure_infer(feat, &out);
                nut[c] = NUT_SCALE * (out > 0 ? out : 0);
            }
}

#ifdef BATCHED
/* The alternative for a larger network: gather every cell's nine features into
 * one device array, then make a single `closure_infer_batch` call instead of
 * one `closure_infer` per cell. Unlike `closure_infer` this is not
 * header-inline, so this path links lib<model>.a.
 *
 * `closure_infer_batch` runs over all nc cells, not just the range the gather
 * loop fills, so the outermost single layer sees uninitialized features and
 * writes a correspondingly meaningless nut. Harmless -- face() never reads nut
 * that far into the ghost region -- but the gather buffer is zeroed once at
 * allocation so the values are at least deterministic. */
static double *g_feat;

static void closure_batched(void) {
    LOCALS;
    const double *w = g.w;
    double *nut = g.nut;
    const double h0 = g.h[0], h1 = g.h[1], h2 = g.h[2];
    double *feat = g_feat;
    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = 1; k < nz + 2 * NG - 1; k++)
        for (int j = 1; j < ny + 2 * NG - 1; j++)
            for (int i = 1; i < nx + 2 * NG - 1; i++) {
                const long c = IDX(i, j, k);
                const double *u = w + nc + c, *v = w + 2 * nc + c, *s = w + 3 * nc + c;
                double *f9 = feat + 9 * c;
                f9[0] = (u[1] - u[-1]) / (2 * h0);
                f9[1] = (u[sx] - u[-sx]) / (2 * h1);
                f9[2] = (u[sy] - u[-sy]) / (2 * h2);
                f9[3] = (v[1] - v[-1]) / (2 * h0);
                f9[4] = (v[sx] - v[-sx]) / (2 * h1);
                f9[5] = (v[sy] - v[-sy]) / (2 * h2);
                f9[6] = (s[1] - s[-1]) / (2 * h0);
                f9[7] = (s[sx] - s[-sx]) / (2 * h1);
                f9[8] = (s[sy] - s[-sy]) / (2 * h2);
            }
    int status;
    /* use_device_addr, not the deprecated use_device_ptr: both arrays are
     * already mapped, and ruling R5 says the call transfers nothing. */
    #pragma omp target data use_device_addr(feat, nut)
    { status = closure_infer_batch((int)nc, feat, nut, 0); }
    if (status) {
        fprintf(stderr, "closure_infer_batch failed: %d\n", status);
        exit(3);
    }
    /* closure.h: "the launch is asynchronous on it". Nothing may read nut
     * until it has finished, and the rescale loop below does. Without this
     * wait the example still printed the right answer, because nvc's OpenMP
     * target regions happen to serialize against the CUDA default stream --
     * an implementation accident, not a guarantee. closure_sync is the
     * backend-agnostic wait: a stream synchronize in the cuda/hip archive,
     * a no-op in the omp one, whose loop is already synchronous. */
    status = closure_sync(0);
    if (status) {
        fprintf(stderr, "closure_sync failed: %d\n", status);
        exit(3);
    }
    /* Scale and floor to match the per-point path. */
    #pragma omp target teams distribute parallel for
    for (long c = 0; c < nc; c++) nut[c] = NUT_SCALE * (nut[c] > 0 ? nut[c] : 0);
}
#endif

/* ------------------------------------------------------------------ face */

#pragma omp declare target
static inline double minmod(double a, double b) {
    if (a * b <= 0) return 0.0;
    return fabs(a) < fabs(b) ? a : b;
}

/* HLLC flux for the Euler part, normal direction `d`, from the two
 * reconstructed primitive states. `fl[NV]` receives the flux. */
static inline void hllc(const double *L, const double *R, int d, double gamma, double *fl) {
    const int mn = 1 + d;                     /* index of the normal momentum */
    const double rL = L[0], pL = L[4], rR = R[0], pR = R[4];
    const double unL = L[1 + d], unR = R[1 + d];
    const double aL = sqrt(gamma * pL / rL), aR = sqrt(gamma * pR / rR);
    const double keL = .5 * rL * (L[1] * L[1] + L[2] * L[2] + L[3] * L[3]);
    const double keR = .5 * rR * (R[1] * R[1] + R[2] * R[2] + R[3] * R[3]);
    const double EL = pL / (gamma - 1) + keL, ER = pR / (gamma - 1) + keR;

    const double SL = fmin(unL - aL, unR - aR), SR = fmax(unL + aL, unR + aR);

    double UL[NV], UR[NV], FL[NV], FR[NV];
    UL[0] = rL; UR[0] = rR;
    for (int i = 0; i < 3; i++) { UL[1 + i] = rL * L[1 + i]; UR[1 + i] = rR * R[1 + i]; }
    UL[4] = EL; UR[4] = ER;
    for (int v = 0; v < NV; v++) { FL[v] = unL * UL[v]; FR[v] = unR * UR[v]; }
    FL[mn] += pL; FR[mn] += pR;
    FL[4] += unL * pL; FR[4] += unR * pR;

    if (SL >= 0) { for (int v = 0; v < NV; v++) fl[v] = FL[v]; return; }
    if (SR <= 0) { for (int v = 0; v < NV; v++) fl[v] = FR[v]; return; }

    const double num = pR - pL + rL * unL * (SL - unL) - rR * unR * (SR - unR);
    const double den = rL * (SL - unL) - rR * (SR - unR);
    const double SM = num / den;

    /* Star states (Toro's HLLC). The chosen side is COPIED rather than
     * selected through a pointer: `const double *S = SM >= 0 ? L : R` over
     * local arrays miscompiles under nvc -O2 -mp=gpu, which turns this kernel
     * into a launch failure (CUDA_ERROR_LAUNCH_FAILED) while -O1 is correct.
     * Copying five doubles costs nothing next to the flux arithmetic. */
    double S[NV], Uk[NV], Fk[NV], Sk;
    if (SM >= 0) {
        Sk = SL;
        for (int v = 0; v < NV; v++) { S[v] = L[v]; Uk[v] = UL[v]; Fk[v] = FL[v]; }
    } else {
        Sk = SR;
        for (int v = 0; v < NV; v++) { S[v] = R[v]; Uk[v] = UR[v]; Fk[v] = FR[v]; }
    }
    const double rk = S[0], pk = S[4], unk = S[1 + d];
    const double Ek = pk / (gamma - 1) + .5 * rk * (S[1] * S[1] + S[2] * S[2] + S[3] * S[3]);
    const double fac = rk * (Sk - unk) / (Sk - SM);

    double Us[NV];
    Us[0] = fac;
    for (int i = 0; i < 3; i++) Us[1 + i] = fac * (i == d ? SM : S[1 + i]);
    Us[4] = fac * (Ek / rk + (SM - unk) * (SM + pk / (rk * (Sk - unk))));

    for (int v = 0; v < NV; v++) fl[v] = Fk[v] + Sk * (Us[v] - Uk[v]);
}
#pragma omp end declare target

/* Flux through the face at c+1/2 normal to d, for every interior face,
 * stored in F[d]. Euler part by MUSCL+HLLC on the primitives; viscous part by
 * face-centred gradients with mu_eff = mu + rho*nut from the closure. */
static void face(int d) {
    LOCALS;
    const double *w = g.w, *nut = g.nut;
    /* The base pointer, not a pre-offset one: `g.F + d*NV*nc` computed here on
     * the host is an address in the middle of the mapped region, and the
     * runtime translates the base of a mapping, not an interior address --
     * capturing the offset pointer dereferences a host address on the device.
     * The `d` offset goes into the index below instead. */
    double *F = g.F;
    const long dof = (long)d * NV * nc;
    const long s = STRIDE(d);
    const long st[3] = {1, sx, sy};
    const double hd = g.h[d], gamma = g.gamma, mu0 = g.mu, pr = g.pr;
    const double h[3] = {g.h[0], g.h[1], g.h[2]};
    const int i0 = NG - (d == 0), j0 = NG - (d == 1), k0 = NG - (d == 2);

    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = k0; k < nz + NG; k++)
        for (int j = j0; j < ny + NG; j++)
            for (int i = i0; i < nx + NG; i++) {
                const long c = IDX(i, j, k);

                /* --- MUSCL: limited slopes, then the two face states. */
                double L[NV], R[NV];
                for (int v = 0; v < NV; v++) {
                    const double *a = w + v * nc + c;
                    const double dL = minmod(a[0] - a[-s], a[s] - a[0]);
                    const double dR = minmod(a[s] - a[0], a[2 * s] - a[s]);
                    L[v] = a[0] + .5 * dL;
                    R[v] = a[s] - .5 * dR;
                }
                /* A limiter cannot guarantee positivity of a reconstructed
                 * state on a coarse grid; fall back to first order where it
                 * fails, which is what keeps the scheme robust. */
                if (L[0] <= 0 || L[4] <= 0 || R[0] <= 0 || R[4] <= 0)
                    for (int v = 0; v < NV; v++) { L[v] = w[v * nc + c]; R[v] = w[v * nc + c + s]; }

                double fl[NV];
                hllc(L, R, d, gamma, fl);

                /* --- viscous: the 3x3 velocity gradient at the face. Normal
                 * derivative across the face; tangential ones averaged from
                 * the two cells' central differences. */
                double du[3][3];
                for (int a = 0; a < 3; a++) {
                    const double *ua = w + (1 + a) * nc + c;
                    for (int b = 0; b < 3; b++) {
                        if (b == d) {
                            du[a][b] = (ua[s] - ua[0]) / hd;
                        } else {
                            const long t = st[b];
                            du[a][b] = .5 * ((ua[t] - ua[-t]) / (2 * h[b]) +
                                             (ua[s + t] - ua[s - t]) / (2 * h[b]));
                        }
                    }
                }
                const double rf = .5 * (w[c] + w[c + s]);
                const double nf = .5 * (nut[c] + nut[c + s]);
                const double mu = mu0 + rf * nf;               /* the closure enters here */
                const double div = du[0][0] + du[1][1] + du[2][2];

                double tau[3];
                for (int a = 0; a < 3; a++)
                    tau[a] = mu * (du[a][d] + du[d][a]) - (a == d ? 2.0 / 3.0 * mu * div : 0.0);

                /* Fourier conduction, with T = p/rho (R = 1) and
                 * kappa = mu*cp/Pr. */
                const double cp = gamma / (gamma - 1);
                const double TL = w[4 * nc + c] / w[c], TR = w[4 * nc + c + s] / w[c + s];
                const double qd = -(mu * cp / pr) * (TR - TL) / hd;

                double uf[3];
                for (int a = 0; a < 3; a++) uf[a] = .5 * (w[(1 + a) * nc + c] + w[(1 + a) * nc + c + s]);

                for (int a = 0; a < 3; a++) fl[1 + a] -= tau[a];
                fl[4] -= uf[0] * tau[0] + uf[1] * tau[1] + uf[2] * tau[2];
                fl[4] += qd;

                for (int v = 0; v < NV; v++) F[dof + v * nc + c] = fl[v];
            }
}

/* ------------------------------------------------------------------- rhs */

/* out = src + dt * (-div F), over the interior. */
static void advance(const double *src, double *out, double dt) {
    LOCALS;
    const double *F = g.F;
    const double h[3] = {g.h[0], g.h[1], g.h[2]};
    const long st[3] = {1, sx, sy};
    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = NG; k < nz + NG; k++)
        for (int j = NG; j < ny + NG; j++)
            for (int i = NG; i < nx + NG; i++) {
                const long c = IDX(i, j, k);
                for (int v = 0; v < NV; v++) {
                    double div = 0;
                    for (int d = 0; d < 3; d++) {
                        const double *Fd = F + (long)d * NV * nc + v * nc;
                        div += (Fd[c] - Fd[c - st[d]]) / h[d];
                    }
                    out[v * nc + c] = src[v * nc + c] - dt * div;
                }
            }
}

static void rhs_eval(double *q) {
    halo(q);
    prim(q);
    /* Every target region in this file is synchronous -- none carries
     * `nowait` -- so wall-clock around the call is the kernel's own cost. */
    const double t0 = omp_get_wtime();
#ifdef BATCHED
    closure_batched();
#else
    closure();
#endif
    g.t_closure += omp_get_wtime() - t0;
    g.n_closure++;
    for (int d = 0; d < 3; d++) face(d);
}

/* SSP-RK3. q1 and qs are scratch, both device-resident. */
static void step(void) {
    LOCALS;
    double *q = g.q, *q1 = g.q1, *qs = g.qs;
    const double dt = g.dt;

    rhs_eval(q);
    advance(q, q1, dt);

    rhs_eval(q1);
    advance(q1, qs, dt);
    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = NG; k < nz + NG; k++)
        for (int j = NG; j < ny + NG; j++)
            for (int i = NG; i < nx + NG; i++) {
                const long c = IDX(i, j, k);
                for (int v = 0; v < NV; v++)
                    q1[v * nc + c] = .75 * q[v * nc + c] + .25 * qs[v * nc + c];
            }

    rhs_eval(q1);
    advance(q1, qs, dt);
    #pragma omp target teams distribute parallel for collapse(3)
    for (int k = NG; k < nz + NG; k++)
        for (int j = NG; j < ny + NG; j++)
            for (int i = NG; i < nx + NG; i++) {
                const long c = IDX(i, j, k);
                for (int v = 0; v < NV; v++)
                    q[v * nc + c] = (1.0 / 3.0) * q[v * nc + c] + (2.0 / 3.0) * qs[v * nc + c];
            }

    g.t += dt;
}

/* ------------------------------------------------------------------ init */

/* Compressible Taylor-Green: smooth, periodic, and it produces a full
 * velocity-gradient tensor, which is what the closure consumes. */
static void init(void) {
    LOCALS;
    const double M = 0.1, gamma = g.gamma;
    const double p0 = 1.0 / (gamma * M * M), r0 = 1.0;
    for (int k = 0; k < nz + 2 * NG; k++)
        for (int j = 0; j < ny + 2 * NG; j++)
            for (int i = 0; i < nx + 2 * NG; i++) {
                const long c = IDX(i, j, k);
                const double x = (i - NG + .5) * g.h[0], y = (j - NG + .5) * g.h[1],
                             z = (k - NG + .5) * g.h[2];
                const double u = sin(x) * cos(y) * cos(z);
                const double v = -cos(x) * sin(y) * cos(z);
                const double s = 0.0;
                const double p = p0 + (r0 / 16.0) * (cos(2 * x) + cos(2 * y)) * (cos(2 * z) + 2.0);
                g.q[c] = r0;
                g.q[nc + c] = r0 * u;
                g.q[2 * nc + c] = r0 * v;
                g.q[3 * nc + c] = r0 * s;
                g.q[4 * nc + c] = p / (gamma - 1) + .5 * r0 * (u * u + v * v + s * s);
            }
}

/* ---------------------------------------------------------------- checks */

static void totals(const double *q, double *mass, double *energy) {
    LOCALS;
    double m = 0, e = 0;
    for (int k = NG; k < nz + NG; k++)
        for (int j = NG; j < ny + NG; j++)
            for (int i = NG; i < nx + NG; i++) {
                const long c = IDX(i, j, k);
                m += q[c];
                e += q[4 * nc + c];
            }
    const double dv = g.h[0] * g.h[1] * g.h[2];
    *mass = m * dv;
    *energy = e * dv;
}

/* The host's own evaluation of the same model on the same primitives, for
 * comparison against what the offloaded closure() wrote. */
static double nut_mismatch(void) {
    LOCALS;
    const double *w = g.w;
    const double h0 = g.h[0], h1 = g.h[1], h2 = g.h[2];
    double worst = 0;
    for (int k = 1; k < nz + 2 * NG - 1; k++)
        for (int j = 1; j < ny + 2 * NG - 1; j++)
            for (int i = 1; i < nx + 2 * NG - 1; i++) {
                const long c = IDX(i, j, k);
                const double *u = w + nc + c, *v = w + 2 * nc + c, *s = w + 3 * nc + c;
                double feat[9] = {(u[1] - u[-1]) / (2 * h0), (u[sx] - u[-sx]) / (2 * h1),
                                  (u[sy] - u[-sy]) / (2 * h2), (v[1] - v[-1]) / (2 * h0),
                                  (v[sx] - v[-sx]) / (2 * h1), (v[sy] - v[-sy]) / (2 * h2),
                                  (s[1] - s[-1]) / (2 * h0), (s[sx] - s[-sx]) / (2 * h1),
                                  (s[sy] - s[-sy]) / (2 * h2)};
                double out;
                closure_infer(feat, &out);
                const double ref = NUT_SCALE * (out > 0 ? out : 0);
                const double d = fabs(g.nut[c] - ref);
                if (d > worst) worst = d;
            }
    return worst;
}

int main(void) {
    g.n[0] = g.n[1] = g.n[2] = NX;
    g.L[0] = g.L[1] = g.L[2] = 2 * PI;
    g.gamma = 1.4;
    g.mu = 1e-3;
    g.pr = 0.72;
    g.cfl = 0.4;
    for (int d = 0; d < 3; d++) g.h[d] = g.L[d] / g.n[d];
    LOCALS;

    const size_t blk = sizeof(double) * NV * nc;
    g.q = malloc(blk); g.q1 = malloc(blk); g.qs = malloc(blk);
    g.w = malloc(blk); g.F = malloc(blk * 3); g.nut = malloc(sizeof(double) * nc);
    if (!g.q || !g.q1 || !g.qs || !g.w || !g.F || !g.nut) return 2;
    memset(g.nut, 0, sizeof(double) * nc);
    memset(g.F, 0, blk * 3);
    init();

    /* One fixed dt from the initial state: a device reduction per step would
     * be a transfer in the loop path, which is the thing this example is
     * demonstrating the absence of. */
    double smax = 0;
    for (long c = 0; c < nc; c++) {
        const double r = g.q[c], u = g.q[nc + c] / r, v = g.q[2 * nc + c] / r,
                     s = g.q[3 * nc + c] / r;
        const double p = (g.gamma - 1) * (g.q[4 * nc + c] - .5 * r * (u * u + v * v + s * s));
        const double a = sqrt(g.gamma * p / r);
        const double sp = fmax(fabs(u), fmax(fabs(v), fabs(s))) + a;
        if (sp > smax) smax = sp;
    }
    const double hmin = fmin(g.h[0], fmin(g.h[1], g.h[2]));
    g.dt = g.cfl * hmin / (3 * smax);

    double m0, e0;
    totals(g.q, &m0, &e0);

    double *q = g.q, *q1 = g.q1, *qs = g.qs, *w = g.w, *F = g.F, *nut = g.nut;
    /* These are used only in the map clauses below, which a compiler built
     * without offload support treats as no-ops -- hence -Wunused there. */
    (void)q1; (void)qs; (void)w; (void)F; (void)nut;
#ifdef BATCHED
    g_feat = malloc(sizeof(double) * 9 * nc);
    if (!g_feat) return 2;
    memset(g_feat, 0, sizeof(double) * 9 * nc);
    double *feat = g_feat;
#endif
    /* Mapped once, outside the time loop. Nothing below transfers. */
    #pragma omp target enter data map(to: q[0:NV*nc], nut[0:nc]) \
        map(alloc: q1[0:NV*nc], qs[0:NV*nc], w[0:NV*nc], F[0:3*NV*nc])
#ifdef BATCHED
    #pragma omp target enter data map(to: feat[0:9*nc])
#endif

    /* One untimed rhs_eval first: the first launch of each kernel pays for
     * module load and any JIT, which would otherwise land entirely in step 1
     * and dominate a short run. rhs_eval writes only w, nut and F, never q,
     * so this does not change the answer. */
    rhs_eval(g.q);
    g.t_closure = 0;
    g.n_closure = 0;

    const double t_loop0 = omp_get_wtime();
    for (int n = 0; n < NSTEPS; n++) step();
    const double t_loop = omp_get_wtime() - t_loop0;

    #pragma omp target exit data map(from: q[0:NV*nc], w[0:NV*nc], nut[0:nc]) \
        map(release: q1[0:NV*nc], qs[0:NV*nc], F[0:3*NV*nc])
#ifdef BATCHED
    #pragma omp target exit data map(release: feat[0:9*nc])
#endif

    double m1, e1;
    totals(q, &m1, &e1);

    /* 4. The two ways of calling the model, compared against each other
     * directly rather than each against the host. The batched path ran during
     * the time loop; re-run the per-point path on the same final primitives
     * and require the same nut. Outside the loop, so it costs the loop path
     * nothing. */
    double nut_paths = 0;
#ifdef BATCHED
    double *nut_b = malloc(sizeof(double) * nc);
    if (!nut_b) return 2;
    memcpy(nut_b, nut, sizeof(double) * nc);
    #pragma omp target enter data map(to: w[0:NV*nc], nut[0:nc])
    closure();
    #pragma omp target exit data map(from: nut[0:nc]) map(release: w[0:NV*nc])
    for (int k = 1; k < nz + 2 * NG - 1; k++)
        for (int j = 1; j < ny + 2 * NG - 1; j++)
            for (int i = 1; i < nx + 2 * NG - 1; i++) {
                const long c = IDX(i, j, k);
                const double d = fabs(nut_b[c] - nut[c]);
                if (d > nut_paths) nut_paths = d;
            }
#endif

    /* 1. conservation */
    const double dm = fabs(m1 - m0) / m0, de = fabs(e1 - e0) / e0;
    /* 2. positivity and finiteness */
    long bad = 0;
    double rmin = 1e300, pmin = 1e300;
    for (int k = NG; k < nz + NG; k++)
        for (int j = NG; j < ny + NG; j++)
            for (int i = NG; i < nx + NG; i++) {
                const long c = IDX(i, j, k);
                const double r = q[c], u = q[nc + c] / r, v = q[2 * nc + c] / r,
                             s = q[3 * nc + c] / r;
                const double p = (g.gamma - 1) * (q[4 * nc + c] - .5 * r * (u * u + v * v + s * s));
                if (!isfinite(r) || !isfinite(p)) { bad++; continue; }
                if (r < rmin) rmin = r;
                if (p < pmin) pmin = p;
            }
    /* 3. the closure agrees with a host evaluation of the same model */
    const double nerr = nut_mismatch();

    /* The closure runs over the padded block minus one layer on each side,
     * which is the count to divide by -- not the interior cell count. */
    const long ncl = (long)(nx + 2 * NG - 2) * (ny + 2 * NG - 2) * (nz + 2 * NG - 2);
    const double per_cell_ns = 1e9 * g.t_closure / (double)(g.n_closure * ncl);

    printf("%dx%dx%d, %d steps, dt %.3e, t %.4f\n", NX, NX, NX, NSTEPS, g.dt, g.t);
    printf("  loop            %8.2f ms total, %7.3f ms/step\n", 1e3 * t_loop,
           1e3 * t_loop / NSTEPS);
    printf("  closure         %8.2f ms total (%4.1f%% of the loop), %ld launches\n",
           1e3 * g.t_closure, 100 * g.t_closure / t_loop, g.n_closure);
    printf("                  %8.2f ns per cell per call, over %ld cells\n",
           per_cell_ns, ncl);
    printf("  mass drift      %.3e (relative)\n", dm);
    printf("  energy drift    %.3e (relative)\n", de);
    printf("  min rho / p     %.6f / %.6f   non-finite cells %ld\n", rmin, pmin, bad);
    printf("  closure nut vs host evaluation: worst |device-host| %.3e\n", nerr);
#ifdef BATCHED
    printf("  batched vs per-point: worst |infer_batch-infer| %.3e\n", nut_paths);
#endif

    const int ok = dm < 1e-12 && de < 1e-12 && bad == 0 && rmin > 0 && pmin > 0 &&
                   nerr < 1e-12 && nut_paths < 1e-12;
    puts(ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
