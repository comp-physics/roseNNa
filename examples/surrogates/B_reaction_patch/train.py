"""Train a learned time-stepper for 2-D FitzHugh-Nagumo and export it to ONNX.

The resolved physics is the reaction-diffusion system

    u_t = Du lap(u) + u - u^3/3 - v
    v_t = Dv lap(v) + eps (u + a - b v)

on a periodic grid, explicit Euler with a 5-point Laplacian and a small time
step DT -- the same scheme react.c / react.F90 use. The surrogate takes one
BIG step of K fine steps at once: from the 3x3 patch of (u, v) around a cell
it predicts that cell's (u, v) K fine steps later. K DT of diffusion spreads
information about sqrt(2 Du K DT) ~ 1.4 cells, which is what a 3x3 patch
sees, so the stepper is well posed; a wider patch would be needed for a
bigger K.

It is a stepper, not a closure, so the solver's loop is: gather every cell's
patch into one array, ONE batched call (stepper_infer_batch over the whole
field, on device-resident data), scatter the result back -- the structure a
larger model wants, and one that needs the file-loaded form with an init
call, which this example forces with --no-embed.

Training is a fit of the one-step map on every (patch, centre K steps
later) pair, with Gaussian noise (NOISE) added to the input patches. The
noise is what makes the rollout stable: fitted on clean inputs the map is
1% accurate per step and blows up after ~40 of its own steps, because it
never learned to contract the perturbations it creates; fitted on noisy
inputs it tracks the fine solution for 100 steps (1000 fine steps) to
within a few percent, with the largest error in the fast early transient.
Unrolled (a posteriori) training, which A needs, was tried here and made
this map worse: the noise does the same job more cheaply for a stepper.

Writes stepper.onnx.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 64                       # training grid (periodic); the solvers use a bigger one
DU, DV = 1.0, 0.05
A, B, EPS = 0.7, 0.8, 0.08
DT, K = 0.1, 10              # fine step; fine steps per surrogate step
N_FIELDS, N_BIG = 32, 60     # training fields, big steps each
NOISE = 0.03                 # input noise during the fit; see above
ITERS = 5000
N_EVAL = 100                 # held-out rollout length, in big steps
SEED = 11


def lap(f):
    return (np.roll(f, 1, -1) + np.roll(f, -1, -1) + np.roll(f, 1, -2) + np.roll(f, -1, -2) - 4 * f)


def fine_step(u, v):
    un = u + DT * (DU * lap(u) + u - u ** 3 / 3 - v)
    vn = v + DT * (DV * lap(v) + EPS * (u + A - B * v))
    return un, vn


def initial_fields(rng, n):
    """Smooth random fields: a few Fourier modes with random phases, then a nonlinearity."""
    x = np.arange(N) / N
    kx, ky = np.meshgrid(x, x, indexing="ij")
    u = np.zeros((n, N, N)); v = np.zeros((n, N, N))
    for f in range(n):
        for _ in range(4):
            p, q = rng.integers(1, 4, 2)
            u[f] += rng.uniform(-1, 1) * np.sin(2 * np.pi * (p * kx + q * ky) + rng.uniform(0, 2 * np.pi))
            v[f] += rng.uniform(-0.5, 0.5) * np.sin(2 * np.pi * (p * kx + q * ky) + rng.uniform(0, 2 * np.pi))
    return 2.0 * np.tanh(u), 0.5 * np.tanh(v)


def make_trajectories(rng):
    """(N_FIELDS, N_BIG + 1, 2, N, N): the fine solution sampled every K steps."""
    u, v = initial_fields(rng, N_FIELDS)
    out = [np.stack([u, v], 1)]
    for _ in range(N_BIG):
        for _ in range(K):
            u, v = fine_step(u, v)
        out.append(np.stack([u, v], 1))
    return np.stack(out, 1)


class Stepper(nn.Module):
    """3x3 patch of (u, v), 18 values, -> (u, v) at the centre one big step later."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(18, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(),
                                 nn.Linear(128, 2))

    def forward(self, patch):
        return self.net(patch)


def patches(field):
    """(B, 2, N, N) -> (B, N, N, 18): periodic 3x3 patches, u's 9 values then v's, row-major."""
    f = F.pad(field, (1, 1, 1, 1), mode="circular")
    p = f.unfold(2, 3, 1).unfold(3, 3, 1)             # (B, 2, N, N, 3, 3)
    return p.reshape(field.shape[0], 2, N, N, 9).permute(0, 2, 3, 1, 4).reshape(field.shape[0], N, N, 18)


def big_step(model, field):
    """One surrogate step over the whole field: gather, MLP, scatter -- as the solvers do."""
    return model(patches(field)).permute(0, 3, 1, 2)


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    traj = torch.tensor(make_trajectories(rng), dtype=torch.float32)   # (F, N_BIG+1, 2, N, N)
    model = Stepper()

    X = patches(traj[:, :-1].reshape(-1, 2, N, N)).reshape(-1, 18)
    Y = traj[:, 1:].reshape(-1, 2, N, N).permute(0, 2, 3, 1).reshape(-1, 2)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
    for it in range(ITERS):
        idx = torch.randint(0, X.shape[0], (8192,))
        loss = ((model(X[idx] + NOISE * torch.randn(len(idx), 18)) - Y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 1000 == 0:
            print(f"iter {it:4d}  one-step mse (noisy inputs) {loss.item():.3e}")
    model.eval()
    with torch.no_grad():
        # Full-trajectory check on a fresh field.
        rng2 = np.random.default_rng(SEED + 1)
        u, v = initial_fields(rng2, 4)
        state = torch.tensor(np.stack([u, v], 1), dtype=torch.float32)
        uf, vf = u, v
        for s in range(1, N_EVAL + 1):
            state = big_step(model, state)
            for _ in range(K):
                uf, vf = fine_step(uf, vf)
            if s in (1, 5, 20, 60, N_EVAL):
                ref = np.stack([uf, vf], 1)
                err = np.sqrt(((state.numpy() - ref) ** 2).sum() / (ref ** 2).sum())
                print(f"held-out fields, {s:3d} surrogate steps: relative L2 error {err:.3e}")
    torch.onnx.export(model, torch.zeros(1, 18), "stepper.onnx", input_names=["patch"],
                      output_names=["uv_next"], opset_version=13, dynamo=False)
    print("wrote stepper.onnx")


if __name__ == "__main__":
    main()
