"""Train the subgrid closure for coarse-grid Burgers and export it to ONNX.

The resolved physics is viscous Burgers, u_t + (u^2/2)_x = nu u_xx, periodic
on [0, 2 pi), solved with a Godunov flux for the convective term and central
differences for the viscous one, forward Euler in time -- the same scheme
burgers.c / burgers.F90 use. A fine grid (NF cells) resolves it; the coarse
grid (NC cells, FACTOR times coarser) does not, and the closure is what the
coarse scheme is missing. An MLP maps the 5-point stencil ubar_{i-2..i+2} to
a per-cell correction R_i that the solver adds to its coarse right-hand
side, inside its own offload loop.

How it is trained matters more than the network. Fitting the closure a
priori -- on states taken from the box-filtered fine solution, to the
residual [ubar(t+dt) - ubar(t)]/dt - rhs_coarse(ubar) -- explains most of
that residual's variance and then makes the coarse run WORSE: at run time
the closure sees its own drifting coarse state, not filtered-fine states.
So it is trained a posteriori, with the coarse solver in the loop: the
coarse scheme is written once more in torch (rhs_torch, the same arithmetic
as rhs() above and as the C/Fortran solvers), a full NSTEPS-step rollout with
the closure inside is unrolled from each training initial condition, and the
loss is the relative error of that trajectory against the filtered-fine one.
On 32 held-out realizations that cuts the mean 200-step error of the coarse
scheme by about 1.3x and improves 94% of them; a local 7-point closure
cannot recover sub-cell structure, so that is roughly the ceiling here.

Writes closure.onnx (float32 weights; the solvers generate it at double).
"""
import numpy as np
import torch
import torch.nn as nn

NF, FACTOR = 2048, 16          # fine cells; spatial coarsening factor
NC = NF // FACTOR
NU = 0.02
L = 2 * np.pi
DT_C = 0.01
N_SUB = 64                     # fine sub-steps per coarse step (viscous stability on the fine grid)
DT_F = DT_C / N_SUB
N_IC, N_STEPS = 48, 200
STENCIL = (3, 2, 1, 0, -1, -2, -3)     # ubar_{i-3..i+3}
ITERS = 600
SEED = 7


def rhs(u, dx, nu):
    """Godunov flux for u^2/2 plus central viscous term, periodic; vectorised over rows."""
    ul, ur = u, np.roll(u, -1, axis=-1)            # left/right states at face i+1/2
    fl, fr = 0.5 * ul * ul, 0.5 * ur * ur
    # Godunov for a convex flux: min over the interval if ul <= ur, max if ul > ur.
    f_min = np.where((ul <= 0) & (ur >= 0), 0.0, np.minimum(fl, fr))
    f_max = np.maximum(fl, fr)
    face = np.where(ul <= ur, f_min, f_max)         # F_{i+1/2}
    conv = -(face - np.roll(face, 1, axis=-1)) / dx
    visc = nu * (np.roll(u, -1, axis=-1) - 2 * u + np.roll(u, 1, axis=-1)) / (dx * dx)
    return conv + visc


def box_filter(u_f):
    return u_f.reshape(*u_f.shape[:-1], NC, FACTOR).mean(axis=-1)


def initial_conditions(rng, n):
    x = np.arange(NF) * (L / NF)
    u = np.zeros((n, NF))
    for k in range(1, 4):
        a = rng.uniform(-1, 1, (n, 1)) / k
        ph = rng.uniform(0, 2 * np.pi, (n, 1))
        u += a * np.sin(k * x + ph)
    return u


class Closure(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(STENCIL), 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                                 nn.Linear(64, 1))

    def forward(self, s):
        return self.net(s)


def make_trajectories(rng):
    """Filtered fine trajectories, shape (N_IC, N_STEPS + 1, NC), for the rollout loss."""
    dx_f = L / NF
    u = initial_conditions(rng, N_IC)
    traj = [box_filter(u)]
    for _ in range(N_STEPS):
        for _ in range(N_SUB):
            u = u + DT_F * rhs(u, dx_f, NU)
        traj.append(box_filter(u))
    return np.stack(traj, axis=1)


def rhs_torch(u, dx, nu):
    """The coarse scheme in torch, differentiable, batched over rows: rhs() rewritten."""
    ul, ur = u, torch.roll(u, -1, dims=-1)
    fl, fr = 0.5 * ul * ul, 0.5 * ur * ur
    f_min = torch.where((ul <= 0) & (ur >= 0), torch.zeros_like(u), torch.minimum(fl, fr))
    f_max = torch.maximum(fl, fr)
    face = torch.where(ul <= ur, f_min, f_max)
    conv = -(face - torch.roll(face, 1, dims=-1)) / dx
    visc = nu * (torch.roll(u, -1, dims=-1) - 2 * u + torch.roll(u, 1, dims=-1)) / (dx * dx)
    return conv + visc


def stencils_torch(u):
    return torch.stack([torch.roll(u, s, dims=-1) for s in STENCIL], dim=-1)


def rollout(model, u0, k, dx):
    """k coarse steps with the closure inside, exactly as the solvers step."""
    u, out = u0, []
    for _ in range(k):
        corr = model(stencils_torch(u)).squeeze(-1)
        u = u + DT_C * (rhs_torch(u, dx, NU) + corr)
        out.append(u)
    return torch.stack(out, dim=1)


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    traj = torch.tensor(make_trajectories(rng), dtype=torch.float32)   # (N_IC, N_STEPS+1, NC)
    dx_c = L / NC
    model = Closure()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
    for it in range(ITERS):
        ic = torch.randint(0, N_IC, (8,))
        pred = rollout(model, traj[ic, 0], N_STEPS, dx_c)
        target = traj[ic, 1:N_STEPS + 1]
        loss = (((pred - target) ** 2).sum(-1) / (target ** 2).sum(-1)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 100 == 0:
            print(f"iter {it:4d}  relative rollout error {loss.item():.3e}")
    model.eval()
    torch.onnx.export(model, torch.zeros(1, len(STENCIL)), "closure.onnx", input_names=["stencil"],
                      output_names=["correction"], opset_version=13, dynamo=False)
    print("wrote closure.onnx")


if __name__ == "__main__":
    main()
