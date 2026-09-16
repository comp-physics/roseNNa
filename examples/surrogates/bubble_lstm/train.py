"""Train a recurrent surrogate for a polydisperse bubble population; writes bubbles.onnx.

Per cell of the acoustic solver, NBIN Rayleigh-Plesset bins driven by the
cell's pressure p'(t),

    R R'' + 3/2 R'^2 = p_g(R) - 1 - p' - 4 mu R'/R,   p_g = (R0/R)^(3 gamma)

(rho = c = p_ambient = 1), RK4 in N_SUB sub-steps per acoustic step DT. The
acoustics need s(t) = d/dt sum_k w_k (R_k/R0_k)^3 back, the source in
p_t = -(u_x + beta s). The surrogate is an LSTM cell per grid cell: input
p', state (h, c) carried by the solver, output s. Exported with the state
as graph inputs and outputs; the generator concatenates them in x and y.

Teacher-forced on random pressure sequences (pulses and weak tones), the
state flowing through whole sequences, as it is used. The head is trained
on s / S_SCALE and the factor folded into its weights before export.
"""
import numpy as np
import torch
import torch.nn as nn

NBIN = 8
R0 = np.geomspace(0.5, 2.0, NBIN)               # bin radii
W = np.ones(NBIN) / NBIN                         # bin weights (volume fraction shares)
GAMMA, MU = 1.4, 0.05
DT, N_SUB = 0.05, 10
HIDDEN = 32
S_SCALE = 10.0               # s has rms ~0.08
N_SEQ, T_SEQ = 256, 400
EPOCHS = 1500
SEED = 5


def rp_rhs(R, V, p, r0):
    """Rayleigh-Plesset right-hand side for radius R, velocity V, forcing p, rest radius r0."""
    pg = (r0 / R) ** (3 * GAMMA)
    return V, (pg - 1.0 - p - 4.0 * MU * V / R - 1.5 * V * V) / R


def rk4_step(R, V, p, r0, h):
    k1r, k1v = rp_rhs(R, V, p, r0)
    k2r, k2v = rp_rhs(R + 0.5 * h * k1r, V + 0.5 * h * k1v, p, r0)
    k3r, k3v = rp_rhs(R + 0.5 * h * k2r, V + 0.5 * h * k2v, p, r0)
    k4r, k4v = rp_rhs(R + h * k3r, V + h * k3v, p, r0)
    return (R + h / 6 * (k1r + 2 * k2r + 2 * k3r + k4r),
            V + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))


def population_source(R, V, r0):
    """s = d/dt sum_k w_k (R_k/R0_k)^3 = sum_k w_k 3 R_k^2 V_k / R0_k^3."""
    return (W * 3.0 * R * R * V / r0 ** 3).sum(axis=-1)


def pressure_signals(rng, n, t):
    """A few pulses of random width and sign plus a weak tone, |p'| below about 0.5."""
    p = np.zeros((n, len(t)))
    for i in range(n):
        for _ in range(rng.integers(1, 4)):
            t0, sig, a = rng.uniform(2, 18), rng.uniform(0.3, 1.5), rng.uniform(-0.35, 0.35)
            p[i] += a * np.exp(-0.5 * ((t - t0) / sig) ** 2)
        p[i] += rng.uniform(0, 0.08) * np.sin(rng.uniform(0.3, 1.5) * t + rng.uniform(0, 6.28))
    return p


def make_sequences(rng):
    """(N_SEQ, T_SEQ) pressures and the exact population source at every acoustic step."""
    t = np.arange(T_SEQ) * DT
    p = pressure_signals(rng, N_SEQ, t)
    R = np.tile(R0, (N_SEQ, 1)); V = np.zeros((N_SEQ, NBIN)); r0 = np.tile(R0, (N_SEQ, 1))
    s = np.zeros((N_SEQ, T_SEQ))
    for n in range(T_SEQ):
        for _ in range(N_SUB):
            R, V = rk4_step(R, V, p[:, n:n + 1], r0, DT / N_SUB)
        s[:, n] = population_source(R, V, r0)
    return p, s


class Bubbles(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, p, h, c):
        out, (hn, cn) = self.lstm(p, (h, c))
        # squeeze, not out[0]: indexing exports an int64 Gather
        return self.head(torch.squeeze(out, 0)), hn, cn


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    p, s = make_sequences(rng)
    print(f"source rms {np.sqrt((s ** 2).mean()):.3f}, max |p'| {np.abs(p).max():.2f}")
    P = torch.tensor(p.T[:, :, None], dtype=torch.float32)      # (T, N, 1)
    S = torch.tensor(s.T[:, :, None] * S_SCALE, dtype=torch.float32)
    model = Bubbles()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    for ep in range(EPOCHS):
        out, _ = model.lstm(P)
        pred = model.head(out)
        loss = ((pred - S) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if ep % 250 == 0:
            print(f"epoch {ep:4d}  relative rms error {np.sqrt(loss.item()) / np.sqrt((S ** 2).mean().item()):.3f}")
    with torch.no_grad():
        model.head.weight /= S_SCALE
        model.head.bias /= S_SCALE
    model.eval()
    with torch.no_grad():                                        # held-out, state fed back step by step
        p2, s2 = make_sequences(np.random.default_rng(SEED + 1))
        h = torch.zeros(1, N_SEQ, HIDDEN); c = torch.zeros(1, N_SEQ, HIDDEN)
        pred = np.zeros_like(s2)
        for n in range(T_SEQ):
            y, h, c = model(torch.tensor(p2[:, n], dtype=torch.float32)[None, :, None], h, c)
            pred[:, n] = y[:, 0].numpy()
        err = np.sqrt(((pred - s2) ** 2).sum() / (s2 ** 2).sum())
        print(f"held-out sequences, state fed back step by step: relative L2 error {err:.3e}")
    torch.onnx.export(model, (torch.zeros(1, 1, 1), torch.zeros(1, 1, HIDDEN), torch.zeros(1, 1, HIDDEN)),
                      "bubbles.onnx", input_names=["p", "h", "c"], output_names=["s", "h_next", "c_next"],
                      opset_version=13, dynamo=False)
    print("wrote bubbles.onnx")


if __name__ == "__main__":
    main()
