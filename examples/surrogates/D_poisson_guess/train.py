"""Train a conv net that guesses the solution of a periodic Poisson problem, and export it.

The PDE is lap(phi) = f on an N x N periodic grid (dx = 1, mean-zero f, phi
fixed to mean zero). The solver in poisson.c / poisson.F90 is a Jacobi
iteration; the surrogate does not replace it, it *starts* it: phi0 = NN(f),
and Jacobi runs from there to a residual tolerance. The check is the
iteration count against two baselines a time-stepping code would use --
a zero guess and the previous step's solution -- with the same tolerance.

The model is three 5x5 convolutions, 1 -> 8 -> 8 -> 1 channels, tanh between,
with NO padding: a 13-cell halo is what three 5x5 layers consume, and the
SOLVER supplies it by wrapping f periodically into a (N + 12) x (N + 12)
array before the call. ONNX Conv only zero-pads, so a periodic halo has to
come from the caller either way, and putting it there keeps the exported
graph plain. The whole field is one input (NCHW, 1 x 1 x 76 x 76) and one
output (1 x 1 x 64 x 64): a single infer call per step, not one per cell.

Training data: random mean-zero f from a few Fourier modes. The loss is
the RESIDUAL of the guess, |lap(NN(f)) - f|^2, not its distance to the
exact phi: Jacobi stops on the residual, and a guess fitted to phi in L2
carries high-mode error that the Laplacian amplifies by k^2 -- the first
version of this model, trained that way, made Jacobi take twice as many
iterations as a zero guess. Trained on the residual, the same 13-cell net
leaves 8% of the zero guess's residual (and, as it happens, a better phi).

Writes poisson_guess.onnx.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 64
HALO = 6                     # three 5x5 valid convolutions eat 2 cells each side, each
N_TRAIN = 512
PHI_SCALE = 10.0             # phi has rms ~14 for unit-rms f (low modes are amplified by 1/k^2);
                             # the net is trained on phi / PHI_SCALE and the factor folded into
                             # its last convolution before export
ITERS = 3000
SEED = 13


def random_rhs(rng, n):
    """Mean-zero right-hand sides: 6 random Fourier modes with wavenumbers up to 8, unit rms."""
    x = np.arange(N)
    kx, ky = np.meshgrid(x, x, indexing="ij")
    f = np.zeros((n, N, N))
    for i in range(n):
        for _ in range(6):
            p, q = rng.integers(-8, 9, 2)
            if p == 0 and q == 0:
                continue
            f[i] += rng.normal() * np.cos(2 * np.pi * (p * kx + q * ky) / N + rng.uniform(0, 2 * np.pi))
    f -= f.mean(axis=(1, 2), keepdims=True)
    return f / np.sqrt((f ** 2).mean(axis=(1, 2), keepdims=True))


def exact_solution(f):
    """lap(phi) = f, periodic, dx = 1, mean(phi) = 0, by the FFT of the 5-point Laplacian."""
    k = np.fft.fftfreq(N) * N
    kx, ky = np.meshgrid(k, k, indexing="ij")
    eig = 2 * np.cos(2 * np.pi * kx / N) + 2 * np.cos(2 * np.pi * ky / N) - 4    # 5-point stencil symbol
    eig[0, 0] = 1.0
    phi_hat = np.fft.fft2(f) / eig
    phi_hat[:, 0, 0] = 0.0
    return np.real(np.fft.ifft2(phi_hat))


def wrap(f, halo=HALO):
    """Periodic halo, the way the solvers build the model's input."""
    return F.pad(f, (halo, halo, halo, halo), mode="circular")


def laplacian(phi):
    """The solvers' 5-point periodic Laplacian, dx = 1."""
    p = F.pad(phi, (1, 1, 1, 1), mode="circular")
    return p[:, :, :-2, 1:-1] + p[:, :, 2:, 1:-1] + p[:, :, 1:-1, :-2] + p[:, :, 1:-1, 2:] - 4 * phi


class Guess(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 8, 5), nn.Tanh(), nn.Conv2d(8, 8, 5), nn.Tanh(),
                                 nn.Conv2d(8, 1, 5))

    def forward(self, f_padded):
        return self.net(f_padded)


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    f = random_rhs(rng, N_TRAIN)
    phi = exact_solution(f)
    Ft = torch.tensor(f[:, None], dtype=torch.float32)
    Fp = wrap(Ft)                                                       # (n, 1, N+12, N+12)
    model = Guess()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
    for it in range(ITERS):
        idx = torch.randint(0, N_TRAIN, (32,))
        loss = ((laplacian(model(Fp[idx]) * PHI_SCALE) - Ft[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0:
            print(f"iter {it:4d}  residual mse {loss.item():.3e}  (|f|^2 mean is 1)")
    with torch.no_grad():
        model.net[-1].weight *= PHI_SCALE
        model.net[-1].bias *= PHI_SCALE
    model.eval()
    with torch.no_grad():
        f2 = random_rhs(np.random.default_rng(SEED + 1), 16)
        phi2 = exact_solution(f2)
        Ft2 = torch.tensor(f2[:, None], dtype=torch.float32)
        guess = model(wrap(Ft2))
        res = torch.sqrt(((laplacian(guess) - Ft2) ** 2).sum() / (Ft2 ** 2).sum()).item()
        rel = np.sqrt(((guess.numpy()[:, 0] - phi2) ** 2).sum() / (phi2 ** 2).sum())
        print(f"held-out: residual of the guess {res:.3f} of the zero guess's; phi error {rel:.3f}")
    torch.onnx.export(model, torch.zeros(1, 1, N + 2 * HALO, N + 2 * HALO), "poisson_guess.onnx",
                      input_names=["f_padded"], output_names=["phi_guess"], opset_version=13, dynamo=False)
    print("wrote poisson_guess.onnx")


if __name__ == "__main__":
    main()
