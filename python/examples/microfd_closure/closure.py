"""Export closure.onnx: a per-cell turbulence-closure MLP for the microfd worked example.

9 inputs -- the velocity-gradient tensor du_i/dx_j at one cell, flattened
row-major (du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz, dw/dx, dw/dy, dw/dz) --
one hidden layer of 16 units with Tanh, and one output (a turbulent-viscosity
correction, added to the molecular mu at a face in patch.md). No output
activation: the closure is a signed correction, not a probability or a
strictly positive quantity, and the network is free to clip or scale it
downstream (patch.md's `muf` line does exactly that: mu + max(...)).

Deterministic: torch.manual_seed pins the initialization so re-running this
script reproduces the same closure.onnx byte for byte (module weight order
and Kaiming/uniform default init are themselves deterministic given the
seed).

This is documentation, not a validated closure model: see README.md in this
directory for what "validated" means here, and patch.md for the exact edits
to microfd.c. Nothing in this repository compiles microfd.c.
"""
from pathlib import Path

import torch
import torch.nn as nn

N_IN, N_HIDDEN, N_OUT = 9, 16, 1


class Closure(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(N_IN, N_HIDDEN)
        self.act = nn.Tanh()
        self.output = nn.Linear(N_HIDDEN, N_OUT)

    def forward(self, x):
        return self.output(self.act(self.hidden(x)))


def main():
    torch.manual_seed(0)
    model = Closure().eval()
    example = torch.zeros(1, N_IN)
    out_path = Path(__file__).parent / "closure.onnx"
    torch.onnx.export(
        model, example, str(out_path),
        export_params=True, dynamo=False,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(out_path)


if __name__ == "__main__":
    main()
