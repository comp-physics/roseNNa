# C. Bubbly acoustics with a recurrent per-cell surrogate

**PDE class**: wave equation coupled to stiff per-cell ODEs (1-D linear
acoustics through a region of dispersed bubbles).
**Where the NN sits**: it replaces the per-cell bubble *population* -- 8
Rayleigh-Plesset oscillators per cell, RK4 with 10 sub-steps per acoustic
step -- with one LSTM step per cell that turns the cell's pressure into the
population's volume-fraction rate, the source the acoustics need.
**Code structure it shows**: a **stateful** surrogate. Every cell owns an
LSTM state `(h, c)` that lives on the device for the whole run. The model
has three graph inputs and three graph outputs; the generator lays them
out concatenated in `x` and `y`, and the solver copies `y`'s state slices
straight back into its state arrays for the next step, on the device.

```
$ rosenna info bubbles.onnx
...
x: p[0:1] h[1:33] c[33:65]
y: s[0:1] h_next[1:33] c_next[33:65]
supported
```

```
make TOOLCHAIN=amd | nvidia | gnu        (gnu: NB=4)
```

## What it does

An ensemble of `NB` lines, each carrying a right-going pressure pulse of
hashed amplitude and width into a bubbly region (`x ∈ [10, 20]`, coupling
`β = 0.1`). Characteristic-form upwind at CFL = 1 (exact transport), so
whatever the bubbles do to the pulse is the whole story. The program runs
the reference population and the surrogate for 400 steps and prints the
relative L2 error of the surrogate's pressure field. On an MI210:

```
64 lines x 512 cells, 400 steps through a bubbly region (8 bins, 10 RK4 sub-steps per step):
  reference population     266.7 ms  (20.3 ns per cell-step)
  LSTM surrogate           255.9 ms  (19.5 ns per cell-step)
  relative L2 error of the surrogate's pressure field: 2.919e-02
OK
```

An LSTM of 32 units costs about what 8 bins × 10 RK4 sub-steps cost on the
GPU. Its cost is fixed; the population's grows with bins, sub-steps and
the physics in each bin (mass transfer, chemistry, a Keller-Miksis
correction), which is where a learned population model pays.

## The model

`train.py`: `nn.LSTM(1, 32)` plus a linear head, teacher-forced on 256
random pressure signals (pulses and weak tones) with the exact population
response, the state flowing through whole 400-step sequences -- exactly how
it is used. Two details matter. The target `s` has rms 0.08, so the head
is trained on `10 s` and the factor folded into its weights before export.
And the head reads the LSTM output with `torch.squeeze(out, 0)`, not
`out[0]`: indexing exports an int64 `Gather`, which the generator refuses.

Exported with `torch.onnx.export(model, (p, h0, c0), ...)`, the state
becomes graph inputs `h`, `c` and outputs `h_next`, `c_next` next to `s`.

## The wiring

```c
#pragma omp target teams distribute parallel for
for (int c = 0; c < NCELL; ++c) {
    const double p = 0.5 * (wp[c] + wm[c]);
    double x[NIN], y[NOUT];
    x[0] = p;
    for (int i = 0; i < HID; ++i) { x[1 + i] = H[c * HID + i]; x[1 + HID + i] = C[c * HID + i]; }
    bubbles_infer(x, y);                     /* one LSTM step */
    src[c] = y[0];
    for (int i = 0; i < HID; ++i) { H[c * HID + i] = y[1 + i]; C[c * HID + i] = y[1 + HID + i]; }
}
```

`H` and `C` are mapped once with the fields; the offsets are the ones
`rosenna info` printed. Two Fortran-specific notes in `bubbles.F90`: the
state arrays are `hs`/`cs` because Fortran is case-insensitive and `C`
collided with the cell index; and the slice copies are element loops, not
section assignments -- a section assignment inside a target region lowers
to a Fortran runtime call (`_FortranAAssign`) the device does not have
under `amdflang`.

## A physics note

The explicit bubble-acoustic coupling is only stable for `β` up to ~0.2 at
this `dt`; at 0.4 the reference itself blows up. The surrogate is not the
fragile part here.
