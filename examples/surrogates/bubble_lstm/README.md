# Bubbly acoustics with a recurrent per-cell surrogate

Wave equation coupled to stiff per-cell ODEs (1-D acoustics through a
region of dispersed bubbles). The network replaces the per-cell bubble
population, 8 Rayleigh-Plesset bins integrated by RK4 with 10 sub-steps
per acoustic step, with one LSTM step per cell that maps the cell's
pressure to the population's volume-fraction rate, the source the
acoustics need. The stateful pattern: each cell's `(h, c)` stays on the
device; the model has three inputs and three outputs, concatenated in `x`
and `y`, and the solver copies `y`'s state slices back into its arrays
for the next step.

```
$ rosenna info bubbles.onnx
x: p[0:1] h[1:33] c[33:65]
y: s[0:1] h_next[1:33] c_next[33:65]
```

```
make TOOLCHAIN=amd | nvidia | gnu      # gnu: host only, use NB=4
```

An ensemble of `NB` lines, each a right-going pulse of hashed amplitude
and width into a bubbly region (`x ∈ [10, 20]`, coupling `β = 0.1`).
Characteristic-form upwind at CFL = 1, so the transport is exact and the
bubbles are the only physics. Exits 0 if the surrogate's pressure field
is within 10% of the reference after 400 steps. MI210:

```
64 lines x 512 cells, 400 steps; 8 bins x 10 RK4 sub-steps per cell-step in the reference:
  reference population    266.7 ms  (20.3 ns per cell-step)
  LSTM surrogate          255.9 ms  (19.5 ns per cell-step)
  relative L2 error of the surrogate's pressure field: 2.919e-02
OK
```

A 32-unit LSTM costs about what 8 bins × 10 RK4 sub-steps cost on the
GPU. Its cost is fixed; the population's grows with bins, sub-steps and
per-bin physics.

## Model

`train.py`: `nn.LSTM(1, 32)` and a linear head, teacher-forced on 256
random pressure sequences with the exact population response. `s` has
rms 0.08, so the head is trained on `10 s` and the factor folded into its
weights before export. The head reads the LSTM output with
`torch.squeeze(out, 0)`; `out[0]` exports an int64 `Gather`, which the
generator refuses.

## Wiring

```c
x[0] = p;
for (int i = 0; i < HID; ++i) { x[1 + i] = H[c * HID + i]; x[1 + HID + i] = C[c * HID + i]; }
bubbles_infer(x, y);
src[c] = y[0];
for (int i = 0; i < HID; ++i) { H[c * HID + i] = y[1 + i]; C[c * HID + i] = y[1 + HID + i]; }
```

`H` and `C` are mapped once with the fields. In `bubbles.F90` the state
arrays are `hs`/`cs` (Fortran is case-insensitive; `C` collided with the
cell index) and the slice copies are element loops: a section assignment
inside a target region needs a runtime call the device lacks under
`amdflang`.

## Note

The explicit bubble-acoustic coupling is stable for `β` up to about 0.2
at this `dt`; at 0.4 the reference itself diverges.
