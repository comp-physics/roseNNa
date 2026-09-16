# cns_closure is C only, so it borrows the surrogates' toolchain block rather
# than restating it: TOOLCHAIN=gnu|nvidia|amd, the offload and archive flags,
# and OMP_TARGET_OFFLOAD=MANDATORY for the GPU toolchains all live there, and
# the nvidia link flags in particular (-cuda -c++libs, not -lcudart) were
# arrived at the hard way.
MODEL := closure
PROG  := cns
# `make train` re-exports closure.onnx, which is checked in like the
# surrogates' models. It is an exporter, not a trainer -- see closure.py.
TRAINER := closure.py
LANGS   := c
GENLANG := c

# A 3D box, so the default is small: NX=32 is 32^3 interior cells.
NX      ?= 32
NSTEPS  ?= 20
DEFS    := -DNSTEPS=$(NSTEPS)

# BATCHED=1 builds the gather + one-call-per-field path instead of the
# per-point one, which links lib$(MODEL).a.
# NO_CLOSURE=1 drops the network entirely: nut stays zero, mu_eff = mu, and
# the program is a plain compressible Navier-Stokes solver. It is the baseline
# the closure's cost is measured against.
ifeq ($(NO_CLOSURE),1)
DEFS    += -DNO_CLOSURE
endif

ifeq ($(BATCHED),1)
DEFS    += -DBATCHED
ARCHIVE := 1
endif

include ../surrogates/common.mk
