# cns_closure is C only, so it borrows the surrogates' toolchain block rather
# than restating it: TOOLCHAIN=gnu|nvidia|amd, the offload and archive flags,
# and OMP_TARGET_OFFLOAD=MANDATORY for the GPU toolchains all live there, and
# the nvidia link flags in particular (-cuda -c++libs, not -lcudart) were
# arrived at the hard way.
MODEL := closure
PROG  := cns
LANGS   := c
GENLANG := c

# A 3D box, so the default is small: NX=32 is 32^3 interior cells.
NX      ?= 32
NSTEPS  ?= 20
DEFS    := -DNSTEPS=$(NSTEPS)

# BATCHED=1 builds the gather + one-call-per-field path instead of the
# per-point one, which links lib$(MODEL).a.
ifeq ($(BATCHED),1)
DEFS    += -DBATCHED
ARCHIVE := 1
endif

include ../surrogates/common.mk
