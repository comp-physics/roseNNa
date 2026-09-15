MODEL   := poisson_guess
PROG    := poisson
ARCHIVE := 1
NSTEPS  ?= 20
DEFS    := -DNSTEPS=$(NSTEPS)
# The module is built host-only: its own infer_batch would instantiate the whole-field
# infer on the device (660 KB of locals). The program uses the archive's infer_one.
MODULE_OFFLOAD :=
include ../common.mk
