# The module is built host-only: a whole-field model's activations (660 KB) exceed
# a device thread's stack, and the module's infer_batch would instantiate infer there.
MODEL          := poisson_guess
PROG           := poisson
MODULE_OFFLOAD :=
include ../common.mk
