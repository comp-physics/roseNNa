# Shared by every example Makefile: pick the toolchain once.
#
#   make TOOLCHAIN=amd      amdclang / amdflang -fopenmp --offload-arch=gfx90a, hip archive
#   make TOOLCHAIN=nvidia   nvc / nvfortran -mp=gpu -gpu=cc80, cuda archive
#   make TOOLCHAIN=gnu      gcc / gfortran -fopenmp on the host (default: runs anywhere)
#
# ARCH overrides the GPU target (gfx90a, cc80, ...). ROSENNA is the generator
# CLI. The same OpenMP-target source builds under all three; only the flags
# and the batched-kernel archive change.
TOOLCHAIN ?= gnu
ROSENNA   ?= rosenna
PYTHON    ?= python3

ifeq ($(TOOLCHAIN),amd)
ARCH    ?= gfx90a
CC      := amdclang
FC      := amdflang
OFFLOAD := -fopenmp --offload-arch=$(ARCH)
BACKEND := hip
DEVCC   := hipcc
DEVFLAGS:= -O2 --offload-arch=$(ARCH)
FMOD    := -J
else ifeq ($(TOOLCHAIN),nvidia)
ARCH    ?= cc80
CC      := nvc
FC      := nvfortran
OFFLOAD := -mp=gpu -gpu=$(ARCH)
BACKEND := cuda
DEVCC   := nvcc
DEVFLAGS:= -O2 -arch=sm_$(patsubst cc%,%,$(ARCH))
FMOD    := -module
else
CC      := gcc
FC      := gfortran
OFFLOAD := -fopenmp
BACKEND := omp
DEVCC   :=
DEVFLAGS:=
FMOD    := -J
endif

CFLAGS ?= -O2
FFLAGS ?= -O2
