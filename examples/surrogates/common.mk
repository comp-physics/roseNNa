# Shared by the example Makefiles. Each one sets MODEL, PROG and GENFLAGS,
# then includes this file. TOOLCHAIN=amd|nvidia|gnu; ARCH overrides the
# GPU target; NB / NX shrink the problem for a host run. An example may add
# its own -D flags in DEFS.
TOOLCHAIN ?= gnu
ROSENNA   ?= rosenna
PYTHON    ?= python3
NB        ?= 64
NX        ?= 256
CFLAGS    ?= -O2
FFLAGS    ?= -O2

ifeq ($(TOOLCHAIN),amd)
ARCH     ?= gfx90a
CC       := amdclang
FC       := amdflang
OFFLOAD  := -fopenmp --offload-arch=$(ARCH)
BACKEND  := hip
DEVCC    := hipcc
DEVFLAGS := -O2 --offload-arch=$(ARCH)
RTLIB    := -L$(dir $(shell which hipcc))../lib -lamdhip64
FMOD     := -J
else ifeq ($(TOOLCHAIN),nvidia)
ARCH     ?= cc80
CC       := nvc
FC       := nvfortran
OFFLOAD  := -mp=gpu -gpu=$(ARCH)
BACKEND  := cuda
DEVCC    := nvcc
DEVFLAGS := -O2 -arch=sm_$(patsubst cc%,%,$(ARCH))
# -cuda -c++libs, not -lcudart: both are nvc/nvfortran's own flags, so neither
# hardcodes a path or a C++ runtime implementation.
#   -cuda      links the CUDA runtime. A bare -lcudart needs a -L the HPC SDK
#              does not put on the default search path (its libcudart lives
#              under cuda/lib64, not beside nvcc), so it fails with
#              "cannot find -lcudart" on a stock install.
#   -c++libs   nvcc compiles <model>_kernel.cu as C++, and a per-op kernel's
#              function-local static leaves __cxa_guard_acquire/_release
#              undefined when the C or Fortran driver links the archive.
RTLIB    := -cuda -c++libs
FMOD     := -module
else
CC       := gcc
FC       := gfortran
OFFLOAD  := -fopenmp
BACKEND  := omp
RTLIB    :=
FMOD     := -J
endif

# ARCHIVE=1: the program links lib$(MODEL).a (infer_batch for $(BACKEND)).
# MODULE_OFFLOAD: flags for the Fortran module; empty builds it host-only.
ARCHIVE        ?=
MODULE_OFFLOAD ?= $(OFFLOAD)
# Which drivers to build and run, and what to ask `generate` for. The four
# surrogates are both languages; an example may be one (cns_closure is C).
LANGS          ?= c f
GENLANG        ?= both
SIZES  := -DNB=$(NB) -DNX=$(NX) $(DEFS)
LIB    := $(if $(ARCHIVE),gen/lib$(MODEL).a $(RTLIB),)
RUNENV := OMP_TARGET_OFFLOAD=$(if $(filter gnu,$(TOOLCHAIN)),DEFAULT,MANDATORY)

.PHONY: all run train clean distclean
all: run
run: $(foreach l,$(LANGS),$(PROG)_$(l))
	# && , not `;`: make checks the exit status of the recipe line, so a
	# semicolon-separated list would hide every failure but the last one.
	$(foreach l,$(LANGS),$(RUNENV) ./$(PROG)_$(l) &&) true

TRAINER ?= train.py
train:                                  # the .onnx is checked in; this rebuilds it
	$(PYTHON) $(TRAINER)

gen/$(MODEL).h gen/$(MODEL)_model.F90 gen/$(MODEL).mk: $(MODEL).onnx
	$(ROSENNA) generate $< --lang $(GENLANG) --precision double $(GENFLAGS) --out gen
	@if [ -f gen/$(MODEL).rwt ]; then cp gen/$(MODEL).rwt .; fi

gen/lib$(MODEL).a: gen/$(MODEL).mk
	$(MAKE) -C gen -f $(MODEL).mk ROSENNA_BACKEND=$(BACKEND) CC=$(CC) "CFLAGS=$(CFLAGS)" \
	    "ROSENNA_OFFLOAD_FLAGS=$(OFFLOAD)" DEVCC=$(DEVCC) "DEVFLAGS=$(DEVFLAGS)"

$(PROG)_c: $(PROG).c gen/$(MODEL).h $(if $(ARCHIVE),gen/lib$(MODEL).a)
	$(CC) $(CFLAGS) $(OFFLOAD) $(SIZES) -Igen $< $(LIB) -lm -o $@

gen/$(MODEL)_model.o: gen/$(MODEL)_model.F90
	$(FC) $(FFLAGS) $(MODULE_OFFLOAD) $(FMOD) gen -c $< -o $@
$(PROG)_f: $(PROG).F90 gen/$(MODEL)_model.o $(if $(ARCHIVE),gen/lib$(MODEL).a)
	$(FC) $(FFLAGS) $(OFFLOAD) $(SIZES) -Igen $< gen/$(MODEL)_model.o $(LIB) -o $@

clean:
	rm -rf gen $(foreach l,$(LANGS),$(PROG)_$(l)) $(MODEL).rwt *.mod
distclean: clean
	rm -f $(MODEL).onnx
