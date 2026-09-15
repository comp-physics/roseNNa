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
RTLIB    := -lcudart
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
SIZES  := -DNB=$(NB) -DNX=$(NX) $(DEFS)
LIB    := $(if $(ARCHIVE),gen/lib$(MODEL).a $(RTLIB),)
RUNENV := OMP_TARGET_OFFLOAD=$(if $(filter gnu,$(TOOLCHAIN)),DEFAULT,MANDATORY)

.PHONY: all run train clean distclean
all: run
run: $(PROG)_c $(PROG)_f
	$(RUNENV) ./$(PROG)_c
	$(RUNENV) ./$(PROG)_f

train:                                  # the .onnx is checked in; this rebuilds it
	$(PYTHON) train.py

gen/$(MODEL).h gen/$(MODEL)_model.F90 gen/$(MODEL).mk: $(MODEL).onnx
	$(ROSENNA) generate $< --lang both --precision double $(GENFLAGS) --out gen
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
	rm -rf gen $(PROG)_c $(PROG)_f $(MODEL).rwt *.mod
distclean: clean
	rm -f $(MODEL).onnx
