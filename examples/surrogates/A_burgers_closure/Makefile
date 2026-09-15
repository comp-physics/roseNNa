# make [TOOLCHAIN=amd|nvidia|gnu] [NB=64]   -- train (if needed), generate, build, run both
include ../common.mk
MODEL := closure
NB    ?= 64

.PHONY: all run c fortran clean distclean
all: run
run: burgers_c burgers_f
	OMP_TARGET_OFFLOAD=$(if $(filter gnu,$(TOOLCHAIN)),DEFAULT,MANDATORY) ./burgers_c
	OMP_TARGET_OFFLOAD=$(if $(filter gnu,$(TOOLCHAIN)),DEFAULT,MANDATORY) ./burgers_f
c: burgers_c
fortran: burgers_f

# closure.onnx is checked in (train.py takes a few minutes); `make train` rebuilds it.
train:
	$(PYTHON) train.py

gen/$(MODEL).h gen/$(MODEL)_model.F90: $(MODEL).onnx
	$(ROSENNA) generate $< --lang both --precision double --out gen

burgers_c: burgers.c gen/$(MODEL).h
	$(CC) $(CFLAGS) $(OFFLOAD) -DNB=$(NB) -Igen $< -lm -o $@

gen/$(MODEL)_model.o: gen/$(MODEL)_model.F90
	$(FC) $(FFLAGS) $(OFFLOAD) $(FMOD) gen -c $< -o $@
burgers_f: burgers.F90 gen/$(MODEL)_model.o
	$(FC) $(FFLAGS) $(OFFLOAD) -DNB=$(NB) -Igen $< gen/$(MODEL)_model.o -o $@

clean:
	rm -rf gen burgers_c burgers_f burgers_c_host *.mod
distclean: clean
	rm -f $(MODEL).onnx
