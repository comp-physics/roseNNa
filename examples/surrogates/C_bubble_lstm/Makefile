# make [TOOLCHAIN=amd|nvidia|gnu] [NB=64]   -- train (if needed), generate, build, run both
include ../common.mk
MODEL := bubbles
NB    ?= 64

.PHONY: all run c fortran clean distclean
all: run
run: bubbles_c bubbles_f
	OMP_TARGET_OFFLOAD=$(if $(filter gnu,$(TOOLCHAIN)),DEFAULT,MANDATORY) ./bubbles_c
	OMP_TARGET_OFFLOAD=$(if $(filter gnu,$(TOOLCHAIN)),DEFAULT,MANDATORY) ./bubbles_f
c: bubbles_c
fortran: bubbles_f

# closure.onnx is checked in (train.py takes a couple of minutes); `make train` rebuilds it.
train:
	$(PYTHON) train.py

gen/$(MODEL).h gen/$(MODEL)_model.F90: $(MODEL).onnx
	$(ROSENNA) generate $< --lang both --precision double --out gen

bubbles_c: bubbles.c gen/$(MODEL).h
	$(CC) $(CFLAGS) $(OFFLOAD) -DNB=$(NB) -Igen $< -lm -o $@

gen/$(MODEL)_model.o: gen/$(MODEL)_model.F90
	$(FC) $(FFLAGS) $(OFFLOAD) $(FMOD) gen -c $< -o $@
bubbles_f: bubbles.F90 gen/$(MODEL)_model.o
	$(FC) $(FFLAGS) $(OFFLOAD) -DNB=$(NB) -Igen $< gen/$(MODEL)_model.o -o $@

clean:
	rm -rf gen bubbles_c bubbles_f bubbles_c_host *.mod
distclean: clean
	rm -f $(MODEL).onnx
