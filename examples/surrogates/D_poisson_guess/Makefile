# make [TOOLCHAIN=amd|nvidia|gnu]   -- generate, build, run both
include ../common.mk
MODEL := poisson_guess

.PHONY: all run c fortran train clean distclean
all: run
run: poisson_c poisson_f
	OMP_TARGET_OFFLOAD=$(if $(filter gnu,$(TOOLCHAIN)),DEFAULT,MANDATORY) ./poisson_c
	OMP_TARGET_OFFLOAD=$(if $(filter gnu,$(TOOLCHAIN)),DEFAULT,MANDATORY) ./poisson_f
c: poisson_c
fortran: poisson_f

# poisson_guess.onnx is checked in (train.py takes about a minute); `make train` rebuilds it.
train:
	$(PYTHON) train.py

gen/$(MODEL).h gen/$(MODEL)_model.F90: $(MODEL).onnx
	$(ROSENNA) generate $< --lang both --precision double --out gen

poisson_c: poisson.c gen/$(MODEL).h
	$(CC) $(CFLAGS) $(OFFLOAD) -Igen $< -lm -o $@

# The module is built WITHOUT offload flags: a whole-field model's infer
# holds the field's activations (660 KB) as locals, more than a device
# thread can hold, and the module's own infer_batch would instantiate it on
# the device even though this program never calls it. The guess runs on the
# host; only the program's own loops are offloaded.
gen/$(MODEL)_model.o: gen/$(MODEL)_model.F90
	$(FC) $(FFLAGS) $(FMOD) gen -c $< -o $@
poisson_f: poisson.F90 gen/$(MODEL)_model.o
	$(FC) $(FFLAGS) $(OFFLOAD) -Igen $< gen/$(MODEL)_model.o -o $@

clean:
	rm -rf gen poisson_c poisson_f *.mod
distclean: clean
	rm -f $(MODEL).onnx
