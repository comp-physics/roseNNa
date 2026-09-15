# make [TOOLCHAIN=amd|nvidia|gnu] [NX=256]   -- generate, build the archive, build, run both
include ../common.mk
MODEL := stepper
NX    ?= 256
# The archive holds infer_batch for the toolchain's native backend (hip, cuda)
# or the OpenMP fallback (gnu); a host that is not linked by hipcc/nvcc names
# that backend's runtime itself.
ifeq ($(BACKEND),hip)
RTLIB := -L$(dir $(shell which hipcc))../lib -lamdhip64
else ifeq ($(BACKEND),cuda)
RTLIB := -lcudart
else
RTLIB :=
endif

.PHONY: all run c fortran train clean distclean
all: run
run: react_c react_f
	OMP_TARGET_OFFLOAD=$(if $(filter gnu,$(TOOLCHAIN)),DEFAULT,MANDATORY) ./react_c
	OMP_TARGET_OFFLOAD=$(if $(filter gnu,$(TOOLCHAIN)),DEFAULT,MANDATORY) ./react_f
c: react_c
fortran: react_f

# stepper.onnx is checked in (train.py takes a minute or two); `make train` rebuilds it.
train:
	$(PYTHON) train.py

# --no-embed: the weights go to stepper.rwt and stepper_init uploads them, once.
gen/$(MODEL).h gen/$(MODEL)_model.F90 gen/$(MODEL).rwt gen/$(MODEL).mk: $(MODEL).onnx
	$(ROSENNA) generate $< --lang both --precision double --no-embed --out gen
$(MODEL).rwt: gen/$(MODEL).rwt
	cp $< $@

gen/lib$(MODEL).a: gen/$(MODEL).mk gen/$(MODEL).h
	$(MAKE) -C gen -f $(MODEL).mk ROSENNA_BACKEND=$(BACKEND) CC=$(CC) "CFLAGS=$(CFLAGS)" \
	    "ROSENNA_OFFLOAD_FLAGS=$(OFFLOAD)" DEVCC=$(DEVCC) "DEVFLAGS=$(DEVFLAGS)"

react_c: react.c gen/$(MODEL).h gen/lib$(MODEL).a $(MODEL).rwt
	$(CC) $(CFLAGS) $(OFFLOAD) -DNX=$(NX) -Igen $< gen/lib$(MODEL).a $(RTLIB) -lm -o $@

gen/$(MODEL)_model.o: gen/$(MODEL)_model.F90
	$(FC) $(FFLAGS) $(OFFLOAD) $(FMOD) gen -c $< -o $@
react_f: react.F90 gen/$(MODEL)_model.o gen/lib$(MODEL).a $(MODEL).rwt
	$(FC) $(FFLAGS) $(OFFLOAD) -DNX=$(NX) -Igen $< gen/$(MODEL)_model.o gen/lib$(MODEL).a $(RTLIB) -o $@

clean:
	rm -rf gen gen_hip react_c react_f react_c_hip $(MODEL).rwt *.mod
distclean: clean
	rm -f $(MODEL).onnx
