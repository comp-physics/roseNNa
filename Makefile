FC=gfortran
FFLAGS=-O2
FFLAGS=-O3
SRC=modelCreator.fpp userTesting.fpp
SRCBASE=activation_funcs.f90 derived_types.f90 layers.f90 reader.f90
OBJ2=${SRC:.fpp=.o}
COMP=${SRCBASE:.f90=.o}

output: $(COMP) $(OBJ2)
	$(FC) $(FFLAGS) -o $@ $(COMP) $(OBJ2)

.PRECIOUS : %.f90
%.f90: %.fpp variables.fpp
	fypp $< $*.f90

%.o: %.f90
	$(FC) $(FFLAGS) -o $@ -c $<

test: ex1 output
	./output 2> outputCase.txt

testing: ex1 output
	./output 2> outputCase.txt
	python3 -Wi goldenFiles/testChecker.py $(case)

capi: capi.c modelCreator.o
	gcc -c capi.c
	gfortran -o capi modelCreator.o capi.o
	./capi

ex1: modelParserONNX.py
	python3 goldenFiles/$(case)/$(case).py 
	python3 modelParserONNX.py -f goldenFiles/$(case)/$(case).onnx -w goldenFiles/$(case)/$(case)_weights.onnx -i goldenFiles/$(case)/$(case)_inferred.onnx 1>/dev/null


graphs: output
	./output 2> outputCase.txt



compile: $(COMP)

clean:
	rm *.o *.mod output
