FC=gfortran -I../fLibrary/objFiles/ -J../fLibrary/objFiles/
DIR = ../fLibrary
MAIN = ../
FFLAGS=-O2
FFLAGS=-O3
SRC=modelCreator.fpp userTesting.fpp
SRCBASE=$(DIR)/activation_funcs.f90 $(DIR)/derived_types.f90 $(DIR)/layers.f90 $(DIR)/reader.f90
OBJ2=${SRC:.fpp=.o}
COMP=${SRCBASE:.f90=.o}

output: $(COMP) $(OBJ2)
	$(FC) $(FFLAGS) -o $@ $(COMP) $(OBJ2)

.PRECIOUS : %.f90
%.f90: %.fpp variables.fpp
	fypp $< $*.f90

%.o: %.f90
	$(FC) $(FFLAGS) -o $@ -c $<

$(DIR)/%.o: $(DIR)/%.f90
	$(FC) $(FFLAGS) -o $@ -c $<

test: ex1 user output
	./output 2> outputCase.txt

user: $(OBJ2) variables.fpp

testing: ex1 compile user output
	./output 2> outputCase.txt
	python3 -Wi testChecker.py $(case)

capi: capi.c modelCreator.o
	gcc -c capi.c
	gfortran -o capi modelCreator.o capi.o
	./capi

ex1: $(DIR)/modelParserONNX.py
	python3 $(MAIN)goldenFiles/$(case)/$(case).py
	python3 $(DIR)/modelParserONNX.py -f $(MAIN)goldenFiles/$(case)/$(case).onnx -w $(MAIN)goldenFiles/$(case)/$(case)_weights.onnx -i $(MAIN)goldenFiles/$(case)/$(case)_inferred.onnx


graphs: output
	./output 2> outputCase.txt



compile: $(COMP)

clean:
	rm *.o *.mod output
