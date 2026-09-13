FC=gfortran -I../fLibrary/objFiles/ -J../fLibrary/objFiles/
DIR = ../fLibrary
MAIN = ../
FFLAGS=-O2
FFLAGS=-O3
SRC=userTesting.fpp
SRCBASE=$(DIR)/activation_funcs.f90 $(DIR)/derived_types.f90 $(DIR)/layers.f90 $(DIR)/reader.f90
OBJ2=modelCreator.o ${SRC:.fpp=.o}
COMP=${SRCBASE:.f90=.o}

output: $(COMP) $(OBJ2)
	$(FC) $(FFLAGS) -o $@ $(COMP) $(OBJ2)

.PRECIOUS : %.f90
%.f90: %.fpp variables.fpp
	fypp -I. $< $*.f90

modelCreator.f90: $(DIR)/modelCreator.fpp variables.fpp
	fypp -I. $(DIR)/modelCreator.fpp modelCreator.f90

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
	python3 $(DIR)/modelParserONNX.py -f $(MAIN)goldenFiles/$(case)/$(case).onnx -i $(MAIN)goldenFiles/$(case)/$(case)_inferred.onnx


graphs: output
	./output 2> outputCase.txt



compile: $(COMP)

unit: $(COMP) unit_tests.o
	$(FC) $(FFLAGS) -o unit_tests $(COMP) unit_tests.o
	./unit_tests
	gcc -c c_default_paths.c -o c_default_paths.o
	$(FC) $(FFLAGS) -o c_default_paths c_default_paths.o $(COMP)
	rm -rf c_default_paths_run && mkdir c_default_paths_run
	cd c_default_paths_run && ! ../c_default_paths 2> err.txt && grep -q "cannot open model file 'onnxModel.txt'" err.txt
	rm -rf c_default_paths_run

clean:
	rm *.o *.mod output
