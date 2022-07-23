FC=gfortran
FFLAGS=-O3
SRC=modelCreator.fpp userTesting.fpp
SRCBASE=activation_funcs.f90 derived_types.f90 layers.f90 readTester.f90
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
	./output

testing: ex1 output
	./output 2> outputCase.txt
	python3 -Wi goldenFiles/testChecker.py $(case)

ex1: modelParserONNX.py
	python3 goldenFiles/$(case)/$(case).py
	python3 modelParserONNX.py $(case)

compile: $(COMP)

clean:
	rm *.o *.mod output
