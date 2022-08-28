FC=gfortran
<<<<<<< HEAD
FFLAGS=-O2
=======
FFLAGS=-O3
>>>>>>> 932293133341125e44857a018a79d106ec53632e
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
<<<<<<< HEAD
	./output 2> outputCase.txt
=======
	./output 
>>>>>>> 932293133341125e44857a018a79d106ec53632e

testing: ex1 output
	./output 2> outputCase.txt
	python3 -Wi goldenFiles/testChecker.py $(case)

ex1: modelParserONNX.py
	python3 goldenFiles/$(case)/$(case).py
<<<<<<< HEAD
	python3 modelParserONNX.py -f $(case)
=======
	python3 modelParserONNX.py $(case)
>>>>>>> 932293133341125e44857a018a79d106ec53632e

compile: $(COMP)

clean:
	rm *.o *.mod output
