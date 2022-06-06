FC=gfortran
FFLAGS=-O3 -Wall -Wextra -fcheck=all -fbacktrace
SRC=activation_funcs.f90 derived_types.f90 layers.f90 readTester.f90 linearV3copy.fpp userTesting.fpp
SRCPY=nntester.py parserProd.py modelParser.py
OBJ=${SRC:.f90=.o}
OBJ2=${OBJ:.fpp=.o}
tests = lstm_gemm lstm_cell

output: $(OBJ2)
	$(FC) $(FFLAGS) -o $@ $(OBJ2)


%.o: %.f90
	$(FC) $(FFLAGS) -o $@ -c $<

.PRECIOUS : %.f90
%.f90: %.fpp variables.fpp
	fypp $< $*.f90

test: ex1 output
	./output 2> outputCase.txt


ex1: modelParserONNX.py
	python3 -Wi goldenFiles/$(case)/$(case).py
	python3 -Wi modelParserONNX.py $(case)

clean:
	rm *.o *.mod output
