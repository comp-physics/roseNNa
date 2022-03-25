FC=gfortran
FFLAGS=-O3 -Wall -Wextra -fcheck=bounds
SRC=activation_funcs.f90 layers.f90 derived_types.f90 readTester.f90 linearV3.f90
SRCPY=nntester.py parserProd.py modelParser.py
OBJ=${SRC:.f90=.o}

output: $(OBJ)
	$(FC) $(FFLAGS) -o $@ $(OBJ)


%.o: %.f90
	$(FC) $(FFLAGS) -o $@ -c $<

run: $(SRCPY)
	python3 nntester.py
	python3 parserProd.py
	python3 modelParser.py

all:
	python3 lstm_linear.py
	python3 parserProd.py
	python3 modelParser.py
	./output

clean:
	rm *.o *.mod output
