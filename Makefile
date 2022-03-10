FC=gfortran
FFLAGS=-O3 -Wall -Wextra
SRC=activation_funcs.f90 layers.f90 derived_types.f90 reader2.f90 linearV3.f90
SRCPY=nntester.py parserProd.py modelParser.py
OBJ=${SRC:.f90=.o}

output: $(OBJ)
	$(FC) -o $@ $(OBJ)


%.o: %.f90
	$(FC) -o $@ -c $<

run: $(SRCPY)
	python3 nntester.py
	python3 parserProd.py
	python3 modelParser.py

clean:
	rm *.o *.mod output
