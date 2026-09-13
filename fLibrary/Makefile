FC=gfortran
FFLAGS=-O2 -JobjFiles
SRC1=modelCreator.fpp rosenna.f90
SRCBASE=activation_funcs.f90 derived_types.f90 layers.f90 reader.f90
OBJ1=${SRC1:.fpp=.o}
OBJ2=${OBJ1:.f90=.o}
COMP=${SRCBASE:.f90=.o}

output: $(COMP) $(OBJ2)

.PRECIOUS : %.f90
%.f90: %.fpp variables.fpp
	fypp $< $*.f90

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

library: output
	mkdir -p objFiles
	ar crv libcorelib.a $(COMP) $(OBJ2)

clean:
	rm -f *.o objFiles/*.mod
	rm -f modelCreator.f90 variables.fpp
	rm -f libcorelib.a
