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
	rm *.o objFiles/*.mod
	rm modelCreator.f90
	rm libcorelib.a
# testing: ex1 outputs
#     # runs userTesting/main file provided by user
#     ./output
#
#     # checks outputs from F90 and python
#     python3 -Wi goldenFiles/testChecker.py $(case)
