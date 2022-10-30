FC=gfortran
FFLAGS=-O2 -JobjFiles
SRC1=modelCreator.fpp rosenna.f90
SRC2=userTesting.fpp
SRCBASE=activation_funcs.f90 derived_types.f90 layers.f90 readTester.f90
OBJ1=${SRC1:.fpp=.o}
OBJ2=${OBJ1:.f90=.o}
OBJ3=${SRC2:.fpp=.o}
COMP=${SRCBASE:.f90=.o}

output: $(COMP) $(OBJ2)

.PRECIOUS : %.f90
%.f90: %.fpp variables.fpp
	fypp $< $*.f90

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@


preprocess: modelParserONNX.py
   	# arg1 = model structure file (.onnx format)
   	# arg2 (optional) = weights file (.onnx format)
	python3 modelParserONNX.py -f $(args)

   	#for *.mod and *.o files
	mkdir -p objFiles

library: output
	ar crv libcorelib.a $(COMP) $(OBJ2)

clean:
	rm *.o objFiles/*.mod
	rm libcorelib.a
# testing: ex1 outputs
#     # runs userTesting/main file provided by user
#     ./output
#
#     # checks outputs from F90 and python
#     python3 -Wi goldenFiles/testChecker.py $(case)
