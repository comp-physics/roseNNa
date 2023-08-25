#file to create pytorch model and convert to ONNX
python3 ../goldenFiles/gemm_small/gemm_small.py

#read and interpret the correspoding output files from last step
python3 modelParserONNX.py -w ../goldenFiles/gemm_small/gemm_small.onnx -f ../goldenFiles/gemm_small/gemm_small_weights.onnx

#compile the library
make library

#compile "source files" (capiTester.f90), link to the library file created, and run
gfortran -c ../examples/capiTester.f90 -IobjFiles/
gfortran -o flibrary libcorelib.a capiTester.o
./flibrary

#check whether python output from PyTorch model = roseNNa's output
python3 ../test/testChecker.py gemm_small