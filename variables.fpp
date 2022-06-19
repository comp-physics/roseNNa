#:set architecture = [('Gemm', ['input', 1], None), ('Relu', ['input'], None), ('Gemm', ['input', 1], None), ('Sigmoid', ['input'], None), ('Gemm', ['input', 1], None), ('Relu', ['input'], None), ('Gemm', ['input', 1], None), ('Tanh', ['input'], None), ('Gemm', ['input', 1], None), ('Sigmoid', ['input'], None)]
#:set inputs = [['input', 2]]
#:set trueInputs = [['input', 2]]
#:set outShape = [['output', 2]]
#:set outputs = {'output': 'input'}
