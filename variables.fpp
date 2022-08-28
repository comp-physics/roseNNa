#:set architecture = [('Gemm', ['input', 1], None), ('Relu', ['input'], [2]), ('Gemm', ['input', 1], None), ('Sigmoid', ['input'], [2]), ('Gemm', ['input', 1], None), ('Sigmoid', ['input'], [2]), ('Gemm', ['input', 1], None), ('Relu', ['input'], [2]), ('Gemm', ['input', 1], None), ('Sigmoid', ['input'], [2])]
#:set inputs = [['input', 2]]
#:set trueInputs = [['input', 2]]
#:set outShape = [['output', 2]]
#:set outputs = {'output': 'input'}
