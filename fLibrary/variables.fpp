#:set architecture = [('Gemm', ['input', 1], None), ('Relu', ['input'], [2]), ('Gemm', ['input', 1], None), ('Relu', ['input'], [2])]
#:set inputs = []
#:set trueInputs = [['input', [1, 2]]]
#:set outShape = [['output', [1, 3]]]
#:set outputs = {'output': 'input'}
