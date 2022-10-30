#:set architecture = [('Transpose', ['input'], [[2, 1, 3]]), ('LSTM', ['input', 'hidden_state', 'cell_state'], ['output0'], None), ('Squeeze', ('output0', 4), ['output1'], [[1]]), ('Transpose', ['output1'], [[2, 1, 3]]), ('Reshape', ('output1', 3), ['output2'], [[2, 2]], [0]), ('Gemm', ['output2', 1], None), ('Relu', ['output2'], [2]), ('Gemm', ['output2', 1], None), ('Sigmoid', ['output2'], [2]), ('Gemm', ['output2', 1], None), ('Relu', ['output2'], [2]), ('Gemm', ['output2', 1], None), ('Sigmoid', ['output2'], [2])]
#:set inputs = [['output0', 4], ['output1', 3], ['output2', 2]]
#:set trueInputs = [['input', [1, 2, 5]], ['hidden_state', [1, 1, 2]], ['cell_state', [1, 1, 2]]]
#:set outShape = [['output', [2, 1]]]
#:set outputs = {'output': 'output2'}
