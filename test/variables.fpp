#:set architecture = [('Transpose', ['input'], [[2, 1, 3]]), ('LSTM', ['input', 'hidden_state', 'cell_state'], ['output0'], [0]), ('Reshape', ('hidden_state', 3), ['output1'], [[1, 2]], [0]), ('Gemm', ['output1', 1], None), ('Relu', ['output1'], [2]), ('Gemm', ['output1', 1], None), ('Sigmoid', ['output1'], [2]), ('Gemm', ['output1', 1], None), ('Relu', ['output1'], [2]), ('Gemm', ['output1', 1], None), ('Sigmoid', ['output1'], [2])]
#:set inputs = [['output0', 4], ['output1', 2]]
#:set trueInputs = [['input', [1, 2, 5]], ['hidden_state', [1, 1, 2]], ['cell_state', [1, 1, 2]]]
#:set outShape = [['output', [1, 1]]]
#:set outputs = {'output': 'output1'}
