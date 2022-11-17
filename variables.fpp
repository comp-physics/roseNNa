#:set architecture = [('Transpose', ['lstm_1_input'], [[2, 1, 3]]), ('LSTM', ['lstm_1_input', 'output1', 'output2'], ['output0'], [1]), ('Squeeze', ('output1', 3), ['output3'], [[0]]), ('Gemm', ['output3', 1], None), ('Add', ['output3'], [[1, 1, 0, 9], [3, 0], 2]), ('Tanh', ['output3'], [2])]
#:set inputs = [['output0', 4], ['output1', 3], ['output2', 3], ['output3', 2]]
#:set trueInputs = [['lstm_1_input', [1, 25, 9]]]
#:set outShape = [['dense_1', [1, 9]]]
#:set outputs = {'dense_1': 'output3'}
