#:set architecture = [('Transpose', ['input'], [[2, 1, 3]]), ('LSTM', ['input', 'hidden_state', 'cell_state'], ['output0'], None), ('Squeeze', ('output0', 4), ['output1'], [[1]]), ('Transpose', ['output1'], [[2, 1, 3]])]
#:set inputs = [['input', 3], ['hidden_state', 3], ['cell_state', 3], ['output0', 4], ['output1', 3]]
#:set trueInputs = [['input', 3], ['hidden_state', 3], ['cell_state', 3]]
#:set outShape = [['output', 3]]
#:set outputs = {'output': 'output1'}
