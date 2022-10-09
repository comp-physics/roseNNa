#:set architecture = [('Reshape', ('Parameter193', 4), ['output0'], [[256, 10]], [1]), ('Conv', ['Input3'], [[1, 1], [5, 5], [2, 2, 2, 2], [1, 1]]), ('Add', ['Input3'], [[1, 8, 28, 28], [3, 28, 4, 28], 4]), ('Relu', ['Input3'], [4]), ('MaxPool', ['Input3'], [0, [0, 0, 0, 0], [2, 2]]), ('Conv', ['Input3'], [[1, 1], [5, 5], [2, 2, 2, 2], [1, 1]]), ('Add', ['Input3'], [[1, 16, 14, 14], [3, 14, 4, 14], 4]), ('Relu', ['Input3'], [4]), ('MaxPool', ['Input3'], [0, [0, 0, 0, 0], [3, 3]]), ('Reshape', ('Input3', 4), ['output1'], [[1, 256]], [0]), ('MatMul', ['output1', 'output0'], [2]), ('Add', ['output1'], [[1, 1, 1, 10], [], 2])]
#:set inputs = [['output0', 2], ['output1', 2]]
#:set trueInputs = [['Input3', [1, 1, 28, 28]]]
#:set outShape = [['Plus214_Output_0', [1, 10]]]
#:set outputs = {'Plus214_Output_0': 'output1'}
