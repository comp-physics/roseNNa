"""Pure helpers shared by modelParserONNX.py. No side effects at import."""
import itertools
import numpy as np


def stranspose(arr):
    """Flatten in Fortran (column-major) order, space separated."""
    shape = arr.shape
    combs = [x for x in range(len(shape))]
    for dim1, dim2 in itertools.combinations(combs, 2):
        dim = combs.copy()
        dim[dim1] = dim2
        dim[dim2] = dim1
        arr = np.transpose(arr, dim)
    return stringer(arr.flatten().tolist())


def stringer(mat):
    return " ".join(str(elem) for elem in mat)


def reshapeParser(reshape, trueShape):
    """Resolve a single -1 in an ONNX Reshape target against the true shape."""
    if -1 not in reshape:
        return reshape
    ind = reshape.index(-1)
    res = np.prod(reshape) * -1
    true = np.prod(trueShape)
    reshape[ind] = int(true / res)
    return reshape


def fakeFourD(inp):
    return (4 - len(inp)) * [1] + list(inp)


def fourDTransform(trueshape, toBeTransformedShape):
    new = [1, 1, 1, 1]
    for dim in toBeTransformedShape:
        try:
            find = trueshape.index(dim)
            new[find - len(trueshape)] = dim
        except ValueError:
            pass
    return new


def spreadInfo(trueShape, toBeTransformedShape):
    ret = []
    for index, dim in enumerate(toBeTransformedShape):
        if trueShape[index] != dim:
            ret.append(index + 1)
            ret.append(trueShape[index])
    return ret


# ONNX stores LSTM gates as (input, output, forget, cell); roseNNa's
# lstm_cell consumes PyTorch order (input, forget, gate/cell, output).
ONNX_TO_ROSENNA_GATES = [0, 2, 3, 1]


def regateLSTM(arr, axis=0):
    """Reorder the 4 gate blocks of an ONNX LSTM W/R/B tensor along `axis`."""
    n = arr.shape[axis]
    if n % 4 != 0:
        raise ValueError(f"LSTM gate axis {axis} has length {n}, not a multiple of 4")
    h = n // 4
    blocks = [
        np.take(arr, range(g * h, (g + 1) * h), axis=axis)
        for g in ONNX_TO_ROSENNA_GATES
    ]
    return np.concatenate(blocks, axis=axis)
