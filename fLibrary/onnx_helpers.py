"""Pure helpers shared by modelParserONNX.py. No side effects at import."""
import itertools
import hashlib
import re
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
    """Right-align `toBeTransformedShape` into 4 dimensions, per ONNX broadcasting.

    Every axis of the result must be either 1 or equal to the corresponding
    axis of `trueshape`, otherwise the two do not broadcast.
    """
    t = list(toBeTransformedShape)
    if len(t) > 4:
        raise ValueError(f"cannot broadcast a {len(t)}-D tensor into 4 dimensions")
    new = [1, 1, 1, 1]
    for i, d in enumerate(reversed(t)):
        new[3 - i] = d
    true4d = fakeFourD(list(trueshape))
    for i, (a, b) in enumerate(zip(true4d, new)):
        if b != 1 and b != a:
            raise ValueError(
                f"axis {i}: cannot broadcast {toBeTransformedShape} "
                f"against {trueshape} ({b} vs {a})"
            )
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


_LOWER_IDENT = re.compile(r"^[a-z][a-z0-9_]{0,60}$")


def sanitize(name):
    """Map an ONNX tensor name onto a Fortran identifier that cannot collide.

    Every result starts with ``v_``. No identifier in the library, the model
    template, or the drivers uses that prefix, so an emitted name can never
    clash with a generated local (``i0``, ``o0``, ``output0``, ``T1``), a
    called procedure (``conv``, ``lstm``), or an intrinsic (``reshape``).

    Fortran identifiers are case-insensitive, so a name that is not already a
    valid all-lowercase identifier is lowercased and given a short digest of the
    original. ``Input`` and ``input`` therefore map to different identifiers.
    Results stay within Fortran's 63-character limit.
    """
    if _LOWER_IDENT.match(name):
        return "v_" + name
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name).lower()
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
    return f"v_{cleaned[:48]}_{digest}"


def checkSupported(op, attrs):
    """Raise NotImplementedError for attributes roseNNa reads but cannot honour.

    MaxPool and AveragePool require kernel_shape (ONNX gives it no default).
    Conv may omit it; the caller infers it from the weight tensor first.
    """
    kernel = attrs.get("kernel_shape")
    if op in ("MaxPool", "AveragePool") and not kernel:
        raise NotImplementedError(
            f"{op}: kernel_shape is required by ONNX but missing from this node")

    if op == "Conv" and int(attrs.get("group", 1)) != 1:
        raise NotImplementedError(
            f"Conv: group={attrs.get('group')} (grouped or depthwise convolution) "
            f"is not supported by roseNNa")

    if op == "AveragePool" and attrs.get("auto_pad", "NOTSET") not in ("NOTSET", "VALID"):
        raise NotImplementedError(
            f"AveragePool: auto_pad={attrs.get('auto_pad')} is not supported by roseNNa")

    dilations = list(attrs.get("dilations", [1, 1]))
    if any(d != 1 for d in dilations):
        raise NotImplementedError(
            f"{op}: dilations={dilations} is parsed but ignored by roseNNa; "
            f"only dilations of 1 are supported")

    if int(attrs.get("ceil_mode", 0)) != 0:
        raise NotImplementedError(
            f"{op}: ceil_mode=1 is parsed but ignored by roseNNa; "
            f"output extents are always floored")

    kernel = list(kernel or [])
    if len(kernel) == 2 and kernel[0] != kernel[1]:
        raise NotImplementedError(
            f"{op}: non-square kernel {kernel} is not supported by roseNNa")

    pads = list(attrs.get("pads", [0, 0, 0, 0]))
    if len(pads) == 4 and (pads[0] != pads[2] or pads[1] != pads[3]):
        raise NotImplementedError(
            f"{op}: asymmetric pads {pads} are not supported by roseNNa; "
            f"padding is applied symmetrically")

    if op == "AveragePool" and any(p != 0 for p in pads) \
            and int(attrs.get("count_include_pad", 0)) != 1:
        raise NotImplementedError(
            f"AveragePool: count_include_pad=0 with pads {pads} is not supported "
            f"by roseNNa, which always divides by the full kernel area")


def checkPadIsNoop(pads):
    """Raise unless an ONNX Pad node's pads are all zero (then it is an identity)."""
    pads = list(pads)
    if any(p != 0 for p in pads):
        raise NotImplementedError(f"Pad: pads={pads} is not supported by roseNNa")
