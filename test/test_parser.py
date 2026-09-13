import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fLibrary"))
import onnx_helpers as H

failures = []

def check(cond, name):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}")
        failures.append(name)

def test_stranspose_is_column_major():
    for shape in [(5,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
        a = np.arange(int(np.prod(shape))).reshape(shape)
        expected = " ".join(str(x) for x in a.flatten(order="F").tolist())
        check(H.stranspose(a) == expected, f"stranspose column-major {shape}")

def test_stringer():
    check(H.stringer([1, 2, 3]) == "1 2 3", "stringer joins with spaces")
    check(H.stringer([]) == "", "stringer handles empty")

def test_reshape_parser_resolves_negative_one():
    check(H.reshapeParser([-1, 4], [2, 2, 4]) == [4, 4], "reshapeParser resolves -1")
    check(H.reshapeParser([2, 4], [2, 4]) == [2, 4], "reshapeParser passes through")

def test_regate_lstm_reorders_iofc_to_ifgo():
    h = 2
    # blocks tagged by gate: ONNX order is i, o, f, c
    onnx_w = np.concatenate([
        np.full((h, 3), 0.0),   # i
        np.full((h, 3), 1.0),   # o
        np.full((h, 3), 2.0),   # f
        np.full((h, 3), 3.0),   # c/g
    ], axis=0)
    got = H.regateLSTM(onnx_w, axis=0)
    check(np.all(got[0*h:1*h] == 0.0), "regateLSTM keeps i first")
    check(np.all(got[1*h:2*h] == 2.0), "regateLSTM moves f second")
    check(np.all(got[2*h:3*h] == 3.0), "regateLSTM moves g third")
    check(np.all(got[3*h:4*h] == 1.0), "regateLSTM moves o last")

def test_regate_lstm_handles_direction_axis():
    h = 2
    onnx_w = np.arange(1 * 4 * h * 3, dtype=float).reshape(1, 4 * h, 3)
    got = H.regateLSTM(onnx_w, axis=1)
    check(got.shape == (1, 4 * h, 3), "regateLSTM preserves shape with direction axis")
    check(np.all(got[0, 1 * h:2 * h] == onnx_w[0, 2 * h:3 * h]), "regateLSTM remaps along axis 1")

def test_four_d_transform_right_aligns():
    # (1,4,3,3) and (4,) do not broadcast under numpy/ONNX semantics
    # (np.broadcast_shapes on that pair raises): right-alignment puts the 4
    # at the last axis, colliding with the true last axis of 3. A validating
    # fourDTransform must reject it rather than guess.
    try:
        H.fourDTransform([1,4,3,3], (4,))
        check(False, "fourDTransform rejects a non-broadcastable trailing vector")
    except ValueError:
        check(True, "fourDTransform rejects a non-broadcastable trailing vector")
    check(H.fourDTransform([1,3,3,3], (3,3)) == [1,1,3,3],
          "fourDTransform right-aligns a 2D add against a 4D target")
    check(H.fourDTransform([1,4,3,3], (3,3)) == [1,1,3,3],
          "fourDTransform right-aligns regardless of channel count")
    check(H.fourDTransform([1,4,3,3], (1,4,1,1)) == [1,4,1,1],
          "fourDTransform passes through an already-4D shape")

def test_sanitize_produces_fortran_identifiers():
    import re
    ident = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")
    for raw in ["onnx::Gemm_0", "3", "/layer1/Gemm_output_0", "input", "a.b.c", "", "x" * 200]:
        got = H.sanitize(raw)
        check(bool(ident.match(got)), f"sanitize({raw[:30]!r}) -> {got!r} is a valid Fortran identifier")
    check(H.sanitize("input") == "v_input", "sanitize prefixes a valid lowercase name")
    check(H.sanitize("onnx::Gemm_0") != H.sanitize("onnx::Gemm_1"),
          "sanitize keeps distinct names distinct")

def test_sanitize_avoids_fortran_collisions():
    # Fortran identifiers are case-insensitive and share one scope with the
    # generated locals and the procedures the generated body calls.
    check(H.sanitize("Input").lower() != H.sanitize("input").lower(),
          "sanitize separates names that differ only by case")
    reserved = {"i0", "o0", "output0", "t1", "t2", "conv", "lstm", "linear_layer", "max_pool",
                "avgpool", "reshape", "size", "transpose", "relu2d", "sigmoid2d", "tanhh2d"}
    for raw in ["i0", "o0", "output0", "T1", "t2", "conv", "reshape", "relu2d", "LSTM"]:
        check(H.sanitize(raw).lower() not in reserved,
              f"sanitize({raw!r}) avoids a generated or called name")

def test_check_supported_rejects_unimplemented_attrs():
    def raises(fn):
        try:
            fn()
        except NotImplementedError:
            return True
        return False

    # attributes roseNNa parses but ignores
    check(raises(lambda: H.checkSupported("Conv", {"dilations": [2, 2], "kernel_shape": [3, 3], "pads": [0]*4, "strides": [1, 1]})),
          "rejects dilations != 1")
    check(raises(lambda: H.checkSupported("MaxPool", {"ceil_mode": 1, "kernel_shape": [2, 2], "pads": [0]*4, "strides": [1, 1]})),
          "rejects ceil_mode = 1")
    check(raises(lambda: H.checkSupported("Conv", {"kernel_shape": [3, 3], "pads": [1, 1, 2, 2], "strides": [1, 1]})),
          "rejects asymmetric pads")
    check(raises(lambda: H.checkSupported("Conv", {"kernel_shape": [3, 5], "pads": [0]*4, "strides": [1, 1]})),
          "rejects non-square kernels")
    check(not raises(lambda: H.checkSupported("Conv", {"dilations": [1, 1], "kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [2, 2]})),
          "accepts a supported Conv")

def test_check_supported_required_kernel_and_pooling_padding():
    def raises(fn):
        try:
            fn()
        except NotImplementedError:
            return True
        return False

    # kernel_shape is required for pooling, optional (inferred) for Conv
    check(raises(lambda: H.checkSupported("MaxPool", {"pads": [0]*4, "strides": [1, 1]})),
          "rejects MaxPool with no kernel_shape")
    check(raises(lambda: H.checkSupported("AveragePool", {"pads": [0]*4, "strides": [1, 1]})),
          "rejects AveragePool with no kernel_shape")
    check(not raises(lambda: H.checkSupported("Conv", {"kernel_shape": [3, 3], "pads": [0]*4})),
          "accepts Conv whose kernel_shape was inferred")
    # AveragePool divisor: roseNNa always divides by the full kernel area
    check(raises(lambda: H.checkSupported("AveragePool", {"kernel_shape": [3, 3], "pads": [1]*4, "count_include_pad": 0})),
          "rejects padded AveragePool with count_include_pad=0")
    check(raises(lambda: H.checkSupported("AveragePool", {"kernel_shape": [3, 3], "pads": [1]*4})),
          "rejects padded AveragePool with count_include_pad absent (ONNX default 0)")
    check(not raises(lambda: H.checkSupported("AveragePool", {"kernel_shape": [3, 3], "pads": [1]*4, "count_include_pad": 1})),
          "accepts padded AveragePool with count_include_pad=1")
    check(not raises(lambda: H.checkSupported("AveragePool", {"kernel_shape": [3, 3], "pads": [0]*4})),
          "accepts unpadded AveragePool regardless of count_include_pad")
    # AveragePool never computes SAME padding
    check(raises(lambda: H.checkSupported("AveragePool", {"kernel_shape": [3, 3], "auto_pad": "SAME_UPPER"})),
          "rejects AveragePool with auto_pad=SAME_UPPER")
    check(not raises(lambda: H.checkSupported("AveragePool", {"kernel_shape": [3, 3], "auto_pad": "VALID"})),
          "accepts AveragePool with auto_pad=VALID")
    # grouped convolution reads past the weight array in the Fortran conv
    check(raises(lambda: H.checkSupported("Conv", {"kernel_shape": [3, 3], "group": 2})),
          "rejects grouped Conv (group=2)")
    check(not raises(lambda: H.checkSupported("Conv", {"kernel_shape": [3, 3], "group": 1})),
          "accepts ungrouped Conv (group=1)")

def test_pad_is_rejected_unless_identity():
    def raises(fn):
        try:
            fn()
        except NotImplementedError:
            return True
        return False

    check(raises(lambda: H.checkPadIsNoop([0, 0, 1, 1, 0, 0, 1, 1])),
          "rejects a Pad node with nonzero pads")
    check(not raises(lambda: H.checkPadIsNoop([0]*8)),
          "accepts an all-zero Pad node")

if __name__ == "__main__":
    test_stranspose_is_column_major()
    test_stringer()
    test_reshape_parser_resolves_negative_one()
    test_regate_lstm_reorders_iofc_to_ifgo()
    test_regate_lstm_handles_direction_axis()
    test_four_d_transform_right_aligns()
    test_sanitize_produces_fortran_identifiers()
    test_sanitize_avoids_fortran_collisions()
    test_check_supported_rejects_unimplemented_attrs()
    test_check_supported_required_kernel_and_pooling_padding()
    test_pad_is_rejected_unless_identity()
    print(f"PARSER TESTS: {len(failures)} failure(s)")
    sys.exit(1 if failures else 0)
