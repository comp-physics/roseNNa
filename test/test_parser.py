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

if __name__ == "__main__":
    test_stranspose_is_column_major()
    test_stringer()
    test_reshape_parser_resolves_negative_one()
    test_regate_lstm_reorders_iofc_to_ifgo()
    test_regate_lstm_handles_direction_axis()
    print(f"PARSER TESTS: {len(failures)} failure(s)")
    sys.exit(1 if failures else 0)
