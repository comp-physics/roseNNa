import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

# An LSTM node with no initial_h/initial_c inputs, so the parser takes the
# zero-initial-state path (readOrNot=1). PyTorch always wires h0/c0 into its
# exported LSTM (as Expand outputs, even when the call omits them), so the
# graph is built directly and onnxruntime supplies the golden outputs.
# The layout mirrors a PyTorch batch_first export:
# Transpose -> LSTM -> Squeeze -> Transpose.

rng = np.random.default_rng(0)
batch_size = 1
seq_len = 4
input_dim = 5
hidden_dim = 3

# ONNX gate order is (i, o, f, c); the parser remaps it for roseNNa.
W = rng.uniform(-0.6, 0.6, (1, 4 * hidden_dim, input_dim)).astype(np.float32)
R = rng.uniform(-0.6, 0.6, (1, 4 * hidden_dim, hidden_dim)).astype(np.float32)
B = rng.uniform(-0.6, 0.6, (1, 8 * hidden_dim)).astype(np.float32)
inp = rng.standard_normal((batch_size, seq_len, input_dim)).astype(np.float32)

nodes = [
    helper.make_node("Transpose", ["input"], ["lstm_in"], perm=[1, 0, 2]),
    helper.make_node("LSTM", ["lstm_in", "W", "R", "B"], ["Y", "Y_h", "Y_c"],
                     hidden_size=hidden_dim),
    helper.make_node("Squeeze", ["Y"], ["Y_sq"], axes=[1]),
    helper.make_node("Transpose", ["Y_sq"], ["output"], perm=[1, 0, 2]),
]
graph = helper.make_graph(
    nodes, "lstm_nostate",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch_size, seq_len, input_dim])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch_size, seq_len, hidden_dim])],
    initializer=[numpy_helper.from_array(W, "W"), numpy_helper.from_array(R, "R"),
                 numpy_helper.from_array(B, "B")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
model.ir_version = 7
onnx.checker.check_model(model)

with open("inputs.fpp",'w') as f:
    inputs = inp.flatten().tolist()
    inpShapeDict = {'inputs': list(inp.shape)}
    inpDict = {'inputs': inputs}
    f.write(f"""#:set inpShape = {inpShapeDict}""")
    f.write("\n")
    f.write(f"""#:set arrs = {inpDict}""")
    f.write("\n")
    f.write("a")

def stringer(mat):
    s = ""
    for elem in mat:
        s += str(elem) + " "
    return s.strip()

filePath = "../goldenFiles/lstm_nostate/"
onnx.save(model, filePath+"lstm_nostate.onnx")

sess = ort.InferenceSession(filePath+"lstm_nostate.onnx", providers=["CPUExecutionProvider"])
logits = sess.run(["output"], {"input": inp})[0]
with open(filePath+"lstm_nostate.txt", "w") as f:
    f.write(stringer(list(logits.shape)))
    f.write("\n")
    f.write(stringer(logits.flatten().tolist()))
print(logits.flatten().tolist())
