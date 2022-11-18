import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnx import numpy_helper
import numpy as np
import timeit

class Batch_Net_5_2(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, out_dim):
        super(Batch_Net_5_2, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5), nn.ReLU(True))
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

fieldNames = ['C2H3O1-2','C3H4-P','C7H14OOH','CH3CHO','H2O','NC7KET','C2H3','C3H5-A','C7H14O','CH3COCH2','H2','O2','C2H4','C3H5O','C7H15-1','CH3CO','HCCO','OH','C2H5CHO','C3H6','C7H15O2','CH3O2H','HCO','O','C2H5COCH2','C4H6','C7H15','CH3O2','HO2','PC4H9','C2H5O','C4H7O','C7H16','CH3OH','H','C2H5','C4H7','CH2CHO','CH3O','N2','C2H6','C4H8-1','CH2CO','CH3','NC3H7CHO','C2H','C5H10-1','CH2OH','CH4','NC3H7COCH2','C2H2','C3H2','C5H11-1','CH2O','CO2','NC3H7COCH3','C2H3CHO','C3H3','C5H9','CH2-S','CO','NC3H7CO','C2H3CO','C3H4-A','C7H14OOHO2','CH3CHCO','H2O2','NC3H7']

INPUT_FEATURE_DIM = 2
OUTPUT_FEATURE_DIM = len(fieldNames)
HIDDEN_1_DIM = 8
HIDDEN_2_DIM = 16
HIDDEN_3_DIM = 32
HIDDEN_4_DIM = 16
HIDDEN_5_DIM = 8
model = Batch_Net_5_2(INPUT_FEATURE_DIM, HIDDEN_1_DIM, HIDDEN_2_DIM, HIDDEN_3_DIM, HIDDEN_4_DIM, HIDDEN_5_DIM, OUTPUT_FEATURE_DIM)
inp = torch.ones(1,2) #change this based on the actual input
logits = model(inp)
print(logits.flatten().tolist())




SETUP_CODE = '''
import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnx import numpy_helper
import numpy as np
import timeit

class Batch_Net_5_2(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, n_hidden_5, out_dim):
        super(Batch_Net_5_2, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, n_hidden_3), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(n_hidden_3, n_hidden_4), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(n_hidden_4, n_hidden_5), nn.ReLU(True))
        self.layer6 = nn.Sequential(nn.Linear(n_hidden_5, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

fieldNames = ['C2H3O1-2','C3H4-P','C7H14OOH','CH3CHO','H2O','NC7KET','C2H3','C3H5-A','C7H14O','CH3COCH2','H2','O2','C2H4','C3H5O','C7H15-1','CH3CO','HCCO','OH','C2H5CHO','C3H6','C7H15O2','CH3O2H','HCO','O','C2H5COCH2','C4H6','C7H15','CH3O2','HO2','PC4H9','C2H5O','C4H7O','C7H16','CH3OH','H','C2H5','C4H7','CH2CHO','CH3O','N2','C2H6','C4H8-1','CH2CO','CH3','NC3H7CHO','C2H','C5H10-1','CH2OH','CH4','NC3H7COCH2','C2H2','C3H2','C5H11-1','CH2O','CO2','NC3H7COCH3','C2H3CHO','C3H3','C5H9','CH2-S','CO','NC3H7CO','C2H3CO','C3H4-A','C7H14OOHO2','CH3CHCO','H2O2','NC3H7']

INPUT_FEATURE_DIM = 2
OUTPUT_FEATURE_DIM = len(fieldNames)
HIDDEN_1_DIM = 8
HIDDEN_2_DIM = 16
HIDDEN_3_DIM = 32
HIDDEN_4_DIM = 16
HIDDEN_5_DIM = 8
model = Batch_Net_5_2(INPUT_FEATURE_DIM, HIDDEN_1_DIM, HIDDEN_2_DIM, HIDDEN_3_DIM, HIDDEN_4_DIM, HIDDEN_5_DIM, OUTPUT_FEATURE_DIM)
inp = torch.ones(1,2) #change this based on the actual input
'''
TEST_CODE = '''
with torch.jit.optimized_execution(False):
    logits = model(inp)'''
t = timeit.repeat(setup = SETUP_CODE,
                    stmt = TEST_CODE,
                    repeat = 100,
                    number = 1)
median = np.median(np.array(t))
print("Python Time:" + str(median))


with open("inputs.fpp",'w') as f1:
    inputs = inp.flatten().tolist()
    inpShapeDict = {'inputs': list(inp.shape)}
    inpDict = {'inputs':inputs}
    f1.write(f"""#:set inpShape = {inpShapeDict}""")
    f1.write("\n")
    f1.write(f"""#:set arrs = {inpDict}""")
    f1.write("\n")
    f1.write("a")

def stringer(mat):
    s = ""
    for elem in mat:
        s += str(elem) + " "
    return s.strip()


filePath = "../goldenFiles/batchnet/"
with open(filePath+"batchnet.txt", "w") as f2:
    f2.write(stringer(list(logits.shape)))
    f2.write("\n")
    f2.write(stringer(logits.flatten().tolist()))

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"batchnet.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )

torch.onnx.export(model,               # model being run
                  inp,                         # model input (or a tuple for multiple inputs)
                  filePath+"batchnet_weights.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  )
