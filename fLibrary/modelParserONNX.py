import torch
import torch.nn as nn
import numpy as np
import torch.onnx
import onnx
from onnx import numpy_helper
import argparse
import sys

from onnx_helpers import (
    stranspose, reshapeParser,
    fourDTransform, fakeFourD, spreadInfo,
    regateLSTM, sanitize,
    checkSupported, checkPadIsNoop,
    checkLSTMSupported, checkGemmBias,
)

parser = argparse.ArgumentParser()

parser.add_argument('--onnxfile',"-f", required=True, help="Please provide .onnx file of your pretrained model.")
parser.add_argument('--weights',"-w", help="(Deprecated, ignored) A second unoptimized .onnx export is no longer needed; weights are read by name from the file given to -f.")
parser.add_argument('--inferred',"-i", help="(Optional) Please provide .onnx file that has inferred shapes")


args = parser.parse_args()

file = args.onnxfile
weights = args.weights
inferred = args.inferred


onnxModel = onnx.load(file)

if weights is not None:
    print("note: --weights/-w is no longer needed and is ignored; "
          "weights are now read by name from the structure file.")

#sometimes the inferred shapes is too big, so we need to store an external file of the precomputed inferred shapes
try:
    inferred = onnx.load(inferred)
    value_info = inferred.graph.value_info
except (TypeError, FileNotFoundError, onnx.checker.ValidationError):
    value_info = onnx.shape_inference.infer_shapes(onnxModel).graph.value_info

nodes = onnxModel.graph.node #all layers of model that will be parsed

ioMap = {} #mapping input names to output names
initializer = {} #holds (weight dimensions, np.array of weights)
intermediateShapes = {} #holds intermediate shapes of layers
inputs = [] #general inputs to the model (including intermediary stuff needed for fortran to process)
input_shapes = {} #shapes of inputs
constants = {} #constants and initalizer weights are the places where weights of the model could be stored
for inp in onnxModel.graph.input:
    ioMap[inp.name] = sanitize(inp.name)
    input_shapes[inp.name] = [d.dim_value for d in inp.type.tensor_type.shape.dim]

for inter in value_info:
    intermediateShapes[inter.name] = [d.dim_value for d in inter.type.tensor_type.shape.dim]

for weights in onnxModel.graph.initializer:
    initializer[weights.name] = (weights.dims,numpy_helper.to_array(weights))
out = {}
for x in onnxModel.graph.output:
    out[x.name] = [d.dim_value if d.dim_value!=0 else 1 for d in x.type.tensor_type.shape.dim]

for x in onnxModel.graph.node:
    if x.op_type == "Constant":
        constants[x.output[0]] = numpy_helper.to_array(x.attribute[0].t)

outputs = {} #what outputs corresponds to, need to export
outShape = [] #what shape to instantiate the output name to, need to export
modelArch = [] #need to export
extra = "0"


def findWeightsInitializer(input_name):
    if input_name in initializer:
        return initializer[input_name][1]
    if input_name in constants:
        return constants[input_name]
    raise KeyError(
        f"no weights found for '{input_name}'; it is neither an initializer "
        f"nor a Constant node output"
    )


#ONNX parser
#onnxModel.txt => holds model structure
#onnxWeights.bin => holds model's weights, as float64 in column-major order, as a raw binary stream

#ioMap => dictionary that maps (outputs) -> (inputs)
#initializer => holds weights dims

#modelArch => (layer_name, input_list[], parameters) to call respective subroutines in fypp
print("starting to write weights..")
print("starting parsing...")
for node in nodes:
    print(node.op_type)
with open('onnxModel.txt','w') as f, open('onnxWeights.bin', 'wb') as f2:
    f.write(str(len(nodes)))
    f.write("\n")
    for node in nodes:
        layer = node.op_type

        for index,x in enumerate(node.output):
            if x in out:
                outShape.append([sanitize(x),out[x]])
        if layer == "Transpose": #for this, make sure order is set to tuple[2] and shape is set accordingly
            f.write(layer)
            f.write("\n")
            names = {n.name:n.i if n.type==2 else n.ints for n in node.attribute}
            try:
                default = [x for x in range(len(intermediateShapes[node.input[0]])-1,-1,-1)]
            except KeyError:
                default = [x for x in range(len(input_shapes[node.input[0]])-1,-1,-1)]
            attributes = names.get('perm', default)
            #make sure there is a node.attribute[0], otherwise default is to reverse all the dimensions
            modelArch.append(("Transpose",[ioMap[node.input[0]]], [list(map(lambda x: x+1,attributes))])) #"order"

            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "LSTM": #changes shape
            # reject attributes that lstm_cell does not implement; absent ones take the ONNX defaults
            lstmAttrs = {}
            for attr in node.attribute:
                if attr.name == "direction":
                    lstmAttrs["direction"] = attr.s.decode("ASCII")
                elif attr.name == "activations":
                    lstmAttrs["activations"] = [a.decode("ASCII") for a in attr.strings]
                elif attr.name == "clip":
                    lstmAttrs["clip"] = attr.f
                elif attr.name in ("input_forget", "layout"):
                    lstmAttrs[attr.name] = attr.i
            checkLSTMSupported(lstmAttrs)
            f.write(layer)
            f.write("\n")
            writeHCs = False
            try:
                modelArch.append(("LSTM", [ioMap[node.input[0]], ioMap[node.input[5]], ioMap[node.input[6]]], ["output"+extra], [0])) #input = ["input", "hidden_state", "cell_state"]
                f.write("0")
                f.write("\n")
            except (KeyError, IndexError):
                modelArch.append(("LSTM", [ioMap[node.input[0]], "output"+str(int(extra)+1),"output"+str(int(extra)+2)], ["output"+extra], [1])) #input = ["input", "hidden_state", "cell_state"]
                writeHCs = True
                f.write("1")
                f.write("\n")
            inputs.append(["output"+extra,len(intermediateShapes[node.output[0]])])
            for inp in node.input[1:3]: #represents ONNX's locations of weights
                for dim in initializer[inp][0]:
                    f.write(str(dim)+" ")
                f2.write(np.asarray(regateLSTM(findWeightsInitializer(inp), axis=1), dtype='<f8').flatten(order='F').tobytes())
                f.write("\n")
            #check if bias exists
            # ONNX packs Wb and Rb into one (num_directions, 8*hidden) tensor;
            # split into the two halves first, then regate each half.
            wb, rb = np.split(findWeightsInitializer(node.input[3]), 2, axis=1)
            for half in (wb, rb):
                f.write(str(int(initializer[node.input[3]][0][1]/2)))
                f.write("\n")
                f2.write(np.asarray(regateLSTM(half, axis=1), dtype='<f8').flatten(order='F').tobytes())
            if writeHCs:
                inpShape = intermediateShapes[node.input[0]]
                batch_size = inpShape[1]
                if inpShape[1] == 0:
                    batch_size = 1
                hidden = 0
                for x in node.attribute:
                    if x.name == "hidden_size":
                        hidden = x.i
                        break
                shape = (1,batch_size,hidden)
                for x in range(2):
                    for s in shape:
                        f.write(str(s)+" ")
                    f2.write(np.asarray(np.zeros(shape), dtype='<f8').flatten(order='F').tobytes())
                    f.write("\n")
            if not writeHCs:
                ioMap[node.output[0]] = "output" + extra
                extra = str(int(extra)+1)
                ioMap[node.output[1]] = ioMap[node.input[-2]]

                ioMap[node.output[2]] = ioMap[node.input[-1]]
            else:
                ioMap[node.output[0]] = "output" + extra
                extra = str(int(extra)+1)

                inputs.append(["output"+extra,len(intermediateShapes[node.output[1]])])
                ioMap[node.output[1]] = "output" + extra
                extra = str(int(extra)+1)

                inputs.append(["output"+extra,len(intermediateShapes[node.output[2]])])
                ioMap[node.output[2]] = "output" + extra
                extra = str(int(extra)+1)

        elif layer == "Gemm":
            if len(node.input) >= 3:
                # read_linear reads a rank-1 bias; check before anything is written
                checkGemmBias(np.shape(findWeightsInitializer(node.input[2])))
            f.write(layer)
            f.write("\n")
            #only need default for node.attribute[2].i for transInput/B=0
            names = {n.name:n.i if n.type==2 else n.ints for n in node.attribute}
            if names.get('transA', 0):
                raise NotImplementedError(
                    "Gemm transA=1 is not supported by roseNNa")
            for attr in node.attribute:
                if attr.name in ('alpha', 'beta') and abs(attr.f - 1.0) > 1e-12:
                    raise NotImplementedError(
                        f"Gemm {attr.name}={attr.f} is not supported by roseNNa "
                        f"(only 1.0)")
            attributes = names.get('transB', 0)
            modelArch.append(("Gemm", [ioMap[node.input[0]],attributes], None))
            numzs = 0
            #check if bias exists
            if len(node.input) < 3:
                for inp in node.input[1:]:
                    numzs = initializer[inp][0][0]
                    for dim in initializer[inp][0]:
                        f.write(str(dim)+ " ")
                    f2.write(np.asarray(findWeightsInitializer(inp), dtype='<f8').flatten(order='F').tobytes())
                    f.write("\n")
                f.write(str(numzs))
                f.write("\n")
                f2.write(np.asarray(np.zeros(numzs), dtype='<f8').flatten(order='F').tobytes())
            else:
                for inp in node.input[1:3]:
                    for dim in initializer[inp][0]:
                        f.write(str(dim)+ " ")
                    f2.write(np.asarray(findWeightsInitializer(inp), dtype='<f8').flatten(order='F').tobytes())
                    f.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]

        #check notion summer start
        elif layer == "Squeeze": #changes shape
            f.write(layer)
            f.write("\n")
            rank = len(intermediateShapes[node.input[0]])
            axes = None
            if len(node.input) > 1:
                axes = findWeightsInitializer(node.input[-1]).tolist()
            else:
                for attr in node.attribute:
                    if attr.name == "axes":
                        axes = list(attr.ints)
                        break
            if axes is None:
                # ONNX default: squeeze every axis of extent 1
                axes = [i for i, d in enumerate(intermediateShapes[node.input[0]]) if d == 1]
            axes = [a if a >= 0 else a + rank for a in axes]
            modelArch.append(("Squeeze", (ioMap[node.input[0]], rank),
                              ["output" + extra], [axes]))
            inputs.append(["output"+extra, len(intermediateShapes[node.output[0]])])
            ioMap[node.output[0]] = "output" + extra
            extra = str(int(extra)+1)


        #check notion summer start
        elif layer == "Reshape": #changes shape
            #no default changes needed
            f.write(layer)
            f.write("\n")
            try:
                modelArch.append(("Reshape", (ioMap[node.input[0]], len(intermediateShapes[node.input[0]])),["output" + extra], [reshapeParser(findWeightsInitializer(node.input[-1]).tolist(), intermediateShapes[node.input[0]])],[0])) #new shape
                f.write("0")
                f.write("\n")
            except KeyError:
                modelArch.append(("Reshape", (ioMap[node.input[0]], len(initializer[node.input[0]][0])),["output" + extra], [reshapeParser(findWeightsInitializer(node.input[-1]).tolist(), initializer[node.input[0]][0])], [1])) #new shape
                f.write(str(len(initializer[node.input[0]][0])))
                f.write("\n")
                for dim in initializer[node.input[0]][0]:
                    f.write(str(dim)+ " ")
                f.write("\n")
                f2.write(np.asarray(findWeightsInitializer(node.input[0]), dtype='<f8').flatten(order='F').tobytes())
            inputs.append(["output"+extra, len(intermediateShapes[node.output[0]])])
            ioMap[node.output[0]] = "output" + extra
            extra = str(int(extra)+1)

        elif layer == "Conv":
            f.write(layer)
            f.write("\n")
            attributes = {}
            auto_pad = False
            for attr in node.attribute:
                name = str(attr.name)
                if name == "group":
                    attributes['group'] = attr.i
                elif name == "auto_pad":
                    attributes['auto_pad'] = attr.s.decode('ASCII')
                    if attributes['auto_pad'] not in ("NOTSET", "VALID"):
                        auto_pad = True
                else:
                    attributes[attr.name] = attr.ints

            # kernel_shape is optional on Conv; infer it from the weight dims (out, in, kh, kw)
            attributes.setdefault('kernel_shape', list(initializer[node.input[1]][0][2:]))

            if auto_pad: # DEAL WITH STRIDE > 1?
                kernel_shape = attributes['kernel_shape'][0]
                pad_total = kernel_shape - 1
                pad = int(pad_total/2)
                if pad_total % 2 != 0:
                    if attributes['auto_pad'] == "SAME_UPPER":
                        attributes['pads'] = [pad,pad,pad+1,pad+1]
                    else:
                        attributes['pads'] = [pad+1,pad+1,pad,pad]
                else:
                    attributes['pads'] = [pad]*4
            attributes.setdefault('pads', [0, 0, 0, 0])
            attributes.setdefault('strides', [1, 1])
            attributes.setdefault('dilations', [1, 1])
            checkSupported("Conv", attributes)
            names = {n.name:n.i if n.type==2 else n.ints for n in node.attribute}
            modelArch.append(("Conv", [ioMap[node.input[0]]], [names.get('dilations', [1,1]), attributes['kernel_shape'], attributes['pads'], names.get('strides', [1,1])])) #(dilations, kernel_shape, pads, strides)

            if len(node.input) < 3: #if bias does not exist, default = 0s
                numzs = 0
                for inp in node.input[1:]:
                    numzs = initializer[inp][0][0]
                    for dim in initializer[inp][0]:
                        f.write(str(dim)+ " ")
                    f2.write(np.asarray(findWeightsInitializer(inp), dtype='<f8').flatten(order='F').tobytes())
                    f.write("\n")
                f.write(str(numzs))
                f.write("\n")
                f2.write(np.asarray(np.zeros(numzs), dtype='<f8').flatten(order='F').tobytes())
            else:
                for inp in node.input[1:3]:
                    for dim in initializer[inp][0]:
                        f.write(str(dim)+ " ")
                    f2.write(np.asarray(findWeightsInitializer(inp), dtype='<f8').flatten(order='F').tobytes())
                    f.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "MaxPool":
            #no default changes needed
            f.write(layer)
            f.write("\n")
            attributes = {}
            auto_pad = False
            #--SOME CONSTANTS THAT ARE REQUIRED IN ARGUMENTS AND MAY NOT APPEAR IN ONNX--
            attributes['ceil_mode'] = 0
            #-------------
            for attr in node.attribute:
                name = str(attr.name)
                if name == "ceil_mode":
                    attributes[attr.name] = attr.i
                elif name == "auto_pad":
                    attributes['auto_pad'] = attr.s.decode('ASCII')
                    if attributes['auto_pad'] not in ("NOTSET", "VALID"):
                        auto_pad = True
                else:
                    attributes[attr.name] = attr.ints
            attributes.setdefault('pads', [0, 0, 0, 0])
            attributes.setdefault('strides', [1, 1])
            if auto_pad:  # DEAL WITH STRIDE > 1?
                kernel_shape = attributes['kernel_shape'][0]
                pad_total = kernel_shape - 1
                pad = int(pad_total/2)
                if pad_total % 2 != 0:
                    if attributes['auto_pad'] == "SAME_UPPER":
                        attributes['pads'] = [pad,pad,pad+1,pad+1]
                    else:
                        attributes['pads'] = [pad+1,pad+1,pad,pad]
                else:
                    attributes['pads'] = [pad]*4
            checkSupported("MaxPool", attributes)
            modelArch.append(("MaxPool", [ioMap[node.input[0]]], [attributes['ceil_mode'],attributes['pads'],attributes['strides']])) #(ceil_mode, pads, strides)
            f.write(str(attributes['kernel_shape'][0]))
            f.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "AveragePool":
            f.write(layer)
            f.write("\n")
            #https://onnx.ai/onnx/api/mapping.html#l-onnx-types-mapping
            names = {n.name:n.i if n.type==2 else n.ints for n in node.attribute}
            poolAttrs = {
                'ceil_mode': names.get('ceil_mode', 0),
                'pads': names.get('pads', [0, 0, 0, 0]),
                'strides': names.get('strides', [1, 1]),
                'kernel_shape': names.get('kernel_shape'),
                'count_include_pad': names.get('count_include_pad', 0),
                # `names` maps STRING attributes to an empty ints list, so read auto_pad directly
                'auto_pad': next((a.s.decode('ASCII') for a in node.attribute if a.name == 'auto_pad'), 'NOTSET'),
            }
            checkSupported("AveragePool", poolAttrs)
            attributes = [poolAttrs['ceil_mode'], poolAttrs['pads'], poolAttrs['strides'], poolAttrs['kernel_shape']]
            modelArch.append(("AveragePool", [ioMap[node.input[0]]], attributes[:3])) #(ceil_mode, pads, strides)
            f.write(str(attributes[-1][0]))
            f.write("\n")
            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "Add":
            #no default changes needed
            f.write(layer)
            f.write("\n")
            fourd = fourDTransform(intermediateShapes[node.input[0]],findWeightsInitializer(node.input[-1]).shape)
            true = fakeFourD(intermediateShapes[node.input[0]])
            modelArch.append(("Add",[ioMap[node.input[0]]], [true, spreadInfo(true,fourd),len(intermediateShapes[node.input[0]])])) #[trueshape, need to be broadcasted and added SHAPE]
            for dim in fourd:
                f.write(str(dim) + " ")
            f.write("\n")
            f2.write(np.asarray(findWeightsInitializer(node.input[1]), dtype='<f8').flatten(order='F').tobytes())
            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "MatMul":
            #no default changes needed
            try:
                modelArch.append(("MatMul",[ioMap[node.input[0]],ioMap[node.input[1]]], [len(intermediateShapes[node.input[0]])])) #[trueshape, need to be broadcasted and added SHAPE]
                f.write(layer)
                f.write("\n")
            except KeyError:
                f.write("Gemm")
                f.write("\n")
                # MatMul is Y = A*B with B stored as (in, out), so transB=0
                modelArch.append(("Gemm", [ioMap[node.input[0]],0], None))
                numzs = 0
                #check if bias exists
                for inp in node.input[1:]:
                    numzs = initializer[inp][0][1] # the zero bias has the output width
                    for dim in initializer[inp][0]:
                        f.write(str(dim)+ " ")
                    f2.write(np.asarray(findWeightsInitializer(inp), dtype='<f8').flatten(order='F').tobytes())
                    f.write("\n")
                f.write(str(numzs))
                f.write("\n")
                f2.write(np.asarray(np.zeros(numzs), dtype='<f8').flatten(order='F').tobytes())


            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "Pad":
            f.write(layer)
            f.write("\n")
            # opset < 11 carries pads as an attribute; opset >= 11 as the second input
            pads = None
            for attr in node.attribute:
                if attr.name == "pads":
                    pads = list(attr.ints)
            if pads is None:
                pads = findWeightsInitializer(node.input[1]).tolist()
            checkPadIsNoop(pads)
            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "Relu":
            f.write(layer)
            f.write("\n")
            modelArch.append(("Relu", [ioMap[node.input[0]]], [len(intermediateShapes[node.input[0]])]))

            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "Sigmoid":
            f.write(layer)
            f.write("\n")
            modelArch.append(("Sigmoid", [ioMap[node.input[0]]], [len(intermediateShapes[node.input[0]])]))

            ioMap[node.output[0]] = ioMap[node.input[0]]

        elif layer == "Tanh":
            f.write(layer)
            f.write("\n")
            modelArch.append(("Tanh", [ioMap[node.input[0]]], [len(intermediateShapes[node.input[0]])]))

            ioMap[node.output[0]] = ioMap[node.input[0]]
        elif layer == "Constant":
            f.write(layer)
            f.write("\n")
            continue
        else:
            raise NotImplementedError(
                f"{layer} is not supported by roseNNa. "
                f"Model architecture parsed so far: {modelArch}")
    for x in list(ioMap.keys()):
        if x in out:
            outputs[sanitize(x)] = ioMap[x]
    trueInputs = [[sanitize(x.name), [a.dim_value if a.dim_value!=0 else 1 for a in x.type.tensor_type.shape.dim]] for x in onnxModel.graph.input if x.name not in initializer]
    print(modelArch)


with open("variables.fpp",'w') as f:
    f.write(f"""#:set architecture = {modelArch}""")
    f.write("\n")
    f.write(f"""#:set inputs = {inputs}""")
    f.write("\n")
    f.write(f"""#:set trueInputs = {trueInputs}""")
    f.write("\n")
    f.write(f"""#:set outShape = {outShape}""")
    f.write("\n")
    f.write(f"""#:set outputs = {outputs}""")
    f.write("\n")
