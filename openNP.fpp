#: set a = ['lstm', 'linear', 'linear', 'linear', 'linear']
#: set layer_dict = {}
#: for layer in a
#: if layer not in layer_dict
$: layer_dict.update([(layer,1)])
#: endif
#: if layer == 'linear'
out = linear_layer(out, layers(${layer_dict[layer]}$)%weights, layers(${layer_dict[layer]}$)%biases)
out = layers(${layer_dict[layer]}$)%fn_ptr(out)
#: elif layer == 'lstm'
CALL lstm_cell(inp${layer_dict[layer]}$, hid1, cell1, lstmLayers(${layer_dict[layer]}$)%whh, lstmLayers(${layer_dict[layer]}$)%wih, lstmLayers(${layer_dict[layer]}$)%bih, lstmLayers(${layer_dict[layer]}$)%bhh, out, cellout)
#: endif
$: layer_dict.update([(layer,layer_dict[layer]+1)])
#: endfor

