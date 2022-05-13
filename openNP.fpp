#: set a = ['conv','maxpool','conv','maxpool', 'linear', 'linear', 'linear', 'linear', 'lstm']
#: set vars = "inp"
#: set layer_dict = {}
#: for layer in a
#: mute
#: if layer not in layer_dict
$: layer_dict.update([(layer,1)])
#: endif
#: endmute
#!Linear Layer
#: if layer == 'linear'
CALL linear_layer(${vars}$, linLayers(${layer_dict[layer]}$))
#!LSTM Layer
#: elif layer == 'lstm'
CALL lstm(${vars}$, hid1, cell1, lstmLayers(${layer_dict[layer]}$)%whh, lstmLayers(${layer_dict[layer]}$)%wih, lstmLayers(${layer_dict[layer]}$)%bih, lstmLayers(${layer_dict[layer]}$)%bhh)
#: mute
#: set vars = "hid1"
#: endmute
#!Convolutional Layer
#: elif layer == 'conv'
CALL conv(${vars}$, convLayers(${layer_dict[layer]}$)%weights, convLayers(${layer_dict[layer]}$)%biases)
#!Max Pooling Layer
#: elif layer == 'maxpool'
CALL max_pool(${vars}$,maxpoolLayers(${layer_dict[layer]}$))
#: endif
#: mute
$: layer_dict.update([(layer,layer_dict[layer]+1)])
#: endmute
#: endfor

