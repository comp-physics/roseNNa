
CALL lstm_cell(inp1, hid1, cell1, lstmLayers(1)%whh, lstmLayers(1)%wih, lstmLayers(1)%bih, lstmLayers(1)%bhh, out, cellout)


out = linear_layer(out, layers(1)%weights, layers(1)%biases)
out = layers(1)%fn_ptr(out)

out = linear_layer(out, layers(2)%weights, layers(2)%biases)
out = layers(2)%fn_ptr(out)

out = linear_layer(out, layers(3)%weights, layers(3)%biases)
out = layers(3)%fn_ptr(out)

out = linear_layer(out, layers(4)%weights, layers(4)%biases)
out = layers(4)%fn_ptr(out)


