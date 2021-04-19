import numpy as np


def add_padding(input, padding):    
    out = np.zeros((input.shape[0], *(input.shape[1:] + padding * 2)))       
    out[:, padding[0] : padding[0] + input.shape[1], padding[1]:padding[1] + input.shape[2]] = input
    return out



channels, in_x, in_y = 2, 4, 4
x_pad, y_pad = 1, 1

input = np.arange(channels * in_x * in_y)
input = input.reshape((channels, in_x, in_y))
print(input)
input = add_padding(input, np.array((x_pad, y_pad)))

filters, x_w, y_w = 3, 3, 3
w = np.arange(filters * channels * x_w * y_w)
w = w.reshape((filters, channels, x_w, y_w))

out = np.zeros((filters, in_x, in_y))
x_out, y_out = in_x, in_y
bias = np.zeros((x_out, y_out)) + 1

print(input)
print(w)
print(out)
print(bias)

for f in range(filters):
    for x in range(x_out):
        for y in range(y_out):        
            # print(input[:, x:x + x_w, y:y + y_w].shape)
            # print(w[0].shape)
            dot = np.dot(input[:, x:x + x_w, y:y + y_w].flatten(), w[f].flatten())
            out[f,x,y] = dot

print('Tensordot:')
print(out)
print('add bias:')
out += bias
print(out)