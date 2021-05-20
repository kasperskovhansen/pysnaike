"""Simple Convolutional Neural Network example.
"""

import numpy as np
import os.path
import sys
from os import path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

from pysnaike import activations, layers, models
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# Sample input
# Load grayscale image as input
# img = Image.open('images/smiley.png').convert('L')
# print(img)
# print(img.size)
# print(img.mode)
# inputs = np.asarray(img)[np.newaxis, np.newaxis, :]

# Load grayscale image as input
# convert('L')

in_path = 'images/colored_squares_36-36.png'
target_path = 'images/colored_squares_12-12_target.png'
mark = 0
if mark:
    in_path = 'images/colored_squares_36-36_m.png'
    target_path = 'images/colored_squares_36-36_target_m.png'    

img = Image.open(in_path).convert('RGB')

inputs = np.asarray(img)
inputs = np.transpose(inputs, axes=(1,0,2))
inputs = inputs.swapaxes(0,-1)[np.newaxis]

img_target = Image.open(target_path).convert('L')
targets = np.asarray(img_target)
targets = np.transpose(targets, axes=(1,0))
targets = targets.swapaxes(0,-1)[np.newaxis, np.newaxis]

# Create model
M = models.Sequential()
M.add(layers.Conv2D(16, kernel_size=(5, 5), input_shape=(inputs.shape[-3:]), padding='same', activation=activations.leaky_relu))
M.add(layers.MaxPooling2D((2,2)))
M.add(layers.Conv2D(32, kernel_size=(3, 3), input_shape=(16, 18, 18), padding='same', activation=activations.leaky_relu))
M.add(layers.MaxPooling2D((2,2)))
M.add(layers.Flatten())
M.add(layers.Dense(144, activation=activations.relu))
M.add(layers.Reshape((1,12,12)))

M.compile()
M.description()

# # Load trained network from .npz file
# network_path = 'network_params.npz'
# if path.exists(network_path):
#     print('Loading params from external file...')
#     params = np.load(network_path)
#     for key in params.files:
#         M.params[key] = params[key]

M.train(inputs, targets, optimizer='SGD', epochs=1, learning_rate=0.0001)
# # Save the network params to disk
# np.savez('network_params.npz', **M.params)
# print('Done saving.')

output = M.forward_pass(np.squeeze(inputs, axis=0))[np.newaxis]

mask = output<0
output[mask] = 0
output = output

inputs = inputs.squeeze(0)
fig = plt.figure(figsize=(12, 6))
ax = {}

plt_y = max(inputs.shape[0], output.shape[0]) + 1

ax = plt.subplot2grid((3, plt_y), (0, 0), rowspan=1, colspan=1)
ax.imshow( np.transpose(inputs[0:3].swapaxes(0,-1), axes=(1,0,2)) )
ax.set_title(f"Input channel sum")
# input channels
for c in range(inputs.shape[0]):
    ax = plt.subplot2grid((3, plt_y), (0, c + 1), rowspan=1, colspan=1)
    if c == 0: col = 'Reds'
    elif c == 1: col = 'Greens'
    elif c == 2: col = 'Blues'
    ax.imshow(inputs[c], cmap=col)
    ax.set_title(f"Input channel {c}")

ax = plt.subplot2grid((3, plt_y), (1, 0), rowspan=1, colspan=1)
ax.imshow(np.transpose(targets[0].swapaxes(0,-1), axes=(1,0,2)), cmap="gray")
ax.set_title(f"Targets")

for c in range(output.shape[0]):
    ax = plt.subplot2grid((3, plt_y), (1, c + 1), rowspan=1, colspan=1)
    ax.imshow(np.transpose(output[c].swapaxes(0,-1), axes=(1,0,2)), cmap='gray')
    ax.set_title(f"Output channel {c}")

test_in_path = "images/test_input.png"
test_img = Image.open(test_in_path).convert('RGB')
test_inputs = np.asarray(test_img)
test_inputs = np.transpose(test_inputs, axes=(1,0,2))
test_inputs = test_inputs.swapaxes(0,-1)[np.newaxis]
test_out = M.forward_pass(np.squeeze(test_inputs, axis=0))[np.newaxis]
test_out = test_out.squeeze(0)
test_inputs = test_inputs.squeeze(0)

ax = plt.subplot2grid((3, plt_y), (2, 0), rowspan=1, colspan=1)
ax.imshow( np.transpose(test_inputs[0:3].swapaxes(0,-1), axes=(1,0,2)) )
ax.set_title(f"Test input channel sum")

ax = plt.subplot2grid((3, plt_y), (2, 1), rowspan=1, colspan=1)
ax.imshow( np.transpose(test_out[0:3].swapaxes(0,-1), axes=(1,0,2)) )
ax.set_title(f"Test output channel sum")
plt.show()