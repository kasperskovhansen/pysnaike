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


inputs = np.array([[[[0,0,0,0,0,1],
                     [0,0,0,0,0,1],
                     [0,0,0,0,0,1],
                     [0,0,0,0,0,1],
                     [1,1,1,1,1,1],
                     [1,1,1,1,1,1]]]])


# target used in training
targets = np.array([[[[0,0,0,0,1,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0]],

                     [[0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [1,1,1,1,1,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0]],
                      
                     [[0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [0,0,0,0,0,0],
                      [1,1,1,1,1,0],
                      [0,0,0,0,0,0]]]])


# Create model
M = models.Sequential()
M.add(layers.Conv2D(3, kernel_size=(3, 3), input_shape=(inputs.shape[-3:]), padding='same', activation=activations.identity))

M.compile()
M.description()

M.train(inputs, targets, optimizer='SGD', epochs=100000, learning_rate=0.001)

# test model on new input
output = M.forward_pass(np.squeeze(inputs, axis=0))

# Preparing for plot
fig = plt.figure(figsize=(12, 6))
inputs = inputs.squeeze(0)
targets = targets.squeeze(0)

# plot size determined by max number of channels
plt_y = max(inputs.shape[0], output.shape[0]) + 1

# plot inputs as an image
ax = plt.subplot2grid((3, plt_y), (0, 0), rowspan=1, colspan=1)
ax.imshow( np.transpose(inputs.swapaxes(0,-1), axes=(1,0,2)) )

# plot input channels
for c in range(inputs.shape[0]):
    ax = plt.subplot2grid((3, plt_y), (0, c + 1), rowspan=1, colspan=1)
    if c == 0: col = 'Reds'
    elif c == 1: col = 'Greens'
    elif c == 2: col = 'Blues'
    ax.imshow(inputs[c], cmap=col)
    ax.set_title(f"Input channel {c}")

# plot target as an image
ax = plt.subplot2grid((3, plt_y), (1, 0), rowspan=1, colspan=1)
ax.imshow(np.transpose(targets.swapaxes(0,-1), axes=(1,0,2)), cmap="gray")
ax.set_title(f"Targets")

# plot output channels
for c in range(output.shape[0]):
    ax = plt.subplot2grid((3, plt_y), (1, c + 1), rowspan=1, colspan=1)
    ax.imshow(np.transpose(output[c].swapaxes(-1,-2)), cmap='gray')
    ax.set_title(f"Output channel {c}")


new_inputs = np.array([[[[0,0,0,0,0,0],
                         [0,0,0,1,1,0],
                         [0,0,0,1,1,0],
                         [0,0,0,1,1,1],
                         [0,1,1,1,1,1],
                         [0,0,0,0,0,0]]]])

new_output = M.forward_pass(np.squeeze(new_inputs, axis=0))

new_inputs = new_inputs.squeeze(0)

# plot new input as an image
ax = plt.subplot2grid((3, plt_y), (2, 0), rowspan=1, colspan=1)
ax.imshow(np.transpose(new_inputs.swapaxes(0,-1), axes=(1,0,2)), cmap="gray")
ax.set_title(f"New input")

# plot output channels
for c in range(new_output.shape[0]):
    ax = plt.subplot2grid((3, plt_y), (2, c + 1), rowspan=1, colspan=1)
    ax.imshow(np.transpose(new_output[c].swapaxes(-1,-2)), cmap='gray')
    ax.set_title(f"New output channel {c}")

plt.show()
