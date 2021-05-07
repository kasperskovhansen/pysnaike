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

# inputs = np.array([[[[1,1,0,0,0,0,1,1,1,1,1,0,0,0],
#                      [1,1,0,0,0,0,1,1,1,1,1,0,0,0],
#                      [1,1,0,0,0,0,1,1,1,1,1,0,0,0],
#                      [1,1,0,0,0,0,1,1,1,1,1,0,0,0],
#                      [1,1,0,0,0,0,1,1,1,1,1,0,0,0],
#                      [1,1,0,0,0,0,1,1,1,1,1,1,1,1],
#                      [1,1,0,0,0,0,1,1,1,1,1,1,1,1],
#                      [0,0,0,0,0,0,1,1,1,1,1,1,1,1],
#                      [0,0,0,0,0,0,1,1,1,1,1,1,1,1],
#                      [0,0,0,1,1,1,1,1,1,1,0,0,1,1],
#                      [0,0,0,1,1,1,1,1,1,1,0,0,1,1],
#                      [0,0,0,1,1,1,1,1,1,1,0,0,1,1],
#                      [0,0,0,1,1,1,1,1,1,1,0,0,1,1],
#                      [0,0,0,1,1,1,1,1,1,1,0,0,1,1]]]])

# inputs = np.array([[[[0,0,0,0,1,1,1,1],
#                      [0,0,0,0,1,1,1,1],
#                      [0,0,0,0,1,1,1,1],
#                      [0,0,0,0,1,1,1,1],
#                      [1,1,1,1,1,1,1,1],
#                      [1,1,1,1,1,1,1,1],                      
#                      [1,1,1,1,1,1,1,1],
#                      [0,0,0,0,1,1,1,1]]]])

# inputs = np.array([[[[0,0,0,0,0,0,1,1,1,0,0,0,0,0],
#                      [0,0,0,0,0,0,1,1,1,0,0,0,0,0],
#                      [0,0,0,1,1,1,1,1,1,0,0,0,0,0],
#                      [0,0,0,1,1,1,1,1,1,0,0,0,0,0],
#                      [0,0,0,1,1,1,1,1,1,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],                     
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                    
#                     [[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [1,1,1,0,0,0,0,0,0,1,1,1,1,1],
#                      [1,1,1,0,0,0,0,0,0,1,1,1,1,1],
#                      [1,1,1,0,0,0,0,0,0,1,1,1,1,1],
#                      [1,1,1,0,0,0,0,0,0,1,1,1,0,0],
#                      [1,1,1,0,0,0,0,0,0,1,1,1,0,0],
#                      [1,1,1,0,0,0,0,0,0,1,1,1,0,0],
#                      [1,1,1,0,0,0,0,0,0,1,1,1,0,0],
#                      [1,1,1,0,0,0,0,0,0,1,1,1,0,0],
#                      [1,1,1,0,0,0,0,0,0,1,1,1,0,0]],                    
#                      ]])


# [[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],                     
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0]]


inputs = np.array([[[[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,1,1,1,0,0,0,0,0,0,0,0],
                     [0,0,0,1,0,1,0,0,0,0,0,0,0,0],
                     [0,0,0,1,1,1,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,1,1,1,0,0,0],
                     [0,1,1,1,0,0,0,0,1,0,1,0,0,0],
                     [0,1,0,1,0,0,0,0,1,1,1,0,0,0],                     
                     [0,1,1,1,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                    
                    [[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0]],                    
                    
                    [[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0]],                    
                     ]])

# Edge detection kernel
kernel = np.array([
                  [[[ 1, 1, 1],                    
                    [-1,-1,-1],
                    [ 0, 0, 0]],

                    [[ 1, 1, 1],                    
                    [-1,-1,-1],
                    [ 0, 0, 0]],

                   [[ 1, 1, 1],                    
                    [-1,-1,-1],
                    [ 0, 0, 0]]],


                  [[[ 0, 0, 0],
                    [-1,-1,-1],
                    [ 1, 1, 1]],

                    [[ 0, 0, 0],
                    [-1,-1,-1],
                    [ 1, 1, 1]],

                   [[ 0, 0, 0],
                    [-1,-1,-1],
                    [ 1, 1, 1]]],


                  [[[ 0,-1, 1],                   
                    [ 0,-1, 1],
                    [ 0,-1, 1]],

                    [[ 0,-1, 1],                   
                    [ 0,-1, 1],
                    [ 0,-1, 1]],

                   [[ 0,-1, 1],                   
                    [ 0,-1, 1],
                    [ 0,-1, 1]]],


                  [[[ 1,-1, 0],                   
                    [ 1,-1, 0],
                    [ 1,-1, 0]],

                    [[ 1,-1, 0],                   
                    [ 1,-1, 0],
                    [ 1,-1, 0]],

                   [[ 1,-1, 0],                   
                    [ 1,-1, 0],
                    [ 1,-1, 0]]]], dtype=np.float32)


# Output used in training

outputs = np.array([[[[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,1,1,1,0,0,0,0,0,0,0,0],
                      [0,0,0,1,1,1,0,0,0,0,0,0,0,0],
                      [0,0,0,1,1,1,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,1,1,1,0,0,0],
                      [0,0,0,0,0,0,0,0,1,1,1,0,0,0],
                      [0,0,0,0,0,0,0,0,1,1,1,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]])

# colored_squares_32-32
# Load grayscale image as input
# convert('L')

in_path = 'images/colored_squares_32-32.png'
target_path = 'images/colored_squares_32-32_target.png'
mark = 0
if mark:
    in_path = 'images/colored_squares_32-32_m.png'
    target_path = 'images/colored_squares_32-32_target_m.png'    

img = Image.open(in_path).convert('RGB')
print(img)
print(img.size)
print(img.mode)
print(inputs.shape)
inputs = np.asarray(img)
print(inputs.shape)
inputs = np.transpose(inputs, axes=(1,0,2))
print(inputs.shape)
inputs = inputs.swapaxes(0,-1)[np.newaxis]
print(inputs.shape)
# inputs = inputs[np.newaxis]
# print(inputs.shape)

# plt.imshow(inputs[0, 0], cmap='gray')
# plt.show()

img_target = Image.open(target_path).convert('L')
targets = np.asarray(img_target)
targets = np.transpose(targets, axes=(1,0))
targets = targets.swapaxes(0,-1)[np.newaxis, np.newaxis]
print("targets.shape")
print(targets.shape)

# plt.imshow(targets[0, 0], cmap='gray')
# plt.show()

in_shape = np.array((1,3,12,12))
inputs = np.random.randint(0, 10, np.prod(in_shape)).reshape(in_shape)
print("inputs")
print(inputs)

target_shape = np.array((1,9))
targets = np.random.randint(0, 10, np.prod(target_shape)).reshape(target_shape)
print("targets")
print(targets)

# Create model
my_model = models.Sequential()
# my_model.add(layers.Input(input_shape=inputs.shape[-3:])) # should not be necessary
my_model.add(layers.Conv2D(5, kernel_size=(3, 3), input_shape=(inputs.shape[-3:]), strides=(1, 1), kernel=None, padding='same', activation=activations.sigmoid))
my_model.add(layers.MaxPooling2D((2,2)))
my_model.add(layers.Conv2D(1, kernel_size=(3, 3), input_shape=(5, 6, 6), strides=(1, 1), kernel=None, padding='same', activation=activations.sigmoid))
my_model.add(layers.MaxPooling2D((2,2)))
my_model.add(layers.Reshape((9)))
my_model.add(layers.Dense(9, activation=activations.sigmoid))
my_model.add(layers.Dense(9, activation=activations.softmax))
# my_model.add(layers.Dense(6, activation=activations.sigmoid))
# # my_model.add(layers.Dense(9, activation=activations.sigmoid))
# my_model.add(layers.Dense(9, activation=activations.sigmoid))
# my_model.add(layers.MaxPooling2D((2,2)))
# my_model.add(layers.MaxPooling2D((2,2)))
# my_model.add(layers.Reshape((9)))
# my_model.add(layers.Reshape((1,1,9)))


# my_model.add(layers.Conv2D(1, kernel_size=(3, 3), input_shape=(3, *inputs.shape[-2:]), strides=(1, 1), padding='same', activation=activations.identity))
# my_model.add(layers.MaxPooling2D((2,2)))
# my_model.add(layers.Conv2D(1, kernel_size=(3, 3), input_shape=(1, *inputs.shape[-2:]), strides=(1, 1), padding='same', activation=activations.identity))


my_model.compile()
my_model.description()


# Load trained network from .npz file
# network_path = 'network_params.npz'
# if path.exists(network_path):
#     print('Loading params from external file...')
#     params = np.load(network_path)
#     for key in params.files:
#         my_model.params[key] = params[key]
print("in shape")
print(inputs.shape)

my_model.train(inputs, targets, optimizer='SGD', epochs=100, learning_rate=0.1)
# # Save the network params to disk
# np.savez('network_params.npz', **my_model.params)
# print('Done saving.')

output = my_model.forward_pass(np.squeeze(inputs, axis=0))[np.newaxis]

print("output.shape")
print(output.shape)

mask = output<0
output[mask] = 0

output = output[np.newaxis]

# output = np.sum(output, axis=0)

print("inputs.shape")
# inputs = inputs.swapaxes
print(inputs.shape)
inputs = inputs.squeeze(0)
# print(inputs.shape)
print(inputs[0].shape)



fig = plt.figure(figsize=(12, 6))
ax = {}

plt_y = max(inputs.shape[0], output.shape[0]) + 1
print(plt_y)
print(output.shape[0])

ax = plt.subplot2grid((2, plt_y), (0, 0), rowspan=1, colspan=1)
ax.imshow( np.transpose(inputs.swapaxes(0,-1), axes=(1,0,2)) )
# ax.imshow( inputs.squeeze(0) )
ax.set_title(f"Input channel sum")
# input channels
for c in range(inputs.shape[0]):
    ax = plt.subplot2grid((2, plt_y), (0, c + 1), rowspan=1, colspan=1)
    if c == 0: col = 'Reds'
    elif c == 1: col = 'Greens'
    elif c == 2: col = 'Blues'
    ax.imshow(inputs[c], cmap=col)
    ax.set_title(f"Input channel {c}")

ax = plt.subplot2grid((2, plt_y), (1, 0), rowspan=1, colspan=1)

targets = targets[np.newaxis]
ax.imshow(targets[0], cmap="gray")
ax.set_title(f"Targets")

for c in range(output.shape[0]):
    ax = plt.subplot2grid((2, plt_y), (1, c + 1), rowspan=1, colspan=1)
    ax.imshow(output[c], cmap='gray')
    ax.set_title(f"Output channel {c}")

# ax2 = plt.subplot2grid((2, 4), (0, 2), rowspan=1, colspan=2)
# ax2.imshow(inputs[1], cmap='gray')
# ax2.set_title("Input channel 2")

plt.show()

# f, axes = plt.subplots(2,2, figsize=(12, 6))
# f.suptitle("Conv2D detecting black and white outline", fontsize=18)
# axes[0,0].imshow(inputs[0], cmap='gray')
# axes[0,0].set_title("Input")
# axes[1,0].imshow(inputs[1], cmap='gray')
# axes[1,0].set_title("Input")
# axes[0,1].imshow(output[0], cmap='gray')
# axes[0,1].set_title("Output")
# axes[1,1].imshow(output[1], cmap='gray')
# plt.show()

# f, axes = plt.subplots(1,2, figsize=(12, 6))
# f.suptitle("Conv2D detecting black and white outline", fontsize=18)
# axes[0].imshow(inputs.squeeze((0,1)), cmap='gray')
# axes[0].set_title("Input")
# axes[1].imshow(output, cmap='gray')
# axes[1].set_title("Output")
# plt.show()

# f, axes = plt.subplots(2,3, figsize=(12, 6))
# f.suptitle("Conv2D detecting horizontal lines with bright left and dark right", fontsize=18)
# axes[0,0].imshow(inputs.squeeze((0, 1)), cmap='gray')
# axes[0,0].set_title("Input")
# axes[0,1].imshow(output[0], cmap='gray')
# axes[0,1].set_title("Output filter 1")
# axes[1,1].imshow(output[1], cmap='gray')
# axes[1,1].set_title("Output filter 2")
# axes[0,2].imshow(outputs[0,0], cmap='gray')
# axes[0,2].set_title("Output target 1")
# axes[1,2].imshow(outputs[0,1], cmap='gray')
# axes[1,2].set_title("Output target 2")
# axes[1,0].axis('off')
# plt.show()

# f, axes = plt.subplots(1,2, figsize=(12, 6))
# f.suptitle("Conv2D detecting horizontal lines with bright left and dark right", fontsize=18)
# axes[0].imshow(inputs.squeeze((0,1)), cmap='gray')
# axes[0].set_title("Input")
# axes[1].imshow(output.squeeze(0), cmap='gray')
# axes[1].set_title("Output")
# plt.show()