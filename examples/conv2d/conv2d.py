"""Simple Convolutional Neural Network example.
"""

import numpy as np
import os.path
import sys

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

inputs = np.array([[[[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,1,1,1,0,0,0,1,1,1,0,0],
                     [0,0,0,1,1,1,0,0,0,1,1,1,0,0],
                     [0,0,0,1,1,1,0,0,0,1,1,1,0,0],
                     [0,0,0,1,1,1,0,0,0,1,1,1,0,0],
                     [0,0,0,1,1,1,0,0,0,1,1,1,0,0],
                     [0,0,0,1,1,1,0,0,0,1,1,1,0,0],
                     [0,0,0,1,1,1,1,1,1,1,1,1,0,0],
                     [0,0,0,1,1,1,1,1,1,1,1,1,0,0],
                     [0,0,0,1,1,1,1,1,1,1,1,1,0,0],                     
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0]]]])

# Edge detection kernel
kernel = np.array([[[ 1, 1, 1],                    
                    [-1,-1,-1],
                    [ 0, 0, 0]],

                   [[ 0, 0, 0],
                    [-1,-1,-1],
                    [ 1, 1, 1]],

                   [[ 0,-1, 1],                   
                    [ 0,-1, 1],
                    [ 0,-1, 1]],

                   [[ 1,-1, 0],                   
                    [ 1,-1, 0],
                    [ 1,-1, 0]]], dtype=np.float32)


# Output used in training

# outputs = np.array([[[[0,0,0,0,0,0,0,0],
#                       [0,0,0,0,0,0,0,0],
#                       [0,0,0,0,0,0,0,0],
#                       [1,1,1,0,0,0,0,0],
#                       [0,0,0,0,0,0,0,0],
#                       [0,0,0,0,0,0,0,0],
#                       [0,0,0,0,0,0,0,0],
#                       [0,0,0,0,0,0,0,0]],

#                      [[0,0,0,1,0,0,0,0],
#                       [0,0,0,1,0,0,0,0],
#                       [0,0,0,1,0,0,0,0],
#                       [0,0,0,1,0,0,0,0],
#                       [0,0,0,0,0,0,0,0],
#                       [0,0,0,0,0,0,0,0],
#                       [0,0,0,0,0,0,0,0],
#                       [0,0,0,1,0,0,0,0]]
#                       ]])

# Create model
my_model = models.Sequential()
my_model.add(layers.Reshape((1, *inputs.shape[-2:])))
my_model.add(layers.Conv2D(4, kernel_size=(3, 3), input_shape=(1, *inputs.shape[-2:]), strides=(1, 1), kernel=kernel, padding='same', activation=activations.identity))
# my_model.add(layers.Conv2D(2, kernel_size=(3, 3), input_shape=(4, *inputs.shape[-2:]), strides=(1, 1), padding='same', activation=activations.leaky_relu))

my_model.compile()
my_model.description()

# my_model.train(inputs, outputs, optimizer='SGD', epochs=2000, learning_rate=0.01)


output = my_model.forward_pass(np.squeeze(inputs, axis=0))

mask = output<0
output[mask] = 0

output = np.sum(output, axis=0)

f, axes = plt.subplots(1,2, figsize=(12, 6))
f.suptitle("Conv2D detecting black and white outline", fontsize=18)
axes[0].imshow(inputs.squeeze((0,1)), cmap='gray')
axes[0].set_title("Input")
axes[1].imshow(output, cmap='gray')
axes[1].set_title("Output")
plt.show()

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