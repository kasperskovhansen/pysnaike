import numpy as np
import os.path
import sys
import time

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

from pysnaike import activations, layers, models


# Sample input
inputs = np.array([[0, 0, 1, 0, 0], 
                   [0, 0, 1, 0, 0], 
                   [0, 0, 1, 0, 0], 
                   [0, 0, 1, 0, 0], 
                   [0, 0, 1, 0, 0]])

# print(inputs.size)

# Create model
my_model = models.Sequential()

my_model.add(layers.Dense(25, activation=input))
# my_model.add(layers.Reshape((5, 5, 1)))
my_model.add(layers.Conv2D(3, kernel_size=(3, 3), input_shape=(2, 5, 5), strides=(1, 1), padding='same'))
# my_model.add(layers.Flatten())

# my_model.add(layers.Dense(5))
# my_model.add(layers.Dense(9))
# my_model.add(layers.Dense(9))
# my_model.add(layers.Reshape((3, 3, 1)))
# my_model.add(layers.Conv2D(1, kernel_size=(3, 3), input_shape=(5, 5, 1), strides=(1, 1), padding='same'))
# my_model.add(layers.Conv2D(1, kernel_size=(3, 3), input_shape=(5, 5, 1), strides=(1, 1), padding='same'))
# my_model.add(layers.Flatten())
# my_model.add(layers.Flatten())
# my_model.add(layers.Reshape((5, 5, 1)))
# my_model.add(layers.Flatten())
# my_model.add(layers.Dense(25))
my_model.compile()
my_model.description()

# print(my_model.layers[0].num_params)
# print(my_model.layers[0].output_shape)

output = my_model.forward_pass(np.reshape(np.arange(50), (2,5,5) ))
print(f'output: {output}')



