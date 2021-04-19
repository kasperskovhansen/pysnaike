"""Simple Deep Neural Network example.
"""

import os.path
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

import numpy as np
from pysnaike import activations, layers, models


# Create dataset.
num_in = 10
options = 10
inputs = np.random.randint(0, options + 1, size=[num_in, 10000])
outputs = (np.sum(inputs, axis=0)) / (inputs.shape[0] * options)
inputs = inputs.T
outputs = outputs.T

# print(inputs)
# print(outputs)

# Create model

my_model = models.Sequential()

my_model.add(layers.Dense(num_in, activation=activations.input))
my_model.add(layers.Dense(10, activation=activations.relu))
my_model.add(layers.Dense(6, activation=activations.softmax))
my_model.add(layers.Dense(1, activation=activations.sigmoid))

my_model.compile()
my_model.description()

my_model.train(inputs, outputs, optimizer='SGD', epochs=4, learning_rate=0.001)

# Create test data
inputs = np.random.randint(0, options + 1, size=[num_in, 100])
outputs = (np.sum(inputs, axis=0)) / (inputs.shape[0] * options)
inputs = inputs.T
outputs = outputs.T

# Compare model output to test data output
for i in range(0, 100):
    test_input = inputs[i]
    test_output = outputs[i]

    output = my_model.forward_pass(test_input)
    print(f"test_input {test_input} test_output {test_output}, output {output}")
