"""Simple Deep Neural Network example.
"""

import os.path
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

import numpy as np
from pysnaike import activations, layers, models
import matplotlib.pyplot as plt

# Create random dataset with 10000 training examples consisting of 10 random values each.
num_in = 10
options = 10
inputs = np.random.randint(0, options + 1, size=[10000, num_in])
# The output is the average value for each training example divided by number of options.
outputs = (np.sum(inputs, axis=-1)) / (inputs.shape[-1] * options)

# Create model
M = models.Sequential()
M.add(layers.Dense(10, activation=activations.leaky_relu))
M.add(layers.Dense(20, activation=activations.leaky_relu))
M.add(layers.Dense(20, activation=activations.leaky_relu))
M.add(layers.Dense(1, activation=activations.sigmoid))
M.compile()
M.description()

M.train(inputs, outputs, optimizer='SGD', epochs=2, learning_rate=0.0001)

# Create test data
test_inputs = np.random.randint(0, options + 1, size=[20, num_in])
test_outputs = (np.sum(test_inputs, axis=-1)) / (test_inputs.shape[-1] * options)
outputs = []
# Compare model output to test data output
print("test_output vs calculated output")
for i in range(0, test_inputs.shape[0]):
    # Generate output based on test data
    output = M.forward_pass(test_inputs[i])
    outputs.append(output[0])
    print(f"{test_outputs[i]} vs {output.round(2)}")

# Create diagram
fig = plt.figure()
ax1 = fig.add_subplot(111)

x = range(len(test_inputs))
ax1.fill_between(x, test_outputs, outputs, color="grey", alpha=0.3)
ax1.scatter(x, test_outputs, s=10, c='b', marker="s", label='test_outputs')
ax1.scatter(x, outputs, s=10, c='r', marker="o", label='outputs')
plt.legend(loc='upper left');
plt.title("Distance between test_output and output for each test example after 2 epochs")
plt.xlabel("Test example idx")
plt.ylabel("Output value")
plt.show()
