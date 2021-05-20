"""Playing around with matrix manipulation used in Conv2D back propagation.
"""

import numpy as np
import matplotlib.pyplot as plt


print("Created array:")
my_shape = np.array([5,3,2,2])
num_items = np.prod(my_shape)
arr = np.arange(num_items).reshape(my_shape)
print(arr)
print(arr.shape)

print("180 rotate (two flips):")
arr = np.flip(np.flip(arr, -2), -1)
print(arr)
print(arr.shape)

print("Swap axes:")
arr = np.swapaxes(arr, 0,1)
print(arr)
print(arr.shape)

num_f, num_c, num_x, num_y = arr.shape

padding = np.array([2,2])

with_padding = np.zeros(num_f, num_c, num_x + padding[0], num_y + padding[1])
print(with_padding)

for x in range(num_x):
    for y in range(num_y):
        print(arr[:,:,x,y])