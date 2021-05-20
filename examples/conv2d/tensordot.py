"Playing around with tensordot used in Conv2D back propagation."

import numpy as np
import matplotlib.pyplot as plt


print("Created array:")
my_shape = np.array([3,7,7])
num_items = np.prod(my_shape)
a = np.arange(num_items).reshape(my_shape)
print(a)
print(a.shape)

print("Created array:")
my_shape = np.array([4,3,2,2])
num_items = np.prod(my_shape)
b = np.arange(num_items).reshape(my_shape)
print(b)
print(b.shape)

print("tensordot")
num_f, num_c, num_x, num_y = b.shape

i = 3
for x in range(num_x):
    for y in range(num_y):
        print(f"{x, y}")
        print(a[:, x : x + num_x, y: y + num_y])
        print("b")
        print(b)
        out = np.tensordot(a[:, x: x + num_x, y: y + num_y], b, axes=([0,1,2], [1,2,3]))    
        print("out")
        print(out)