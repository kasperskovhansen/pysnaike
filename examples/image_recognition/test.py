import numpy as np
import os.path
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

from pysnaike import activations, layers, models, callbacks, constraints
from datetime import datetime

d = 32
a_size = np.array([d,28,28])
a = np.arange(0,np.prod(a_size)).reshape(a_size)
a_pad = np.zeros([a_size[0], a_size[1] + 2, a_size[2] + 2])
a_pad[:,1:1 + a_size[1], 1:1 + a_size[2]] = a
b_size = np.array([64,d,3,3])
b = np.arange(0,np.prod(b_size)).reshape(b_size)
print(a)
print()
print(b)
i0 = np.repeat(np.arange(b_size[2]), b_size[2])
i1 = np.repeat(np.arange(a_size[2]), a_size[2])
i = i0.reshape(-1,1) + i1.reshape(1, -1)
j0 = np.tile(np.arange(b_size[3]), b_size[3])
j1 = np.tile(np.arange(a_size[2]), a_size[2])
j = j0.reshape(-1,1) + j1.reshape(1, -1)
k = np.repeat(np.arange(d), np.prod(b_size[2:])).reshape(-1,1)


select_img = a_pad[:, i, j]
print(select_img.shape)
weights = b.reshape(b_size[0], b_size[1], b_size[2] * b_size[3])
start_time = datetime.now()
dot_product = np.tensordot(weights[:], select_img)
print(f"time {datetime.now() - start_time}")


# d = 3
# print("depth")
# print(d)

# a_size = np.array([3,4,4])
# a = np.arange(0,np.prod(a_size)).reshape(a_size)
# print("a")
# print(a)
# a_pad = np.zeros([a_size[0], a_size[1] + 2, a_size[2] + 2])
# print(a_pad.shape)
# a_pad[:,1:1 + a_size[1], 1:1 + a_size[2]] = a
# print(a_pad)

# b_size = np.array([5,d,3,3])
# b = np.arange(0,np.prod(b_size)).reshape(b_size)
# print("b")
# print(b)

# i0 = np.repeat(np.arange(b_size[2]), b_size[2])
# i1 = np.repeat(np.arange(a_size[2]), a_size[2])
# print(i0)
# print(i0.reshape(-1,1))
# print(i1)
# print(i1.reshape(1,-1))
# print()
# i = i0.reshape(-1,1) + i1.reshape(1, -1)
# print(i)

# print()
# j0 = np.tile(np.arange(b_size[3]), b_size[3])
# print(j0)
# print(j0.reshape(-1,1))
# j1 = np.tile(np.arange(a_size[2]), a_size[2])
# print(j1)
# print(j1.reshape(1,-1))
# j = j0.reshape(-1,1) + j1.reshape(1, -1)
# print()
# print(j)
# print()
# k = np.repeat(np.arange(d), np.prod(b_size[2:])).reshape(-1,1)
# print("k")
# print(k)

# select_img = a_pad[:, i, j]
# print(select_img)

# weights = b.reshape(b_size[0], b_size[1], b_size[2] * b_size[3])
# print(weights)
# start_time = datetime.now()
# print(np.tensordot(weights[:], select_img))
# print(f"time {datetime.now() - start_time}")
# start_time = datetime.now()
# # print(np.dot(weights.transpose(), select_img))
# print(select_img.shape)
# print(np.dot(weights.transpose(), select_img))
# print(f"time {datetime.now() - start_time}")