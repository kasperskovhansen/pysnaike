import numpy as np


arr_1 = np.arange(36)
arr_1 = arr_1.reshape((6, 6))[np.newaxis]

arr_2 = np.arange(1,2)[np.newaxis, np.newaxis]
print(arr_2)
arr_3 = np.repeat(arr_2, 3, axis=1)
arr_3 = np.repeat(arr_3, 3, axis=2)
print(arr_3)

print(arr_1)

arr_1 -= arr_2
print(arr_1)