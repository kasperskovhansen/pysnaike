import os.path
import sys
import time

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from pysnaike import callbacks


# Callback functions are created
def batch_callback(**kvargs):
    for i in kvargs:
        print(f"{i}: {kvargs[i]}")
    print()

def training_end(**kvargs):
    if 'total_batches' in kvargs:
        print(f"Total batches: {kvargs['total_batches']}")

# A Callbacks instance is setup to execute the functions declared above
# on events 'batch' and 'end'
myCallbacks = callbacks.Callbacks()
myCallbacks.on('batch', batch_callback)
myCallbacks.on('last_batch', training_end)


# Time has come to execute some callback functions in a loop
num_batches = 10
for batch in range(1, num_batches + 1):
    myCallbacks.execute('batch', curr_batch=batch, score=10 - 10 / batch)

    if batch == num_batches:
        myCallbacks.execute('last_batch', total_batches=batch)