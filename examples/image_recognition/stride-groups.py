import numpy as np
import os.path
import sys
import math
from datetime import datetime

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

from pysnaike import activations, layers, models, callbacks, constraints

input_blocks = []
def forward_pass_optimized():
    # I     :   Input matrix
    # W     :   Weights / kernel matrix
    # b     :   Biases vector
    # Z     :   Z = I * W + b   (* is convolution)
    # z_h   :   Height of Z
    # z_w   :   Width of Z
    ### I.shape = (i_n, i_d, i_h, i_w)
    # i_n   :   Batch size (number of training examples)
    # i_d   :   Input depth
    # i_h   :   Input image height
    # i_w   :   Input image width
    ### Other
    # f_w   :   Filter width/height
    # f_n   :   Number of filters
    # s     :   OUT OF FUNCTION. Stride. Same in x and y directions.
    # p     :   Padding. Same on all sides
    ### Stride groups
    # t     :   Top index of stride group to be split into arrays with kernel size
    # b     :   Bottom index of stride group to be split into arrays with kernel size
    # l     :   Left index
    # r     :   Right index

    # Define input
    inputs_shape = np.array([1,32,28,28])

    I = np.random.random(inputs_shape)

    # Add padding to input
    p = 1
    if p > 0:
        I = np.pad(I, [(0,0), (0,0), (p,p), (p,p)])
    i_n, i_d, i_h, i_w = I.shape                        # Assigning dimension variables after padding
    
    # Define weights
    f_n = 64
    f_w = 3
    w_shape = np.array([f_n, i_d, f_w, f_w])
    W = np.random.random(w_shape) * 0.01

    # Number of stride group types
    n_rows = math.ceil(min(f_w, i_h - f_w + 1))
    n_cols = math.ceil(min(f_w, i_w - f_w + 1))

    # Z matrix
    z_h = int(((i_h - f_w)) + 1)
    z_w = int(((i_w - f_w)) + 1)
    Z = np.empty([i_n, f_n, z_h, z_w])

    # Iterating stride groups
    # input_blocks = []
    # Iterate over top indices of row stride groups ignoring stride
    for t in range(n_rows):
        input_blocks.append([])
        # Calculate bottom idx of last block in row stride group
        b = i_h - (i_h - t) % f_w
        # Matrix for current stride group row selection given current row top and bottom
        cols = np.empty([i_n, f_n, int((b - t) / f_w), z_w])

        # Iterate over left indices of column stride groups ignoring stride
        for l in range(n_cols):
            r = i_w - (i_w - l) % f_w
            # Select overall block
            block = I[:, :, t:b, l:r]
            # Split overall block into small blocks with kernel size
            block = np.array(np.split(block, (r - l) / f_w, 3))
            # An axis has been added
            block = np.array(np.split(block, (b - t) / f_w, 3))
            # Move training examples-axis up front
            block = np.moveaxis(block, 2, 0)
            # Split last
            block = np.expand_dims(block, 3)
            # Save for use in back propagation
            input_blocks[t].append(block)
            # Matrix-matrix product between input and weights
            block = block * W
            # Sum up across input channels
            block = np.sum(block, 6)
            block = np.sum(block, 5)
            block = np.sum(block, 4)
            block = np.moveaxis(block, 3, 1)
            # Save to column output matrix
            cols[:, :, :, l::n_cols] = block
        Z[:, :, t::n_rows, :] = cols

    # Z += b        # Add bias here
    A = Z           # Activation function here
    print(A.shape)
    print(datetime.now() - start_time)


def backward_pass_optimized():
    dZ = np.random.random([1, 64, 28, 28])
    inputs_shape = np.array([1,32,30,30])
    i_n, i_d, i_h, i_w = inputs_shape

    # From init
    f_n = 64 
    p = 1
    f_w = 3   
    W = np.random.random([64, 32, 3, 3])
    

    # From forward pass
    # Number of stride group types
    n_rows = math.ceil(min(f_w, i_h - f_w + 1))
    n_cols = math.ceil(min(f_w, i_w - f_w + 1))

    dA_prev = np.zeros([i_n, i_d, i_h, i_w])
    dW = np.zeros([f_n, i_d, f_w, f_w])

    for t in range(n_rows):
        row = dZ[:, :, t::n_rows, :]
        b = (i_h - t) % f_w       # Cells below. Different from forward pass
        for l in range(n_cols):
            r = (i_w - l) % f_w
            block = row[:, :, :, l::n_cols]
            block = np.moveaxis(block, 1, 3)
            block = np.expand_dims(block, 4)
            block = np.expand_dims(block, 4)
            block = np.expand_dims(block, 4)
            dW_block = block * input_blocks[t][l]

            dW_block = np.sum(dW_block, 2)
            dW_block = np.sum(dW_block, 1)
            dW_block = np.sum(dW_block, 0)
            dW += dW_block

            dA_prev_block = block * W
            dA_prev_block = np.sum(dA_prev_block, 3)
            dA_prev_block = np.reshape(dA_prev_block, (i_n, i_d, i_h - b - t, i_w - r - l))            
            dA_prev[:, :, t:i_h - b, l:i_w - r] += dA_prev_block            
        
    if p > 0:
        dA_prev = dA_prev[:, :, p:-p, p:-p]          
    print(datetime.now() - start_time)
    breakpoint()

start_time = datetime.now()
forward_pass_optimized()
start_time = datetime.now()
backward_pass_optimized()











