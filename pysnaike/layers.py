"""Different types of layers in a network."""


import numpy as np
from pysnaike import activations
import sys


class Dense():
    def __init__(self, size: int, name: str = "dense", activation=activations.relu):
        """Dense layer.

        Args:
            size (int): Number of perceptrons in the layer.
            name (str, optional): Name of the input layer.
            activation (optional): Activation function used at this layer.
        """

        self.size = size
        self.name = name
        self.activation = activation

        self.num_params = self.size * 2
        self.output_shape = self.size
        self.layer_idx = None


    def setup(self, **kvargs):
        """Called on model compile.

        Returns:
            (w, b): Weight and bias.
        """

        self.layer_idx = kvargs['layer_idx']
        w = np.random.randn(kvargs['size_curr'], kvargs['size_prev']) * 0.1
        b = np.zeros(kvargs['size_curr'])
        return w, b

    def forward_pass(self, params):
        """Forward pass through dense layer.

        Args:
            params (dict): Model parameters.
        """
        w = params[f'W{self.layer_idx}']
        b = params[f'B{self.layer_idx}']
        prev_a = params[f'A{self.layer_idx - 1}']

        # Referenced object `params` is modified
        params[f'Z{self.layer_idx}'] = np.dot(w, prev_a) + b
        params[f'A{self.layer_idx}'] = self.activation(params[f'Z{self.layer_idx}'])

    def backward_pass(self, error, is_last=False, output=None, target=None, new_params=None, model=None, **kvargs):

        if not is_last:
            error = np.dot(model.params['W' + str(self.layer_idx + 1)].T, error) * model.layers[self.layer_idx].activation(model.params['Z' + str(self.layer_idx)], derivative=True)
        else:
            error = 2 * (output - target) / output.shape[0] * model.layers[self.layer_idx].activation(model.params['Z' + str(self.layer_idx)], derivative=True)

        new_params['W' + str(self.layer_idx)] = np.outer(error, model.params['A' + str(self.layer_idx - 1)])
        new_params['B' + str(self.layer_idx)] = error
        return error


    def __str__(self) -> str:
        return f"<Dense layer class object named '{self.name}' of size {self.size} using activation function '{self.activation.__name__}'>"


class Conv2D():
    def __init__(self, num_filters, kernel_size=(3, 3), input_shape=(1, 5, 5), strides=(1,1), padding="same", activation=activations.identity, name: str = "conv2d"):
        """2D convolutional layer.

        Not sure activations are working properly.

        Args:
            num_filters (int): Number of filters in layer.
            kernel_size ((x, y), optional): Size of kernel Both dimensions must be odd. Defaults to (3, 3).
            input_shape ((channels, x, y), optional): Shape of input data. Defaults to (2, 5, 5).
            strides ((x, y), optional): NOT implemented. Step size when sliding across the grid. Defaults to (1, 1).
            padding (optional): Either "same", "valid" or an integer. "same" means padding is added evenly on all sides, so that output shape equals input shape. "valid" is the same as no padding. Defaults to "same".
            activation (optional): Activation function used at this layer.
            name (str, optional): Name of the layer. Defaults to "conv2d".

        Todo:
            Stride
        """

        self.num_filters = num_filters
        self.kernel_size = np.array(kernel_size)
        self.input_shape = np.array(input_shape)
        self.strides = np.array(strides)
        self.padding = self.calc_padding(padding)
        self.activation = activation
        self.name = name

        self.layer_idx = 0 # Changed at model compile
        self.num_params = self.calc_num_params()
        self.output_shape = self.calc_output_shape()

    def setup(self, **kvargs):
        self.layer_idx = kvargs['layer_idx']

        w = np.random.randn(self.num_filters, self.input_shape[0], *self.kernel_size)
        b = np.arange(self.num_filters)

        return w, b

    def calc_padding(self, padding):
        """Calculate padding size left and right in x-direction and up and down in y-direction.

        Args:
            padding: [description]

        Returns:
            [type]: [description]
        """

        pad = np.array([0,0])

        if padding == 'same':
            # Padding is added equally on all sides. Output grid has same size as input.
            pad = (self.kernel_size - 1) // 2

        return pad

    def calc_num_params(self):
        """Calculate number of tweakable weights and biases in layer.

        Output as calculated with the formula:
        Filter size prod (m x n) * num channels * num filters + bias (which is num filters)

        Returns:
            int: Number of parameters.
        """

        return np.prod(self.kernel_size) * self.input_shape[0] * self.num_filters + self.num_filters

    def calc_output_shape(self):
        """Output shape of convolutional layer.

        Returns:
            [out_x, out_y]: Numpy array with layer output size x and output size y.
        """
        return np.append(self.num_filters, [(self.input_shape[1:] + self.padding * 2 - self.kernel_size) // self.strides + 1])


    def forward_pass(self, params):
        w = params[f'W{self.layer_idx}']
        b = params[f'B{self.layer_idx}']
        prev_a = params[f'A{self.layer_idx - 1}']

        # Adding 2 dimensions to b using np.newaxis
        params[f'Z{self.layer_idx}'] = self.convolve(w, prev_a) + b[:, np.newaxis, np.newaxis]
        params[f'A{self.layer_idx}'] = self.activation(params[f'Z{self.layer_idx}'])

    def convolve(self, w, a):
        # Add padding to activations       
        with_pad = np.zeros((a.shape[0], *(a.shape[1:] + self.padding * 2)))
        with_pad[:, self.padding[0] : self.padding[0] + a.shape[1], self.padding[1]:self.padding[1] + a.shape[2]] = a
        out = np.zeros((self.num_filters, *self.output_shape[-2:]))                
        for f in range(self.output_shape[0]):
            for x in range(self.output_shape[1]):
                for y in range(self.output_shape[2]):                    
                    dot = np.dot(with_pad[:, x:x + self.kernel_size[0], y:y + self.kernel_size[1]].flatten(), w[f].flatten())
                    out[f,x,y] = dot        
        return out

    def add_padding(input, padding):
        out = np.zeros((input.shape[0], *(input.shape[1:] + padding * 2)))
        out[:, padding[0] : padding[0] + input.shape[1], padding[1]:padding[1] + input.shape[2]] = input
        return out

    def backward_pass():
        pass

    def __str__(self) -> str:
        return f"<Conv2D layer class object named '{self.name}' with input shape {self.input_shape} and filter shape '{self.kernel_size}'>"


class MaxPooling2D():
    def __init__(self, pool_size):
        pass


class AvgPooling2D():
    def __init__(self, pool_size):
        pass


class Flatten():
    def __init__(self, name='flatten'):
        self.name = name
        self.output_shape = None
        self.num_params = None

    def setup(self, **kvargs):
        self.layer_idx = kvargs['layer_idx']

        if kvargs['size_prev'].shape is not ():
            self.output_shape = np.prod(kvargs['size_prev'][0:-1])
        else: self.output_shape = kvargs['size_prev']
        return None, None

    def forward_pass(self, params):
        pass
        # params[f'Z{self.layer_idx}'] = params[f'Z{self.layer_idx - 1}']
        # params[f'A{self.layer_idx}'] = params[f'A{self.layer_idx - 1}']





class Reshape():
    def __init__(self, output_shape, name='reshape'):
        self.name = name
        self.output_shape = np.array(output_shape)
        self.num_params = None
        self.layer_idx = None
        self.activation = None

    def setup(self, **kvargs):
        self.layer_idx = kvargs['layer_idx']
        assert np.prod(kvargs['size_prev']) == np.prod(self.output_shape), f'Reshape not possible from {kvargs["size_prev"]} to {self.output_shape}'

        return None, None

    def forward_pass(self, params):
        params[f'Z{self.layer_idx}'] = None
        params[f'A{self.layer_idx}'] = None
