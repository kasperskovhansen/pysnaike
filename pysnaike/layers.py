"""Different types of layers in a network."""


import numpy as np
import math
# np.random.seed(2)
from pysnaike import activations, constraints
from datetime import datetime

class Dense():
    def __init__(self, size: int, activation=activations.leaky_relu, kernel_constraint=constraints.no_norm(), name: str = "dense"):
        """Dense layer.

        Args:
            size (int): Number of perceptrons in the layer.
            activation (optional): Activation function used at this layer.
            kernel_constraint: Regularization of kernel weights.
            name (str, optional): Name of the dense layer.
        """

        self.size = size
        self.input_shape = np.array(self.size)
        self.kernel_constraint = kernel_constraint
        self.name = name
        self.activation = activation

        self.num_params = self.size * 2
        self.output_shape = self.input_shape
        self.layer_idx = None


    def setup(self, **kvargs):
        """Called on model compile.

        Returns:
            (w, b): Weight and bias.
        """

        self.layer_idx = kvargs['layer_idx']        
        w = np.random.randn(kvargs['size_curr'], np.prod(kvargs['size_prev'])) * 0.1
        b = np.ones(kvargs['size_curr'], dtype=np.float32)        
        return w, b

    def forward_pass(self, params, **kwargs):
        """Forward pass through dense layer.

        Args:
            params (dict): Model parameters.
        """
        w = params[f'W{self.layer_idx}']        
        b = params[f'B{self.layer_idx}']                
        prev_a = self.kernel_constraint.constrain(params[f'A{self.layer_idx - 1}'])

        # Referenced object `params` is modified        
        params[f'Z{self.layer_idx}'] = np.dot(w, prev_a) + b
        params[f'A{self.layer_idx}'] = self.activation(params[f'Z{self.layer_idx}'])

    def backward_pass(self, gradient, is_last=False, output=None, target=None, new_params=None, model=None, **kvargs):
        """Performs backpropagation by calculating weight and bias gradients and returning `gradient` which is dL/dx.

        Args:
            gradient: Partial derivative with respect to activations from previous layer.
            is_last (bool, optional): Whether this layer is the last one in the model. Defaults to False.
            output (optional): Output from this layer during forward pass. Defaults to None.
            target (optional): The target output from the model. Only used when `is_last` is True. Defaults to None.
            new_params (dict, optional): Dictionary with model gradients for weights and biases. Defaults to None.
            model (optional): Reference to the `Sequential` model. Defaults to None.

        Returns:
            np.array: dL/dx. How the loss-function is affected by input changes.
        """

        activation_deriv = model.layers[self.layer_idx].activation(model.params[f'Z{self.layer_idx}'], derivative=True)
        if not is_last:
            local = gradient * activation_deriv
        else:
            local = 2 * (output - target) / output.shape[0] * activation_deriv

        new_params[f'W{self.layer_idx}'] = np.outer(local, model.params[f'A{self.layer_idx - 1}'])        
        new_params[f'B{self.layer_idx}'] = local

        out = np.dot(model.params[f'W{self.layer_idx}'].T, local.T)
        return out

    def __str__(self) -> str:
        return f"<Dense layer class object named '{self.name}' of size {self.size} using activation function '{self.activation.__name__}'>"


class Conv2D():
    def __init__(self, num_filters, kernel_size=(3, 3), input_shape=(1, 5, 5), strides=(1,1), kernel=None, padding="same", activation=activations.identity, kernel_constraint=constraints.no_norm(), name: str = "conv2d"):
        """2D convolutional layer.        

        Args:
            num_filters (int): Number of filters in layer.
            kernel_size ((x, y), optional): Size of kernel Both dimensions must be odd. Defaults to (3, 3).
            input_shape ((channels, x, y), optional): Shape of input data. Defaults to (1, 5, 5).
            strides ((x, y), optional): NOT implemented. Step size when sliding across the grid. Defaults to (1, 1).
            kernel (optional): Weights for kernel. Must have shape `(num_filters, *input_shape)`. If nothing is provided, weights will be instantiated randomly. Defaults to None.
            padding (optional): Either "same" or "valid". "same" means padding is added evenly on all sides, so that output shape equals input shape. "valid" is the same as no padding. Defaults to "same".
            activation (optional): Activation function used at this layer.
            kernel_constraint (optional): How the kernel should be constrained.
            name (str, optional): Name of the layer. Defaults to "conv2d".

        Todo:
            Stride.
            Merge `full_back_conv()` with `convolve()`.
        """

        self.num_filters = num_filters
        self.kernel_size = np.array(kernel_size)
        self.input_shape = np.array(input_shape)
        self.strides = np.array(strides)
        self.kernel = kernel
        self.padding = self.calc_padding(padding)
        self.activation = activation
        self.kernel_constraint = kernel_constraint
        self.name = name

        self.layer_idx = 0  # Updated at `Sequential.compile()`
        self.num_params = self.calc_num_params()
        self.output_shape = self.calc_output_shape()

    def setup(self, **kvargs):
        """Setup layer and instantiate weights and biases.

        Returns:
            list: Weights and biases.
        """

        self.layer_idx = kvargs['layer_idx']

        w = self.kernel
        w_shape = (self.num_filters, self.input_shape[0], self.kernel_size[0], self.kernel_size[1])

        if w is not None:
            assert w.shape == w_shape
        else:
            w = np.random.randn(*w_shape)

        b = np.zeros(self.num_filters)
        return w, b

    def calc_padding(self, padding):
        """Calculate padding size left and right in x-direction and up and down in y-direction.

        Args:
            padding: Either "same" or "valid". "same": Padding is added equally on all sides and output grid has same size as input. "valid": No padding is added.

        Returns:
            list: Padding in x and y-direction.
        """

        pad = np.array([0,0])
        if padding == 'same': pad = (self.kernel_size - 1) // 2
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


    # Previously named `forward_pass`
    def forward_pass_naive(self, params, **kwargs):
        """Perform forward pass through layer based on weights and biases from `params`.
        Model `params` is updated in place.

        Args:
            params (dict): Model parameters for all layers.
        """
        w = params[f'W{self.layer_idx}']
        b = params[f'B{self.layer_idx}']
        prev_a = self.kernel_constraint.constrain(params[f'A{self.layer_idx - 1}'])
        conv = self.convolve(prev_a, w, self.padding)
        params[f'Z{self.layer_idx}'] = (conv / np.prod(self.kernel_size)) + b[:, np.newaxis, np.newaxis]
        params[f'A{self.layer_idx}'] = self.activation(params[f'Z{self.layer_idx}'])

    def forward_pass(self, params, **kvargs):
        """Perform forward pass through layer based on weights and biases from `params` using 'stride groups' instead of nested for loops.
        Model `params` is updated in place.

        Args:
            params (dict): Model parameters for all layers.
        """
        # Variables used in this function
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
        # p     :   Padding
        ### Stride groups
        # t     :   Top index of stride group to be split into arrays with kernel size
        # b     :   Bottom index of stride group to be split into arrays with kernel size
        # l     :   Left index
        # r     :   Right index

        I = self.kernel_constraint.constrain(params[f'A{self.layer_idx - 1}'])
        I = np.expand_dims(I, 0)                # Temporary. Adds dimension for training examples
        W = params[f'W{self.layer_idx}']
        # b = params[f'B{self.layer_idx}']      # Bias not implemented right now

        # Add padding to input        
        if self.padding[0] > 0:
            I = np.pad(I, [(0,0), (0,0), (self.padding[0],self.padding[0]), (self.padding[1],self.padding[1])])

        i_n, i_d, i_h, i_w = I.shape                        # Assigning dimension variables after padding
        self.inputs_shape = I.shape
        f_n = self.num_filters
        f_w = self.kernel_size[0]

        # Number of stride group types
        self.n_rows = math.ceil(min(f_w, i_h - f_w + 1))
        self.n_cols = math.ceil(min(f_w, i_w - f_w + 1))

        # Z matrix
        z_h = int(((i_h - f_w)) + 1)
        z_w = int(((i_w - f_w)) + 1)
        Z = np.empty([i_n, f_n, z_h, z_w])
        self.input_blocks = []
        # Iterating stride groups
        # input_blocks = []
        # Iterate over top indices of row stride groups ignoring stride
        for t in range(self.n_rows):
            self.input_blocks.append([])
            # Calculate bottom idx of last block in row stride group
            b = i_h - (i_h - t) % f_w
            # Matrix for current stride group row selection given current row top and bottom
            cols = np.empty([i_n, f_n, int((b - t) / f_w), z_w])

            # Iterate over left indices of column stride groups ignoring stride
            for l in range(self.n_cols):
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
                self.input_blocks[t].append(block)
                # Matrix-matrix product between input and weights
                block = block * W
                # Sum up across input channels
                block = np.sum(block, 6)
                block = np.sum(block, 5)
                block = np.sum(block, 4)
                block = np.moveaxis(block, 3, 1)
                # Save to column output matrix
                cols[:, :, :, l::self.n_cols] = block
            Z[:, :, t::self.n_rows, :] = cols

        # Z += b                        # Add bias here
        Z = np.sum(Z, 0)                # Temp
        A = self.activation(Z)          # Activation function here        
        
        params[f'Z{self.layer_idx}'] = Z
        params[f'A{self.layer_idx}'] = A

    def back_convolve_loopless(self, a, b, padding, back_conv = False):
        """Convolution between matrices `a` and `b`.

        Args:
            a: Stationary matrix.
            b: Moving matrix.
            padding: Two numbers representing padding.
            back_conv (bool): Whether the convolution is during a backward_pass or not. Defaults to False.

        Todo:
            Not completely dynamic right now

        Returns:
            array: Matrix containing convoluted output.
        """
        print("back convolve")
        print(a.shape)
        print(b.shape)
        start_time = datetime.now()
        padding[0] = (a.shape[1] + 2*padding[0] - b.shape[1] + 1 + b.shape[1] - 1 - a.shape[1]) / 2
        padding[1] = padding[0]
        print(padding[0])
        a_pad = np.zeros([a.shape[0], a.shape[1] + padding[0] * 2, a.shape[2] + padding[1] * 2])
        a_pad[:,padding[0]:padding[0] + a.shape[1], padding[1]:padding[1] + a.shape[2]] = a        
        i0 = np.repeat(np.arange(28), 28)
        i1 = np.repeat(np.arange(3), 3)
        i = i0.reshape(-1,1) + i1.reshape(1, -1)

        j0 = np.tile(np.arange(28), 28)
        j1 = np.tile(np.arange(3), 3)
        j = j0.reshape(-1,1) + j1.reshape(1, -1)
        # np.set_printoptions(threshold=np.inf)
        # print(i)
        # print(i.shape)
        # print()
        # print(j)
        # print(j.shape)
        # k = np.repeat(np.arange(b.shape[0]), np.prod(b.shape[2:])).reshape(-1,1)
        print(a_pad.shape)
        select_img = a_pad[:, i, j]
        print(select_img.shape)
        b_reshaped = b.reshape(b.shape[0], b.shape[1] * b.shape[2])        
        print("back_convolve")
        print(b_reshaped.shape)
        print(select_img.shape)
        print(f"time {datetime.now() - start_time}")
        dot_product = np.tensordot(b_reshaped[:], select_img[:], axes=([1],[1])).reshape(64,32,3,3)
        print(dot_product.shape)
        return dot_product

    def convolve_loopless(self, a, b, padding, back_conv = False):
        """Convolution between matrices `a` and `b`.

        Args:
            a: Stationary matrix.
            b: Moving matrix.
            padding: Two numbers representing padding.
            back_conv (bool): Whether the convolution is during a backward_pass or not. Defaults to False.

        Returns:
            array: Matrix containing convoluted output.
        """

        start_time = datetime.now()
        a_pad = np.zeros([a.shape[0], a.shape[1] + padding[0] * 2, a.shape[2] + padding[1] * 2])
        a_pad[:,padding[0]:padding[0] + a.shape[1], padding[1]:padding[1] + a.shape[2]] = a
        
        i0 = np.repeat(np.arange(b.shape[2]), b.shape[2])
        i1 = np.repeat(np.arange(a.shape[2]), a.shape[2])
        i = i0.reshape(-1,1) + i1.reshape(1, -1)

        j0 = np.tile(np.arange(b.shape[3]), b.shape[3])
        j1 = np.tile(np.arange(a.shape[2]), a.shape[2])
        j = j0.reshape(-1,1) + j1.reshape(1, -1)

        # k = np.repeat(np.arange(b.shape[0]), np.prod(b.shape[2:])).reshape(-1,1)

        select_img = a_pad[:, i, j]
        weights = b.reshape(b.shape[0], b.shape[1], b.shape[2] * b.shape[3])        
        dot_product = np.tensordot(weights[:], select_img).reshape(b.shape[0], 28, 28)        
        print(f"time {datetime.now() - start_time}")
        print(dot_product.shape)
        return dot_product


    def convolve(self, a, b, padding, back_conv = False):
        """Convolution between matrices `a` and `b`.

        Args:
            a: Stationary matrix.
            b: Moving matrix.
            padding: Two numbers representing padding.
            back_conv (bool): Whether the convolution is during a backward_pass or not. Defaults to False.

        Returns:
            array: Matrix containing convoluted output.
        """
        
        a_with_pad = np.zeros((a.shape[0], *(a.shape[1:] + padding * 2)))
        a_with_pad[:, padding[0] : padding[0] + a.shape[1], padding[1]:padding[1] + a.shape[2]] = a
        if not back_conv:
            out = np.zeros((b.shape[0], *(a.shape[2:] + padding * 2 - b.shape[2:] + 1)))            
        else:
            out = np.zeros((b.shape[0], a.shape[0], *(a.shape[2:] + padding * 2 - b.shape[2:] + 1)))            
        
        for x in range(out.shape[-2]):
            for y in range(out.shape[-1]):
                if not back_conv:
                    dot = np.tensordot(a_with_pad[:, x: x + b.shape[2], y: y + b.shape[3]], b, axes=([0,1,2], [1,2,3]))
                    out[:,x,y] = dot
                else:                    
                    dot = np.tensordot(a_with_pad[:, x: x + b.shape[1], y: y + b.shape[2]], b, axes=([1,2], [1,2]))
                    out[:,:,x,y] = dot.T
        return out

    def full_back_conv(self, a, b, padding, next_padding):
        """Full convolution between matrices `a` and `b`.

        Args:
            a: Stationary matrix.
            b: Moving matrix.
            padding: Two numbers representing padding.
            next_padding: Two numbers representing padding to be romoved before returning.

        Returns:
            array: Matrix containing convoluted output.
        """

        a_with_pad = np.zeros((*a.shape[:-2], *(a.shape[-2:] + padding * 2)))
        a_with_pad[:,:, padding[0] : padding[0] + a.shape[-2], padding[1]:padding[1] + a.shape[-1]] = a
        
        out = np.zeros((a.shape[0], *(a.shape[2:] + padding * 2 - b.shape[2:] + 1)))
        
        for x in range(out.shape[-2]):
            for y in range(out.shape[-1]):                
                dot = np.tensordot(a_with_pad[:, :, x: x + b.shape[1], y: y + b.shape[2]], b, axes=([1,2,3], [0,1,2]))
                out[:,x,y] = dot

        # Remove original padding
        out = out[:,next_padding[0]:-next_padding[0], next_padding[1]:-next_padding[1]]
        return out

    def full_back_conv_loopless(self, a, b, padding, next_padding):
        """Full convolution between matrices `a` and `b`.

        Args:
            a: Stationary matrix.
            b: Moving matrix.
            padding: Two numbers representing padding.
            next_padding: Two numbers representing padding to be romoved before returning.

        Returns:
            array: Matrix containing convoluted output.
        """
        print("1 full back")
        print(a.shape)
        print(b.shape)
        print(padding)
        print(next_padding)
        a_with_pad = np.zeros((*a.shape[:-2], *(a.shape[-2:] + padding * 2)))
        a_with_pad[:,:, padding[0] : padding[0] + a.shape[-2], padding[1]:padding[1] + a.shape[-1]] = a
        print(a_with_pad.shape)

        i0 = np.repeat(np.arange(b.shape[1]), b.shape[1])
        i1 = np.repeat(np.arange(b.shape[1]), b.shape[1])
        i = i0.reshape(-1,1) + i1.reshape(1, -1)
        # print(i)

        j0 = np.tile(np.arange(b.shape[2]), b.shape[2])
        j1 = np.tile(np.arange(b.shape[2]), b.shape[2])
        j = j0.reshape(-1,1) + j1.reshape(1, -1)
        # print(j)
        # k = np.repeat(np.arange(b.shape[0]), np.prod(b.shape[2:])).reshape(-1,1)
        b_reshaped = b.reshape(b.shape[0], b.shape[1] * b.shape[2])        

        out = None
        for f in range(a_with_pad.shape[0]):
            # for each filter
            print("a_with_pad.shape")
            print(a_with_pad.shape)
            select_img = a_with_pad[f][:, i, j]
            print("select_img")
            print(select_img.shape)
            
            print("b_reshaped.shape")
            print(b_reshaped.shape)

            dot_product = np.tensordot(b_reshaped, select_img, axes=([1], [1]))
            print("dot_product.shape")
            print(dot_product.shape)
            if not out is None:
                out = dot_product
            else: out += dot_product

        # print(select_img.shape)


        # print("2")
        # out = np.zeros((a.shape[0], *(a.shape[2:] + padding * 2 - b.shape[2:] + 1)))

        # print(f"{out.shape[-2], out.shape[-1]}")
        # for x in range(out.shape[-2]):
        #     for y in range(out.shape[-1]):
        #         print(f"{x, y}")
        #         dot = np.tensordot(a_with_pad[:, :, x: x + b.shape[1], y: y + b.shape[2]], b, axes=([1,2,3], [0,1,2]))
        #         out[:,x,y] = dot

        # # Remove original padding
        # out = out[:,next_padding[0]:-next_padding[0], next_padding[1]:-next_padding[1]]
        # return out

    # Previously named `backward_pass`
    def backward_pass_naive(self, gradient, is_last=False, output=None, target=None, new_params=None, model=None, **kvargs):
        """Performs backpropagation by calculating weight and bias gradients and returning `gradient` which is dL/dx.
            Possible gradient. Not keeping track of layers.


        Args:
            gradient: Partial derivative of the final loss function with respect to activations from n+1 layer.
            is_last (bool, optional): Whether this layer is the last one in the model. Defaults to False.
            output (optional): Output from this layer during forward pass. Defaults to None.
            target (optional): The target output from the model. Only used when `is_last` is True. Defaults to None.
            new_params (dict, optional): Dictionary with model gradients for weights and biases. Defaults to None.
            model (optional): Reference to the `Sequential` model. Defaults to None.

        Returns:
            np.array: dL/dx. How the loss-function is affected by input changes.
        """
        
        activation_deriv = model.layers[self.layer_idx].activation(model.params['Z' + str(self.layer_idx)], derivative=True)
        if not is_last:
            local = gradient * activation_deriv
        else:
            local = 2 * (output - target) * activation_deriv

        prev_a = model.params['A' + str(self.layer_idx - 1)]        
        new_w = self.convolve(prev_a, local, self.padding, back_conv=True)
        new_params['W' + str(self.layer_idx)] = new_w / np.prod(new_w.shape[1:]) * model.learning_rate
        new_params[f'B{self.layer_idx}'] = np.zeros(self.num_filters)

        # prepare gradient to return from this layer
        w = model.params['W' + str(self.layer_idx)]
        arr = np.flip(np.flip(w, -2), -1)
        if len(arr.shape) > 3:
            arr = np.swapaxes(arr, 0,1)
        else: print("didn't swap axes")

        out_padding = np.array(local.shape[-2:]) - 1
        in_padding = self.padding

        return self.full_back_conv(arr, local, out_padding, in_padding)

    def backward_pass(self, gradient, is_last=False, output=None, target=None, new_params=None, model=None, **kvargs):
        """Performs backpropagation by calculating weight and bias gradients and returning `gradient` which is dL/dx.
            Possible gradient. Not keeping track of layers.


        Args:
            gradient: Partial derivative of the final loss function with respect to activations from n+1 layer.
            is_last (bool, optional): Whether this layer is the last one in the model. Defaults to False.
            output (optional): Output from this layer during forward pass. Defaults to None.
            target (optional): The target output from the model. Only used when `is_last` is True. Defaults to None.
            new_params (dict, optional): Dictionary with model gradients for weights and biases. Defaults to None.
            model (optional): Reference to the `Sequential` model. Defaults to None.

        Returns:
            np.array: dL/dx. How the loss-function is affected by input changes.
        """

        activation_deriv = model.layers[self.layer_idx].activation(model.params['Z' + str(self.layer_idx)], derivative=True)
        if not is_last:
            local = gradient * activation_deriv
        else:
            local = 2 * (output - target) * activation_deriv

        dZ = np.expand_dims(local, 0)
        i_n, i_d, i_h, i_w = self.inputs_shape        
        f_n = self.num_filters        
        f_w = self.kernel_size[0]
        W = model.params[f'W{self.layer_idx}']                

        dA_prev = np.zeros([i_n, i_d, i_h, i_w])
        dW = np.zeros([f_n, i_d, f_w, f_w])

        for t in range(self.n_rows):
            row = dZ[:, :, t::self.n_rows, :]
            b = (i_h - t) % f_w       # Number of cells below. Different from `b` from forward pass
            for l in range(self.n_cols):
                r = (i_w - l) % f_w
                block = row[:, :, :, l::self.n_cols]
                block = np.moveaxis(block, 1, 3)
                block = np.expand_dims(block, 4)
                block = np.expand_dims(block, 4)
                block = np.expand_dims(block, 4)
                dW_block = block * self.input_blocks[t][l]

                dW_block = np.sum(dW_block, 2)
                dW_block = np.sum(dW_block, 1)
                dW_block = np.sum(dW_block, 0)
                dW += dW_block

                dA_prev_block = block * W
                dA_prev_block = np.sum(dA_prev_block, 3)
                dA_prev_block = np.reshape(dA_prev_block, (i_n, i_d, i_h - b - t, i_w - r - l))            
                dA_prev[:, :, t:i_h - b, l:i_w - r] += dA_prev_block            
            
        if self.padding[0] > 0:
            dA_prev = dA_prev[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]        
       
        new_params['W' + str(self.layer_idx)] = dW
        new_params[f'B{self.layer_idx}'] = np.zeros(self.num_filters)      
        
        return np.sum(dA_prev, 0)

    def __str__(self) -> str:
        return f"<Conv2D layer class object named '{self.name}' with input shape {self.input_shape} and filter shape '{self.kernel_size}'>"


class MaxPooling2D():    
    def __init__(self, pool_size, kernel_constraint=constraints.no_norm(), name="max_pool"):
        """Max pooling layer often used to highlight features after a convolutional layer.

        Args:
            pool_size: Tuple of two integers representing the kernel shape.
            kernel_constraint: Regularization of kernel weights.
            name (str, optional): Name of the layer. Defaults to "max_pool".
        """

        self.pool_size = np.array(pool_size)
        self.kernel_constraint = kernel_constraint
        self.name = name
        self.output_shape = None
        self.num_params = None

    def setup(self, **kvargs):
        """Setup layer and instantiate index matrix.

        Returns:
            list: Idx matrix and biases.
        """

        self.layer_idx = kvargs['layer_idx']
        self.input_shape = kvargs['size_prev']
        self.output_shape = (self.input_shape[0], *(self.input_shape[-2:] // self.pool_size))

        # Updating pool size with prev layer channel dimension
        self.pool_size = np.array((self.input_shape[0], *self.pool_size))
        idx = np.ones(self.input_shape)
        return idx, None

    def pool(self, a, b, out_shape):
        """Convolve `b` across `a`.

        Args:
            a: Stationary matrix.
            b: Moving pool matrix shape.

        Todo:
            Use numpy slide and stride. [::]
        """

        stride = b[-2:]

        out = np.zeros(out_shape)
        out_coords = np.zeros((*out.shape, 2))

        for x in range(out.shape[-2]):
            for y in range(out.shape[-1]):
                arr_x = x * stride[0]
                arr_y = y * stride[1]

                curr_a = a[:, arr_x : arr_x + stride[0], arr_y : arr_y + stride[1]]
                curr_a_reshape = curr_a.reshape(curr_a.shape[0], -1)
                max = curr_a_reshape.argmax(axis=1)
                coords = np.column_stack(np.unravel_index(max, curr_a.shape[1:]))
                vals = curr_a_reshape[np.arange(curr_a_reshape.shape[0]), max]

                coords = coords.T
                coords[0] += arr_x
                coords[1] += arr_y                
                out_coords[:,x,y,:] = coords.T
                out[:,x,y] = vals

        return out_coords, out


    def forward_pass(self, params, **kwargs):
        """Forward pass through `MaxPooling2D` layer. Max value in each kernel stride is saved 
        as output activations, and indices of max values are saved as well.

        Args:
            params (dict): Model parameters.
        """
        
        prev_a = self.kernel_constraint.constrain(params[f'A{self.layer_idx - 1}'])
        out_coords, out = self.pool(prev_a, self.pool_size, self.output_shape)

        # Normalize each channel separately
        for channel in range(out.shape[0]):
            max = np.max(out[channel])
            min = np.min(out[channel])
            divisor = np.max([max, np.absolute(min)])
            out[channel] /= divisor

        params[f'A{self.layer_idx}'] = out
        params[f'I{self.layer_idx}'] = out_coords.astype(int)   # Indices used during backward_pass

    def backward_pass(self, gradient, is_last=False, output=None, target=None, model=None, **kvargs):
        """Performs backpropagation by calculating weight and bias gradients and returning `gradient` which is dL/dx.

        Args:
            gradient: Partial derivative with respect to activations from previous layer.
            is_last (bool, optional): Whether this layer is the last one in the model. Defaults to False.
            output (optional): Output from this layer during forward pass. Defaults to None.
            target (optional): The target output from the model. Only used when `is_last` is True. Defaults to None.
            new_params (dict, optional): Dictionary with model gradients for weights and biases. Defaults to None.
            model (optional): Reference to the `Sequential` model. Defaults to None.

        Returns:
            np.array: dL/dx. How the loss-function is affected by input changes.
        """

        if not is_last:
            local = gradient
        else:
            local = 2 * (output - target)

        arr_idx = model.params['I' + str(self.layer_idx)].copy()
        out_gradient = np.ones(self.input_shape)
        for n in range(arr_idx.shape[0]):
            n_coords = np.concatenate(arr_idx[n], 0)
            n_coords = n_coords.T
            out_gradient[n, n_coords[0], n_coords[1]] = local[n].flatten()

        return out_gradient


class Flatten():
    def __init__(self, name='flatten'):
        """Layer used for flattening the data matrix.

        Args:            
            name (str, optional): Name of the layer. Defaults to 'flatten'.
        """
        
        self.name = name
        self.output_shape = None
        self.num_params = None
        self.size_prev = None

    def setup(self, **kvargs):
        """Setup layer and instantiate input and output shapes.

        Returns:
            list: Weights and biases.
        """

        self.layer_idx = kvargs['layer_idx']
        self.size_prev = kvargs['size_prev']
        self.output_shape = np.array([np.prod(self.size_prev)])
        return None, None

    def forward_pass(self, params, **kwargs):
        """Forward pass through `Flatten` layer. Data matrix is flattened.

        Args:
            params (dict): Model parameters.
        """

        params[f'A{self.layer_idx}'] = params[f'A{self.layer_idx - 1}'].flatten()

    def backward_pass(self, gradient, is_last=False, output=None, target=None, **kvargs):
        """Performs backpropagation by calculating or reshaping input gradients and returning `gradient` which is dL/dx.

        Args:
            gradient: Partial derivative with respect to activations from previous layer.
            is_last (bool, optional): Whether this layer is the last one in the model. Defaults to False.
            output (optional): Output from this layer during forward pass. Defaults to None.
            target (optional): The target output from the model. Only used when `is_last` is True. Defaults to None.

        Returns:
            np.array: dL/dx. How the loss-function is affected by input changes.
        """
        if not is_last:
            local = gradient
        else:
            local = 2 * (output - target)

        return local.reshape(self.size_prev)


class Reshape():
    def __init__(self, output_shape, input_shape=None, name='reshape'):
        """Layer used for reshaping the data matrix.

        Args:
            output_shape (tuple): Shape of output data.
            input_shape (tuple, optional): Shape of input data used when retransforming during backward pass. Defaults to None.
            name (str, optional): Name of the layer. Defaults to 'reshape'.
        """

        self.name = name
        self.input_shape = input_shape if input_shape is not None else np.array([0])
        self.output_shape = output_shape
        self.num_params = None
        self.size_prev = None

    def setup(self, **kvargs):
        """Setup layer and instantiate input and output shapes.

        Returns:
            list: Weights and biases.
        """

        self.layer_idx = kvargs['layer_idx']
        self.size_prev = kvargs['size_prev']
        return None, None

    def forward_pass(self, params, **kwargs):
        """Forward pass through `Reshape` layer. Data matrix is reshaped.

        Args:
            params (dict): Model parameters.
        """

        params[f'A{self.layer_idx}'] = params[f'A{self.layer_idx - 1}'].reshape(self.output_shape)

    def backward_pass(self, gradient, is_last=False, output=None, target=None, **kvargs):
        """Performs backpropagation by calculating or reshaping input gradients and returning `gradient` which is dL/dx.

        Args:
            gradient: Partial derivative with respect to activations from previous layer.
            is_last (bool, optional): Whether this layer is the last one in the model. Defaults to False.
            output (optional): Output from this layer during forward pass. Defaults to None.
            target (optional): The target output from the model. Only used when `is_last` is True. Defaults to None.

        Returns:
            np.array: dL/dx. How the loss-function is affected by input changes.
        """

        if not is_last:
            local = gradient
        else:
            local = 2 * (output - target)
        return local.reshape(self.size_prev)


class Dropout():
    def __init__(self, keep_prob, name='dropout'):
        """Layer used for regulating output by dropping out nodes.

        Args:
            keep_prob (float): Probability of keeping a given unit.
            name (str, optional): Name of the layer. Defaults to 'dropout'.
        """

        self.name = name
        assert keep_prob <= 1
        self.keep_prob = keep_prob
        self.output_shape = None
        self.num_params = None
        self.size_prev = None

    def setup(self, **kvargs):
        """Setup layer and instantiate output shapes.

        Returns:
            list: Weights and biases.
        """

        self.layer_idx = kvargs['layer_idx']
        self.size_prev = kvargs['size_prev']        
        self.output_shape = self.size_prev
        return None, None

    def forward_pass(self, params, **kwargs):
        """Forward pass through `Dropout` layer. Node activations are set to 0.

        Args:
            params (dict): Model parameters.
        """
        is_training = kwargs['is_training']

        self.last_a = params[f'A{self.layer_idx - 1}'].copy()
        if is_training:
            self.mask = np.random.choice([False, True], size=self.last_a.shape, p=[self.keep_prob, 1 - self.keep_prob])
            
            params[f'A{self.layer_idx}'] = self.last_a
            params[f'A{self.layer_idx}'][self.mask] *= 0
        else:
            params[f'A{self.layer_idx}'] = self.last_a

    def finalize(self, params):
        """Scale activations when network is not training.

        Args:
            params: Network parameters.
        """

        params[f'A{self.layer_idx}'] /= self.keep_prob
    
    def unfinalize(self, params):
        """Rescale activations when network will be training again.

        Args:
            params: Network parameters.
        """

        params[f'A{self.layer_idx}'] *= self.keep_prob

    def backward_pass(self, gradient, is_last=False, output=None, target=None, **kvargs):
        """Performs backpropagation by calculating or reshaping input gradients and returning `gradient` which is dL/dx.

        Args:
            gradient: Partial derivative with respect to activations from previous layer.
            is_last (bool, optional): Whether this layer is the last one in the model. Defaults to False.
            output (optional): Output from this layer during forward pass. Defaults to None.
            target (optional): The target output from the model. Only used when `is_last` is True. Defaults to None.

        Returns:
            np.array: dL/dx. How the loss-function is affected by input changes.
        """

        if not is_last:
            local = gradient
        else:
            local = 2 * (output - target)
        
        local[self.mask] *= 0

        return local