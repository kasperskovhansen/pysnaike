"""Different types of models used in ML and AI."""


import numpy as np
import pysnaike.activations as activations
from tqdm import tqdm
import os.path
from tabulate import tabulate


class Sequential:
    """Deep Neural Network made of stacked layers.

    Attributes:
        layers (list): List of layers in the model.
        params (dict): Model parameters.
        learning_rate (float): Size of learning steps.

    Todo:
        Add checkpoint callback support.
    """

    def __init__(self) -> None:
        self.layers = []
        self.params = {}
        self.learning_rate = 0.001

    def add(self, layer):
        """Add a layer to the model.

        Args:
            layer: The layer to add.
        """

        self.layers.append(layer)

    def description(self):
        """Print a table containing information about the layers in the model.
        """

        data = []
        sum_params = 0
        for layer in self.layers:
            if layer.num_params: sum_params += layer.num_params
            activation = getattr(getattr(layer, 'activation', None), '__name__', None)
            data.append([layer.layer_idx, layer.name, layer.output_shape, activation, layer.num_params])
        data.append([None])
        data.append(['Total:', None, None, None, sum_params])
        print(tabulate(data, headers=['Idx', 'Name', 'Output shape', 'Activation', 'Num params'], tablefmt='pretty'))

    def compile(self):
        """Compile model architecture.
        """

        layer_names = {}
        # size_prev = self.layers[0].input_shape
        size_prev = None
        for i in range(0, len(self.layers)):
            curr = self.layers[i]

            # Naming layers
            if not curr.name in layer_names:
                layer_names[curr.name] = 1
            else: layer_names[curr.name] += 1
            curr.name += f'_{str(layer_names[curr.name])}'

            # Setup layers based on surrounding layers
            # if i > 0:
            if size_prev is None: size_prev = self.layers[0].input_shape
            size_curr = curr.output_shape
            size_prev = self.layers[i - 1].output_shape
            self.params[f'W{i}'], self.params[f'B{i}'] = curr.setup(size_curr=size_curr, size_prev=size_prev, layer_idx=i)

    def sgd_update_network_params(self, new_params):
        """Adjust network parameters according to update rule from Stochastic Gradient Descent.

        Args:
            new_params (dict, required): Dictionary containing the gradients.
        """

        for key, value in new_params.items():
            if key == 'error': continue
            self.params[key] -= self.learning_rate * value


    def mini_b_update_network_params(self, new_params, divisor):
        """Adjust network parameters with `new_params` divided by `divisor`.

        Args:
            new_params (dict, required): Adjustments to model parameters.
            divisor (int, required): Mini batch size.
        """
        for key, value in new_params.items():
            if key == 'error': continue
            self.params[key] -= value / divisor

    def iterate_minibatches(self, inputs, outputs, batch_size, shuffle=False):
        """Split dataset into a number of batches with size `batch_size`.

        Args:
            inputs (list, required): All training example inputs.
            outputs (list, required): All training example outputs.
            batch_size (int, required): Size of each mini batch.
            shuffle (bool, optional): Whether the training dataset should be shuffled before being split. Defaults to False.

        Yields:
            list: Examples included in a mini batch. Yielded as a list containing a list with inputs and a list with outputs.
        """

        # assert inputs.shape[0] == outputs.shape[0]

        indices = np.arange(inputs.shape[0])
        if shuffle: np.random.shuffle(indices)

        mod = indices.shape[0] % batch_size
        if not mod == 0: indices = indices[0: - int(indices.shape[0] % batch_size)]

        for start_idx in range(0, indices.shape[0] - batch_size + 1, batch_size):
            end_idx = start_idx + batch_size
            if shuffle:
                excerpt = indices[start_idx : end_idx]
            else:
                excerpt = slice(start_idx, end_idx)
            yield [inputs[excerpt], outputs[excerpt]]

    def train(self, inputs, outputs, optimizer='SGD', epochs=10, learning_rate=0.001, mini_b_size=16, mini_b_shuffle=False, callbacks=None):
        """Train the model.

        Args:
            inputs (list, required): Training example inputs.
            outputs (list, required): Training example outputs.
            optimizer (str, optional): The optimization algorithm used in training. Defaults to 'SGD'.
            epochs (int, optional): Number of epochs. Defaults to 10.
            learning_rate (float, optional): Size of steps used in backpropagation. Defaults to 0.001.
            mini_b_size (int, optional): Size of each mini batch. Defaults to 16.
            mini_b_shuffle (bool, optional): Whether mini batch division should be random. Defaults to False.
        """

        # assert inputs.shape[0] == outputs.shape[0]

        self.learning_rate = learning_rate

        for epoch in tqdm(range(epochs)):
            if optimizer.upper() == 'SGD':
                mini_b_size = 1
            elif optimizer.upper() == 'BATCH':
                mini_b_size = inputs.shape[0]

            mini_batches = self.iterate_minibatches(inputs, outputs, mini_b_size, mini_b_shuffle)

            for mini_batch in mini_batches:
                batch_x, batch_y = mini_batch
                avg_new_weights = {}

                for x, y in zip(batch_x, batch_y):
                    output = self.forward_pass(x)
                    new_weights = self.backward_pass(output, y)

                    if not avg_new_weights: avg_new_weights = new_weights
                    else:
                        for key, value in new_weights.items():
                            avg_new_weights[key] -= learning_rate * value
                self.mini_b_update_network_params(new_weights, batch_x.shape[0])

    def compute_accuracy(self, inputs, outputs):
        """Test the network accuracy by comparing the network output with
        correct labels for a number of unknown test examples.

        Args:
            inputs (list, required): List of input vectors.
            outputs (list, required): List of vectors containing output labels.

        Returns:
            float: The accuracy as a number between 0 and 1.
        """

        predictions = []
        for _, (x, y) in enumerate(zip(inputs, outputs)):
            pred = self.forward_pass(x)
            predictions.append(pred.argmax() == y.argmax())

        return np.mean(predictions)

    def forward_pass(self, inputs):
        """Calculate the output from the network for a given set of input values.

        Args:
            inputs (required): Input values for the network.

        Returns:
            list: The output values from the network.
        """

        params = self.params
        params['A0'] = inputs

        for i in range(1, len(self.layers)):
            self.layers[i].forward_pass(params)

        return params.get(f'A{len(self.layers) - 1}', f'No output from layer {len(self.layers) - 1}')

    def backward_pass(self, output, target):
        """Propagate backward through the network.

        Using the backpropagation algorithm to calculate the updates for the neural network parameters.

        Args:
            output: Result from forward pass.
            target: Correct labels for given example.

        Returns:
            dict: Gradients for the network stored in a dictionary.
        """

        new_params = {}
        last_layer = len(self.layers) - 1
        is_last = True
        gradient = None


        # Propagate backward through every layer. Model parameters are mutated inplace and the immutable `gradient` is returned
        for i in reversed(range(1, last_layer + 1)):
            gradient = self.layers[i].backward_pass(gradient, is_last=is_last, output=output, target=target, new_params=new_params, model=self)
            if is_last:
                is_last = False

        return new_params

    def load_params(self, params_path):
        """Load network parameters from `params_path`.

        Args:
            params_path (str): Path to .npz file.
        """
        # Load trained network from .npz file
        if os.path.exists(params_path):
            params = np.load(params_path)
            for key in params.files:
                self.params[key] = params[key]

    def __str__(self) -> str:
        return f"<Sequential model class object>"
