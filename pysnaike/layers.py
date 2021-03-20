"""Different types of layers in a network."""


import numpy as np


class Dense():
    def __init__(self, size: int, name: str = "dense", activation: str = "relu"):        
        """Dense layer.
        
        Args:
            size (int): Number of perceptrons in the layer.
            name (str, optional): Name of the input layer.
            activation (str, optional): Activation function used at this layer.
        """

        self.size = size      
        self.name = name
        self.activation = activation            

    def __str__(self) -> str:
        return f"<Dense layer class object named '{self.name}' of size {self.size} using activation function '{self.activation.__name__}'>"


