"""AI client able to interact with connect four-game.
"""

import os.path
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
import numpy as np
from pysnaike import activations, dql, layers, models


class Connect_four_ai():
    def __init__(self, params_path=None, memory_path=None):
        """Connect four client.

        Args:
            params_path (str, optional): Path of .npz file with params to load. Defaults to None.
            memory_path (str, optional): Path of .npz file with memory_bank to load. Defaults to None.
        """

        # AI client network
        self.Q = models.Sequential()
        self.Q.add(layers.Reshape((1,6,6)))
        self.Q.add(layers.Conv2D(128, kernel_size=(4,4), input_shape=(1,6,6), padding="valid", activation=activations.sigmoid))        
        self.Q.add(layers.Flatten())
        self.Q.add(layers.Dense(64, activation=activations.relu))
        self.Q.add(layers.Dense(64, activation=activations.relu))
        self.Q.add(layers.Dense(6, activation=activations.softmax))        
        self.Q.compile()
        # self.Q.description()
        # Reinforcement client
        self.client = dql.Client(Q_1=self.Q, memory_capacity=1000, batch_size=4, params_path=params_path, memory_path=memory_path)
        self.first_iteration = True

    def make_move(self, state, is_terminal, reward):
        """The move is made based on the given `state`. `is_terminated` and `reward` is saved for future use.

        Args:
            state: The current state of the environment.
            is_terminal (bool): Whether the current state is terminal or not.
            reward (float): The reward received for being in this state.
        """

        self.client.state = state
        self.client.is_terminal = is_terminal
        self.client.reward = reward

        return self.client.choose_action()

    def restart(self):
        """Update the client's target neural network if the episode is right.
        """
        
        self.is_terminal = False
        self.client.episode += 1
        self.client.epsilon = 1 / np.sqrt(self.client.episode * 0.007 + 1)
        if self.client.episode % 500 == 0:
            print(f'Copying network {self.client.episode}')
            self.client.Q_2 = self.client.Q_1