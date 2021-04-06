import math
import os.path
import random
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

import numpy as np
from pysnaike import activations, dql, layers, models

class Connect_four_ai():
    def __init__(self, params_path=None, memory_path=None):
        # AI client
        self.Q = models.Sequential()
        self.Q.add(layers.Dense(36, activation=activations.input))
        self.Q.add(layers.Dense(30, activation=activations.relu))
        self.Q.add(layers.Dense(30, activation=activations.relu))
        self.Q.add(layers.Dense(18, activation=activations.relu))
        self.Q.add(layers.Dense(6, activation=activations.softmax))
        self.Q.compile()

        self.client = dql.Client(Q_1=self.Q, memory_capacity=10000, batch_size=16, params_path=params_path, memory_path=memory_path)

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
        self.client.epsilon = 1 / np.sqrt(self.client.episode * 0.0001 + 1)
        if self.client.episode % 500 == 0:
            print(f'Copying network {self.client.episode}')
            self.client.Q_2 = self.client.Q_1


# from random import randint
# import pickle
# from numpy import linalg

# class Connect_four_ai():

#     def __init__(self):
#         self.training_data = set()
#         self.add_data(([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],3))
#         self.load()
#         print("Loaded {} training examples".format(len(self.training_data)))

#     def make_move(self, game):
#         n = self.find_nearest(game.grid)
#         return n[1]

#     def add_data(self, move):
#         v = [item for sublist in move[0] for item in sublist]
#         self.training_data.add((tuple(v),move[1]))
#         print("{} training examples!".format(len(self.training_data)))

#     def load(self):
#         self.training_data = pickle.load(open("training_data.p", "rb"))

#     def save(self):
#         pickle.dump(self.training_data, open("training_data.p", "wb"))

#     def find_nearest(self, state):
#         flat = [item for sublist in state for item in sublist]
#         n = min(self.training_data, key=lambda x:linalg.norm([flat[i] - x[0][i] for i in range(len(flat))]))
#         return n
