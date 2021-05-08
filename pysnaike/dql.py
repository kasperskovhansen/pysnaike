"""Classes related to the Deep Q Network algorithm.
Perhaps use name Agent for a class.
"""


import random
import numpy as np
import os.path
from pysnaike import activations, layers, models

class Client():
    """Deep Q Learning client who acts on states and learns by its mistakes.
    """

    def __init__(self, epsilon=0, gamma=0.9, memory_capacity=10000, batch_size=10, learning_rate=0.01, Q_1=None, num_inputs=None, num_actions=None, params_path=None, memory_path=None):
        
        self.state = None
        self.is_terminal = False
        self.action = None
        self.reward = 0

        self.episode = 0

        # DQL
        self.epsilon = epsilon          # Exploitation or exploration
        self.gamma = gamma              # Discount factor
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_inputs = num_inputs
        self.num_actions = num_actions

        # Replay memory
        self.exp_bank = self.reset_memory()
        if memory_path and os.path.exists(memory_path):
            memory = np.load(memory_path, allow_pickle=True)            
            self.exp_bank.memory = memory['exp_bank'].tolist()
            print("loaded memory_bank")
            # print(self.exp_bank.memory)
            # print(self.exp_bank.memory.count(None))
            # print(len(self.exp_bank.memory))
            # breakpoint()

        # Action-value function
        if Q_1 is not None:
            self.Q_1 = Q_1
            self.num_inputs = self.Q_1.layers[0].input_shape
            self.num_actions = self.Q_1.layers[-1].output_shape
        elif num_actions is not None:
            self.Q_1 = models.Sequential()
            self.Q_1.add(layers.Dense(self.num_inputs, activation=activations.relu))
            self.Q_1.add(layers.Dense(10, activation=activations.sigmoid))
            self.Q_1.add(layers.Dense(20, activation=activations.sigmoid))
            self.Q_1.add(layers.Dense(self.num_actions, activation=activations.softmax))
            self.Q_1.compile()
        else:
            raise ValueError()

        if params_path:
            self.Q_1.load_params(params_path)
            self.Q_2 = self.Q_1

        self.Q_2 = self.Q_1

    def reset_memory(self):
        return ExpReplay(self.memory_capacity)

    def choose_action(self):
        """Choose an action from the current state.

        Returns:
            int: The number of the chosen action.
        """
        
        # Eploration or exploitation
        p = random.random()
        action = 0
        if p <= self.epsilon:
            # Random action
            action = random.randint(0, self.num_actions - 1)
        else:
            # Choose best action
            action = self.max_action(self.state)
        return action

    def max_action(self, state):
        """Find the best action.

        Args:
            state: Current state.

        Returns:
            int: The number of the chosen action.
        """

        return self.argmax(self.Q_1.forward_pass(state))
    
    def argmax(self, iterable):
        """Find index of the maximum value in a list.

        Args:
            iterable (list): An iterable with integers.

        Returns:
            int: Index of the maximum value in the list.
        """

        return max(enumerate(iterable), key=lambda x: x[1])[0]

    def add_exp(self, state, action, reward, new_state, new_is_terminal):
        """Add experience to the experience bank.

        Args:
            state
            action
            reward
            new_state: The new state.
            new_is_terminal (bool): Whether `new_state` is terminal.
        """

        exp = [self.state,
                self.action,
                self.reward,
                (new_state, new_is_terminal)]

        self.exp_bank.push(exp)

    def train(self):
        """Train the model based on samples from the experience bank.
        """

        samples = self.exp_bank.sample(self.batch_size)
        # [state, action, reward, (new_state, is_terminal)]
        outputs = None
        targets = None          
        if not samples or self.exp_bank.memory.count(None) is not 0:
            return            

        for sample in samples:
            if sample[3][1]:
                # new_state is terminal
                target = sample[2]
            else:
                target = sample[2] + self.gamma * max(self.Q_2.forward_pass(sample[3][0]))

            new_outputs = self.Q_1.forward_pass(sample[0])
            new_targets = new_outputs.copy()
            new_targets[sample[1]] = target

            if targets is not None:
                targets = np.vstack((targets, new_targets))
                outputs = np.vstack((outputs, new_outputs))
            else:
                targets = np.array([new_targets])
                outputs = np.array([new_outputs])

        avg_new_weights = {}
        for output, target in zip(outputs, targets):
            new_weights = self.Q_1.backward_pass(output, target)

            if not avg_new_weights: avg_new_weights = new_weights
            else:
                for key, value in new_weights.items():
                    avg_new_weights[key] -= self.learning_rate * value
            self.Q_1.mini_b_update_network_params(new_weights, len(outputs))


class Transition():
    """A transition used in experience replay.
    """

    def __init__(self, state, action, reward, new_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state

    def __str__(self):
        return str((self.state, self.action, self.reward, self.new_state))


class ExpReplay():
    """Experience replay used in training a DQL client.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [None] * self.capacity
        self.position = 0

    def push(self, memory):
        """Add a transition to the memory
        """

        # while len(self.memory) < self.capacity:
        #     self.memory.append(None)
        # self.memory[self.position] = Transition(**kvargs)                
        self.memory.pop(0)
        # self.memory[self.position]
        self.memory.append(memory)
        # self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Retrieve a sample from the memory of size `batch_size`

        Args:
            batch_size (int): Size of the sample to return.

        Returns:
            list: The samples retrieved.
        """

        return random.sample(self.memory, batch_size)

    def __len__(self):        
        return len(self.memory) - self.memory.count(None)

    def __str__(self) -> str:
        return f"<ExpReplay class object with capacity {self.capacity} and {len(self)} item(s) stored>"