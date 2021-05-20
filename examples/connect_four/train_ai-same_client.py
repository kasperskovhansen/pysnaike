"""Train ai client by literally playing agains the same version of itself.
"""

from game import Connect_four_game
from connec_four_ai import Connect_four_ai


import math
import os.path
import random
import sys
import time

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

import numpy as np
from pysnaike import activations, dql, layers, models

game = Connect_four_game()
ai = Connect_four_ai(params_path='network_params.npz', memory_path='exp_bank.npz')
# ai = Connect_four_ai()

ai.client.episode = 0
while True:
    ai.client.episode += 1
    should_break = False
    is_terminal = 0
    curr_player = 1
    game.reset()
    game.state = curr_player
    ai.restart()
    # ai.client.epsilon = 0
    prev_states = [None, None]
    prev_actions = [None, None]


    while True:
        state = game.get_state()
        if curr_player is not 1:
            state = game.invert_state(state)

        is_terminal = game.win()

        if is_terminal == curr_player:
            # curr_player won
            if game.bad_move:
                reward = 0
            else:
                reward = 1
        elif is_terminal == 0:
            # no one won
            reward = 0
        elif not is_terminal == curr_player:
            # other player won
            reward = -1


        prev_action = prev_actions[curr_player - 1]
        prev_state = prev_states[curr_player - 1]

        if prev_action is not None:
            print(f'TRAIN Eps: {ai.client.epsilon} Curr: {curr_player}, prev_action: {prev_action}, reward {reward}, is_terminal: {is_terminal}, should_break: {should_break}')

            ai.client.add_exp(prev_state, prev_action, reward, state, is_terminal)
            ai.client.train()

        if is_terminal is not 0:
            # print(f'is_terminal: {is_terminal}')
            if should_break:
                break
            else: should_break = True
        else:

            x = ai.make_move(state, is_terminal, reward)
            prev_states[curr_player - 1] = state
            prev_actions[curr_player - 1] = x
            ai.client.action = x
            game.print_game(np.array(game.grid).reshape(6,6).T)
            print(f'Player: {curr_player}, is_terminal: {is_terminal}, action: {x}, should_break: {should_break}')
            game.place(x)

        if curr_player == 1:
            curr_player = 2
            game.state = 3
        else:
            curr_player = 1
            game.state = 1

    if ai.client.episode % 10 == 0:
        print(f'Episode {ai.client.episode}. Epsilon : {ai.client.epsilon}')

        if ai.client.episode % 20 == 0:
            print('Saving ...')
            np.savez(f'network_params.npz', **ai.client.Q_1.params)
            np.savez(f'exp_bank.npz', exp_bank=ai.client.exp_bank.memory)

