from game import Connect_four_game
from connec_four_ai import Connect_four_ai

import math
import os.path
import random
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

import numpy as np
from pysnaike import activations, dql, layers, models



game = Connect_four_game()

ai_1 = Connect_four_ai()

# Hukommelsen bliver ikke gemt mellem hvert spil...

# Load trained network from .npz file
# params_path = 'network_params_p1.npz'
# if os.path.exists(params_path):
#     print(f"Loading training data from external file '{params_path}'...")
#     params = np.load(params_path)
#     for key in params.files:
#         ai_1.client.Q_1.params[key] = params[key]
#     ai_1.client.Q_2 = ai_1.client.Q_1


# # Load client memory from .npz file
# memory_path = 'exp_bank_p1.npz'

# if os.path.exists(memory_path):
#     print('Loading memory from external file...')
#     memory = np.load(memory_path)
#     ai_1.client.exp_bank = memory['exp_bank'][-ai_1.client.memory_capacity:-1]    

ai_2 = ai_1
ais = [ai_1, ai_2]




should_break = False

score = [0, 0]

episode = 0
while episode < 1000000:
    episode += 1
    ai_1.client.episode = episode
    should_break = False
    is_terminal = 0
    curr_player = 1
    game.state = curr_player
    game.reset()
    ai_1.restart()
    ai_2.restart()

    
    epsilon = 1 / np.sqrt(episode * 0.1 + 1)
    ai_1.client.episode, ai_2.client.episode = episode, episode
    ai_1.client.epsilon, ai_2.client.epsilon = epsilon, epsilon
        
    while True:        
        ai = ais[curr_player - 1]
        
        state = game.get_state()
        is_terminal = game.win()

        if is_terminal == 1:
            reward = 1
        elif is_terminal == 2:
            reward = -1
        if not is_terminal:
            reward = 0.2
        
        if not ai.first_iteration:
            ai.client.add_exp(state, is_terminal)
            print(len(ai.client.exp_bank), episode)
            ai.client.train()
        else: ai.first_iteration = False

        if is_terminal is not 0:
            score[is_terminal - 1] += 1

            if score[is_terminal - 1] % 100 == 0:
                print(f'Saving p{is_terminal}')
                if is_terminal == 1:
                    np.savez(f'network_params_p{is_terminal}.npz', **ai.client.Q_1.params)          
                    np.savez(f'exp_bank_p{is_terminal}.npz', exp_bank=ai.client.exp_bank.memory)    
                    for exp in ai.client.exp_bank.memory:
                        print(f"exp: ")
                        print(exp)   
                    # print(ai.client.exp_bank.memory)   

            if should_break:
                break
            else: should_break = True
        else:
            x = ai.make_move(state, is_terminal, reward)
            ai.client.action = x            
            game.place(x)            

        if curr_player == 1: 
            curr_player = 2
            game.state = 3
        else: 
            curr_player = 1
            game.state = 1

    if episode % 10 == 0:
        print(f'Episode {episode}. Epsilon : {ai_1.client.epsilon}. Score {score}')

