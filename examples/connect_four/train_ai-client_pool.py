"""Train client by playing against previous versions of itself.
"""

from game import Connect_four_game
from connec_four_ai import Connect_four_ai
import numpy as np


clients_paths = "client_bank"

def save_ai_load_opp(ai, num_clients):
    """Decide who to play against based on difficulty. 
    Currently not based on difficulty.

    Returns:
        Connect_four_ai: Returns an AI with the correct set of parameters loaded.
    """

    # Saving first version of client to client-bank
    np.savez(f'{clients_paths}/C{num_clients}_network_params.npz', **ai.client.Q_1.params)    

    # Loading opponent    
    opp = Connect_four_ai()
    client = np.random.randint(0, num_clients + 1)
    opp.client.Q_1.load_params(f'{clients_paths}/C{client}_network_params.npz')
    return opp, client

# Setup 
game = Connect_four_game()
num_clients = 0

# AI to be trained
ai = Connect_four_ai()

opp, opp_id = save_ai_load_opp(ai, num_clients)
num_clients += 1


score = np.array([0]*50)    # One for AI and zero for opponent victory

ai.client.episode = 0
while True:
    ai.client.episode += 1
    should_break = False
    is_terminal = 0
    curr_player = np.random.randint(1, 2 + 1)
    game.reset()
    game.state = curr_player
    ai.restart()
    # ai.client.epsilon = 0
    prev_states = [None, None]
    prev_actions = [None, None]
    gave_score = False

    # Check if opponent has been defeated.     
    if score.shape[0] == 50 and np.sum(score) / score.shape[0] > 0.75:    
        print("Client defeated opponent. Finding new opponent ...")
        opp, opp_id = save_ai_load_opp(ai, num_clients)
        score = np.array([0]*50)
        num_clients += 1
        ai.client.episode = 0
        ai.client.reset_memory() # Don't bring memories from previous games.

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
        
        if not gave_score:            
            if reward > 0:
                score = np.append(score, curr_player % 2)                
            elif reward < 0:
                score = np.append(score, (curr_player - 1) % 2)
            if reward != 0:
                game.print_game(np.array(game.grid).reshape(6,6).T)
                print(f'C vs {opp_id}, Score: {np.sum(score) / score.shape[0]}. Player {curr_player} won. Eps: {ai.client.epsilon.round(2)}')
                score = score[1:]
                gave_score = True                    

        prev_action = prev_actions[curr_player - 1]
        prev_state = prev_states[curr_player - 1]

        if prev_action is not None:
            # Debug print:
            # print(f'TRAIN Eps: {ai.client.epsilon} Curr: {curr_player}, prev_action: {prev_action}, reward {reward}, is_terminal: {is_terminal}, should_break: {should_break}')
            ai.client.add_exp(prev_state, prev_action, reward, state, is_terminal)
            ai.client.train()

        if is_terminal is not 0:
            if should_break:
                break
            else: should_break = True
        else:
            if curr_player is 1:
                x = ai.make_move(state - 1, is_terminal, reward)
                prev_states[curr_player - 1] = state
                prev_actions[curr_player - 1] = x
                ai.client.action = x
            else:
                x = opp.make_move(state, is_terminal, reward)
            # game.print_game(np.array(game.grid).reshape(6,6).T)   # Print every move to console colored.
            # print(f'Player: {curr_player}, is_terminal: {is_terminal}, action: {x}, should_break: {should_break}')   # More debug print            
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
