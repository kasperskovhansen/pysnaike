"""Setup every version of the client against each other and determine who is best.
"""

from game import Connect_four_game
from connec_four_ai import Connect_four_ai
import os
import numpy as np
import matplotlib.pyplot as plt


clients_paths = "client_bank"

# Setup 
game = Connect_four_game()

def play(c1, c2, score):
    while True:
        is_terminal = 0
        curr_player = 0
        should_break = False
        curr_player = 1
        game.reset()
        game.state = curr_player
        gave_score = False        
        # Check if opponent has been defeated.     
        if score.shape[0] == 50:    
            print("Client defeated opponent. Finding new opponent ...")
            final_score = np.sum(score) / score.shape[0]                    
            return final_score            

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
                    gave_score = True
            if reward != 0:
                pass

            if is_terminal is not 0:
                # print(f'is_terminal: {is_terminal}')
                if should_break:
                    break
                else: should_break = True
            else:
                if curr_player is 1:
                    x = c1.make_move(state, is_terminal, reward)                    
                else:
                    x = c2.make_move(state, is_terminal, reward)
                game.place(x)

            if curr_player == 1:
                curr_player = 2
                game.state = 3
            else:
                curr_player = 1
                game.state = 1


def tournament():
    num_clients = len([x for x in os.listdir(f'{clients_paths}') if x[-19:] == '_network_params.npz'])    
    tournament_score = np.zeros((num_clients, num_clients))
    for c_x in range(num_clients):
        c1 = Connect_four_ai(params_path=f"client_bank/C{c_x}_network_params.npz")
        c1.client.epsilon = 0.05
        for c_y in range(num_clients):
            c2 = Connect_four_ai(params_path=f"client_bank/C{c_y}_network_params.npz")
            c2.client.epsilon = 0.05
            score = play(c1, c2, np.array([]))
            print(f"{c_x} vs {c_y}. {c_x} wins {score*100} %")
            tournament_score[c_x, c_y] = score

    return tournament_score    

load_dont_play = False
if load_dont_play:
    tournament_score = np.load("tournament_score_card.npz", allow_pickle=True)["tournament_score"]
    print("tournament_score")
    print(tournament_score)
else:
    tournament_score = tournament()
    np.savez("tournament_score_card.npz", tournament_score=tournament_score)
tournament_score = tournament_score.T
plt.imshow(tournament_score, cmap="gray")
plt.gca().invert_yaxis()
plt.title("Scoreboard of tournament. Player x VS Player y. White indicates player x victory.")
plt.xlabel("Player x client number")
plt.ylabel("Player y client number")
plt.show()