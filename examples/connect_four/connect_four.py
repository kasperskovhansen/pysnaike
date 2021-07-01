"""Play against a trained client using pygame. Most of this code is borrowed by SPR.
"""

import pygame
from game import Connect_four_game
from connect_four_ai import Connect_four_ai
import numpy as np
import os


# Setup pygame
pygame.init()
screen = pygame.display.set_mode((1000, 600))#, pygame.FULLSCREEN)
myfont = pygame.font.SysFont("monospace", 12)
clock = pygame.time.Clock()

# Initialize game variables
done = False
game = Connect_four_game()

ai = Connect_four_ai()

# Load trained network from .npz file
params_path = 'C13_network_params.npz'
if os.path.exists(params_path):
    print(f"Loading training data from external file '{params_path}'...")
    params = np.load(params_path, allow_pickle=True)
    for key in params.files:
        ai.client.Q_1.params[key] = params[key]
    ai.client.Q_2 = ai.client.Q_1

# Load client memory from .npz file
memory_path = 'exp_bank.npz'

if os.path.exists(memory_path):
    print(f"Loading memory from external file '{memory_path}...")
    memory = np.load(memory_path, allow_pickle=True)
    print(type(memory['exp_bank']))
    ai.client.exp_bank.memory = memory['exp_bank'][-ai.client.memory_capacity:].tolist()
    print(len(ai.client.exp_bank.memory))
    print(ai.client.memory_capacity)

ai.client.epsilon = 0.3

first_iter_p1 = True
first_iter_p2 = True
should_break = False
should_draw = True

score = [0,0]

x_off = 130
y_off = 130
size = 70
players = 1

# tile vars
player_colors = [(100,100,100), (255,30,30), (255,255,30)]


def draw_game():
    if not should_draw:
        return
    pygame.draw.rect(screen, (0,0,0), pygame.Rect(0,0,1000,800))
    screen.blit(myfont.render("state: {}".format(game.state), 0, (255,255,255)), (30,30))
    screen.blit(myfont.render("score: {}".format(score), 0, (255,255,255)), (30,50))

    if 0 < game.state <= 4:
        #Draw the board
        for x in range(len(game.grid[0])):
            for y in range(len(game.grid)):
                pygame.draw.rect(screen, player_colors[game.grid[x][y]], pygame.Rect(x_off + size * x * 1.1, y_off + size * y * 1.1, size, size))
        for pos in range(len(game.grid)):
            screen.blit(myfont.render(str(pos+1), 0, (200,200,200)), (x_off + size/2 + size * pos * 1.1, y_off + size/2 + size * -1 * 1.1))
        pos = pygame.mouse.get_pos()
        if x_off <= pos[0] <= x_off + 6*(size*1.1) and y_off > pos[1]:
            x = int((pos[0] - x_off)/(size*1.1))
            pygame.draw.rect(screen, player_colors[game.turn()], pygame.Rect(x_off + size * x * 1.1, y_off + size * (-1) * 1.1, size, size))
    elif game.state == 0:
        screen.blit(myfont.render("Click to start", 0, (255,255,255)), (470,380))
        screen.blit(myfont.render("One player vs ai", 0, (255,255,255)), (170,480))
        screen.blit(myfont.render("Two players", 0, (255,255,255)), (670,480))
    elif game.state == 5:
        screen.blit(myfont.render("Game was won!", 0, (255,255,255)), (470,380))

# Tilstandsmaskine
#Main game loop
while not done:
    for event in pygame.event.get():
        # pygame.mouse.set_pos(pygame.mouse.get_pos()[0] % 10 + 1,10)
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_d):
            should_draw = not should_draw
            print(f"should_draw = {should_draw}")
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            # ai.save()
            done = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()                        
            if game.state == 0:
                if pos[0] < 500:
                    players = 1
                else:
                    players = 2
                game.state = 1

            elif game.state == 1:
                if x_off <= pos[0] <= x_off + 6*(size*1.1) and y_off > pos[1]:
                    x = int((pos[0] - x_off)/(size*1.1))

                game.place(x)
                    
                if game.win():
                    game.state = 5
                else:
                    game.state = 3                             
            elif game.state == 3:
                if players == 2:
                    if x_off <= pos[0] <= x_off + 6*(size*1.1) and y_off > pos[1]:
                        x = int((pos[0] - x_off)/(size*1.1))
                        game.place(x)

                    if game.win() > 0:
                        game.state = 5
                    else:
                        game.state = 1
            elif game.state == 5:                
                game.reset()
                game.state == 1
        if players == 1 and game.state == 3:
            # Player against AI

            state = game.invert_state(game.get_state())
            is_terminal = game.win()           
            if is_terminal == 1:
                reward = 1
            elif is_terminal == 2:
                reward = -1
            if not is_terminal:
                reward = 0.2
            
            if is_terminal is not 0 or game.bad_move:
                game.state = 5
                score[is_terminal - 1] += 1                
            else:
                x = ai.make_move(state, False, 0)
                game.place(x)                     
                game.state = 1

    draw_game()
    pygame.display.flip()
    clock.tick(60)
