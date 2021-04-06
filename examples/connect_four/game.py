

import math
import os.path
import random
import sys

import pygame

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))

import numpy as np
from pysnaike import activations, dql, layers, models


class Connect_four_game:

    def __init__(self):
        self.n_in_a_row = 4
        self.board_dim = 6
        self.reset()
        self.bad_move = 0
        #States:
        # 0: start
        # 1: player 1
        # 2: check for win?
        # 3: player 2
        # 4: check for win?
        # 5: game over
        # 6: reset game        
        

    def reset(self):
        self.grid = [[0 for i in range(self.board_dim)] for i in range(self.board_dim)]
        self.last_move = [0,0]
        self.state = 0
        self.bad_move = 0


    def get_state(self):
        return np.array(self.grid).flatten()

    def place(self, col):
        if self.state == 1 or self.state == 3:
            if self.grid[col][0] == 0:
                #Valid move
                i = 5
                while self.grid[col][i] != 0:                    
                    i -= 1                    
                self.grid[col][i] = self.turn()
                self.state += 1
                self.last_move = [col, i]
            else:
                self.bad_move = self.turn()
                self.state += 1
                return

    def turn(self):
        if self.state < 3:
            return 1
        else:
            return 2

    def win(self):      
        # print(f'bad_move {self.bad_move}')
        if self.bad_move == 1:
            # self.bad_move = 0
            return 2
        elif self.bad_move == 2:
            # self.bad_move = 0
            return 1            

        #Check for win - Tjekker alle muligheder for begge spillere: 3*6*2+3*3*2 = 54 kompinationer for hvert træk - mange flere, hvis brættet havde været større  
        p_x = self.last_move[0]
        p_y = self.last_move[1]
        p = self.grid[p_x][p_y]

        
        if self.vert_horiz([p_x, p_y], p) or self.vert_horiz([p_y, p_x], p, switch=True) or self.diagonal([p_x, p_y], p) or self.diagonal([p_x, p_y], p, switch=True):
            return p
        return 0
        

    def vert_horiz(self, coords, p, switch=False):                
        # Antal kolonner / rækker venstre og højre eller op og ned:
        lu = 3 if coords[0] - self.n_in_a_row - 1 >= 0 else coords[0] # left or up
        rd = min(self.board_dim -1 - coords[0], self.n_in_a_row - 1)   # right or down
        
        in_row = 0
        reset = False
        for i in range(coords[0] - lu, coords[0] + rd+1):       
            field_check = self.grid[i][coords[1]] if not switch else self.grid[coords[1]][i]
            if field_check == p:
                if reset: # Reset
                    in_row = 1
                    reset = False
                else: in_row += 1 # Endnu en på stribe    
            else: reset = True # Reset næste gang et felt passer
            # print("coords: {}, in_row {}".format(coords, in_row))
            if in_row == self.n_in_a_row: return True

    def diagonal(self, coords, p, switch=False):
        # Antal felter skråt ned højre        
        if not switch:
            r = 3 if coords[0] + self.n_in_a_row - 1 < self.board_dim and coords[1] + self.n_in_a_row - 1 < self.board_dim else min(self.board_dim -1 - coords[0], self.board_dim -1 - coords[1]) # right down
            l = 3 if coords[0] - self.n_in_a_row + 1 > 0 and coords[1] - self.n_in_a_row + 1 > 0 else min(coords[0], coords[1]) # left up
        else:
            r = 3 if coords[0] + self.n_in_a_row - 1 < self.board_dim and coords[1] - self.n_in_a_row - 1 > 0 else min(self.board_dim - 1 - coords[0], coords[1], 3) # right up
            l = 3 if coords[0] - self.n_in_a_row + 1 > 0 and coords[1] + self.n_in_a_row - 1 < self.board_dim - 1 else min(coords[0], self.board_dim - 1 - coords[1], 3) # left up        

        in_row = 0
        reset = False
        for i in range(coords[0] - l, coords[0] + r + 1):
            if not switch:
                field_check = self.grid[i][coords[1] - (coords[0] - i)]
            else:
                # print("i" + str(i) + str(p))
                field_check = self.grid[i][coords[1] + coords[0] - i]
            if field_check == p:
                if reset: # Reset
                    in_row = 1
                    reset = False
                else: in_row += 1 # Endnu en på stribe    
            else: reset = True # Reset næste gang et felt passer
            # print("d coords: {}, in_row {}".format([i, coords[1] - (coords[0] - i)], in_row))
            if in_row == self.n_in_a_row: return True
   

    def print_game(self, grid):
        for row in grid:
            row_print = ''
            for field in row:
                if field == 0:
                    color = bcolors.ENDC + bcolors.ENDB
                if field == 1:
                    color = bcolors.BBLUE + bcolors.CWHITE
                elif field == 2:
                    color = bcolors.BRED + bcolors.CWHITE
                row_print += (f'{color} {field} {bcolors.ENDC}')
            print(row_print)

    def invert_state(self, state):
        indices_one = state == 1
        indices_two = state == 2
        state[indices_one] = 2
        state[indices_two] = 1
        return state


class bcolors:
    CBLUE = '\033[94m'
    CRED = '\033[31m'
    BBLUE = '\033[44m'
    BRED = '\033[41m'
    
    ENDC = '\033[0m'
    ENDB = '\033[0;39m'
    
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    CWHITE = '\033[1;37m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'