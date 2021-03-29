"""An example of using the Deep Q Learning algorithm to train a car to drive itself.
A DQL `Client` is added to the `Car` object. The client decides the car's action from the car's `update`-function, 
and experience is added to the exp bank, and the client is trained from the `update`-function of the `Game`-class.

Primarily inspired by this article series:
https://towardsdatascience.com/self-learning-ai-agents-part-ii-deep-q-learning-b5ac60c3f47
"""


import math
import os.path
import random
import sys

import pygame

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
from pysnaike import activations, dql, layers, models


# Setup pygame
pygame.init()
screen = pygame.display.set_mode((600, 400)) #, pygame.FULLSCREEN)
myfont = pygame.font.SysFont("monospace", 12)
clock = pygame.time.Clock()
bane = pygame.image.load('images/bane2.png')
should_draw = True
# Initialize game variables
done = False
tilstand = 1

class Car():
    def __init__(self):
        self.pos = [0,250]
        self.dir = 0
        self.steer = 0

        self.client = dql.Client(num_inputs=5, num_actions=20)

    def update(self, state, is_terminal, reward):
        """The car is updated based on the given `state`. `is_terminated` and `reward` is saved for future use.

        Args:
            state: The current state of the environment.
            is_terminal (bool): Whether the current state is terminal or not.
            reward (float): The reward received for being in this state.
        """

        self.client.state = state
        self.client.is_terminal = is_terminal
        self.client.reward = reward

        self.client.action = self.client.choose_action()
        self.apply_action(self.client.action)

        if is_terminal:
            self.restart()

    def apply_action(self, action):
        """Apply an `action`.

        Args:
            action (int): The number of the action to be executed.
        """

        L = 2
        k = 1
        x = action - self.client.num_actions // 2
        steer = (L/(1+math.exp(-k*x)))-1
        self.dir += steer
        if self.dir > 1:
            self.dir = 1
        if self.dir < -1:
            self.dir = -1
        self.pos[0] += math.cos(self.dir) * 3
        self.pos[1] += math.sin(self.dir) * 3


    def restart(self):
        """Move the car to the start state and update the client's target neural network if the episode is right.
        """

        self.pos = [0,250]
        self.dir = 0
        self.is_terminal = False
        self.client.episode += 1
        self.client.epsilon = 1 / np.sqrt(self.client.episode * 0.1 + 1)
        if self.client.episode % 10 == 0:
            print(f'Copying network {self.client.episode}')
            self.client.Q_2 = self.client.Q_1


class Game():
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.car = Car()
        self.first_iter = True

    def update(self):
        """Retrieve and evaluate this state and train and update the client.
        """

        new_state = self.get_state()
        reward, is_terminal = self.evaluate_state()

        if not self.first_iter:
            self.car.client.add_exp(new_state, is_terminal)
            self.car.client.train()

        self.car.update(new_state, is_terminal, reward)

        self.first_iter = False

    def get_state(self):
        """Get the state from the environment.

        Returns:
            list: The current state.
        """

        inputs = []
        game.points = []
        for v in [-math.pi/3,-math.pi/6,0,math.pi/6,math.pi/3]:
            l = 1
            x = self.car.pos[0] + l * math.cos(v + self.car.dir)
            y = self.car.pos[1] + l * math.sin(v + self.car.dir)
            c = bane.get_at((int(x),int(y)))
            while c == (0,0,0,255) and l < 150:
                x = self.car.pos[0] + l * math.cos(v + self.car.dir)
                y = self.car.pos[1] + l * math.sin(v + self.car.dir)
                try:
                    c = bane.get_at((int(x),int(y)))
                except:
                    c = (0,0,0,255)
                l += 1

            game.points.append((x,y))
            inputs.append(l/150)
        return inputs

    def evaluate_state(self):
        """Calculate rewards and whether the state is terminal or not.

        Returns:
            list: The reward and `is_terminal`.
        """

        if bane.get_at((int(self.car.pos[0]),int(self.car.pos[1]))) != (0,0,0,255):
            # Loose
            reward = -1
            is_terminal = True
        elif self.car.pos[0] > 550:
            # Win
            reward = 1
            is_terminal = True
        else:
            # Survive
            reward = 0.001
            is_terminal = False

        return reward, is_terminal

game = Game(500,500)

def draw_game():
    if not should_draw:
        return
    pygame.draw.rect(screen, (0,0,0), pygame.Rect(0,0,800,600))
    screen.blit(bane, [0, 0])


    pygame.draw.ellipse(screen, (255,255,255), pygame.Rect(game.car.pos[0]-6, game.car.pos[1]-6,12,12))


    # screen.blit(myfont.render("step: {}".format(game.car.t), 0, (255,255,255)), (20,20))
    screen.blit(myfont.render("episode: {}".format(game.car.client.episode), 0, (255,255,255)), (20,40))
    screen.blit(myfont.render("epsilon: {}".format(game.car.client.epsilon), 0, (255,255,255)), (20,60))

    for p in game.points:
        pygame.draw.ellipse(screen, (200,100,200), pygame.Rect(p[0],p[1], 5, 5))


def output_logic(tilstand):
    if tilstand == 1:
        draw_game()
    elif tilstand == 0:
        draw_menu()

def draw_menu():
    pass


#Main game loop
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            done = True
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            bane = pygame.image.load('images/bane1.png')
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_w):
            bane = pygame.image.load('images/bane2.png')
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_d):
            should_draw = not should_draw
            print(f"should_draw = {should_draw}")

        #Håndtering af input fra mus
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
    game.update()

    output_logic(tilstand)

    #pygame kommandoer til at vise grafikken og opdatere 60 gange i sekundet.
    pygame.display.flip()
    # clock.tick(60)
