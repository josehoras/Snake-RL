from game.snake_engine import GameSession
import numpy as np
import pygame
from pygame.locals import *

def check_quit_event():
    for event in pygame.event.get():
        if event.type == KEYDOWN and event.key == K_ESCAPE:
            return True
        elif event.type == QUIT:
            return True
    return False


screen_size = np.array([400, 400])
grid_size = np.array([20, 20])
env = GameSession(screen_size, grid_size, delay=40)#, fix_number = [4,4])
# pygame.init()
# screen = pygame.display.set_mode(screen_size)
env.start_game()

while not(check_quit_event()):
    # action = pygame.key.get_pressed()
    action_taken, reward, alive = env.step(pygame.key.get_pressed())
    # new_state = (env.snake.head.grid[0] + env.snake.head.dir[0], env.snake.head.grid[1] + env.snake.head.dir[1],
    #              env.number['grid'][0], env.number['grid'][1], dir2i(env.snake.head.dir))
    # print(alive)
    # print(new_state)

    # print('key: ', np.argmax(pygame.key.get_pressed()), np.sum(pygame.key.get_pressed()))
    # print('action: ', np.argmax(action), np.sum(action))

# env.exit()
