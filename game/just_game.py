from game.snake_engine import GameSession

import pygame
from pygame.locals import *

def check_quit_event():
    for event in pygame.event.get():
        if event.type == KEYDOWN and event.key == K_ESCAPE:
            return True
        elif event.type == QUIT:
            return True
    return False

def check_continue_event():
    for event in pygame.event.get():
        if event.type == KEYDOWN and event.key == K_SPACE:
            return True
    return False

def key2action(key):
    if key[K_LEFT]:
        return "left"
    if key[K_RIGHT]:
        return "right"
    if key[K_UP]:
        return "up"
    if key[K_DOWN]:
        return "down"

screen_size = [400, 400]
grid_size = [20, 20]
env = GameSession(screen_size, grid_size, delay=40, fix_number=[12,10])
env.start_game()

while not(check_quit_event()):
    # print('-'*80)
    # print("Before move")
    # old_state = env.get_state()
    # next_state = env.get_next_state()
    # print("Old:  ", old_state, env.snake.head.rect.topleft)
    # print("On grid: ", env.snake.head.on_grid())
    # print("Next: ", next_state)
    # while not check_continue_event():
    #     pass

    pkey = pygame.key.get_pressed()
    action_taken, reward, alive = env.step(pkey)

    # print(key2action(pkey))
    # print("After move")
    # print("Reward: ", reward)
    # old_state = env.get_state()
    # next_state = env.get_next_state()
    # print("Old:  ", old_state, env.snake.head.rect.topleft)
    # print("On grid: ", env.snake.head.on_grid())
    # print("Next: ", next_state)
    # print('-' * 80)
    # while not check_continue_event():
    #     pass

# env.exit()
