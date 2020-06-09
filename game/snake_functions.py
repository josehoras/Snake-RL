import random
import numpy as np
import pygame
from pygame.locals import *
import matplotlib.pyplot as plt

black = 0, 0, 0
white = 255, 255, 255
grey = 200, 200, 200
red = 255, 0, 0
green = 0, 255, 0
blue = 0, 0, 255


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


def check_pause_event():
    for event in pygame.event.get():
        if event.type == KEYDOWN and event.key == K_SPACE:
            return True
    return False

def check_delay(delay, key):
    if key[K_PAGEUP]:
        delay += 1
        print(delay)
        pygame.time.delay(100)
    if key[K_PAGEDOWN] and delay >=1:
        delay -= 1
        print(delay)
        pygame.time.delay(100)
    return delay

# Functions to generte numbers and plot messages
def generate_number(n, filled_grid, grid_size, font, color=(255, 255, 255), set=0):
    n += 1
    c = [0]
    n_grid = []
    gap = 0
    while c:
        n_grid = np.array([random.randint(gap, grid_size[0]-1-gap), random.randint(gap, grid_size[1]-1-gap)])
        c = [g for g in filled_grid if (n_grid == g).all()]
    if set != 0:
        n_grid = np.array(set)
    n_txt = font.render(str(n), True, white, grey)
    return n, n_grid, n_txt


def plot_msg(msg, screen, font):
    x = (screen.get_width() - font.size(msg)[0]) / 2
    y = (screen.get_height() - font.size(msg)[1]) / 2
    msg_surface = font.render(msg, True, (250, 0, 0))
    screen.blit(msg_surface, (x, y))
    pygame.display.update()


def update_game_screen(screen, snake, number, number_xy, number_txt, font, render=True):
    if render:
        screen.fill(black)
        number_xy -= [font.get_height() // 6 * (number > 9), font.get_height() // 6]
        screen.blit(number_txt, number_xy)
        snake.plot(screen)
        pygame.display.update()