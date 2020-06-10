from game.snake_engine import GameSession
import numpy as np
import pygame


screen_size = np.array([400, 400])
grid_size = np.array([20, 20])
env = GameSession(screen_size, grid_size, delay=30)#, fix_number = [4,4])

env.start_game()

for i in range(50):
    action = pygame.key.get_pressed()
    env.step(action)

env.exit()
