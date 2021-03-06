from game.snake_objects import Snake
import random
import numpy as np
import pygame
from pygame.locals import *

black = 0, 0, 0
white = 255, 255, 255
grey = 200, 200, 200
red = 255, 0, 0
green = 0, 255, 0
blue = 0, 0, 255


class GameSession:
    def __init__(self, screen_size, grid_size, delay=0, render=True, fix_number=''):
        self.screen_size = np.array(screen_size)
        self.grid_size = np.array(grid_size)
        self.sq_size = self.screen_size // self.grid_size
        self.delay = delay
        self.render = render
        self.number = {'n': 0, 'grid': np.array([0,0]), 'txt': 0}
        self.fix_number = fix_number
        self.alive = True
        # Start pygame screen
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption('Snake')
        if not render:
            pygame.display.iconify()
        # Define font for screen messages
        self.font = pygame.font.SysFont("ubuntumono",  int(self.sq_size[1]*1.5))

    def gen_number(self, n, color):
        self.number['n'] = n + 1
        gap = 0
        if self.fix_number != '':# and self.number['n']==1:
            self.number['grid'] = np.array(self.fix_number)
        else:
            c = [0]
            while c:
                self.number['grid'] = np.array(
                    [random.randint(gap, self.grid_size[0] - 1 - gap), random.randint(gap, self.grid_size[1] - 1 - gap)])
                c = [g for g in self.snake.grid if (self.number['grid'] == g).all()]
        self.number['txt'] = self.font.render(str(self.number['n']), True, color, grey)

    def dir2i(self):
        directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        return [i for i in range(4) if (directions[i] == self.snake.head.dir).all()][0]

    def update_screen(self):
        if self.render:
            self.screen.fill(black)
            number_xy = self.number['grid'] * self.sq_size
            number_xy -= [self.font.get_height() // 6 * (self.number['n'] > 9), self.font.get_height() // 6]
            self.screen.blit(self.number['txt'], number_xy)
            self.snake.plot(self.screen)
            pygame.display.update()

    def start_game(self):
        # Create snake and first number
        self.alive = True
        self.snake = Snake(self.screen_size, self.grid_size, speed=5)  # style "square" or "round"
        self.gen_number(0, white)
        self.update_screen()
        return False, 0, self.alive

    def check_game_event(self, event):
        reward = 0
        s = self.get_state()
        if event == "just_ate":
            reward = 1
            self.gen_number(self.number['n'], white)
        if event == "dead":
            reward = -1
            self.alive = False
        return reward, s

    def get_state(self):
        return self.snake.head.grid[0], self.snake.head.grid[1], self.number['grid'][0], self.number['grid'][1], self.dir2i()

    def get_next_state(self):
        return (self.snake.head.grid[0] + self.snake.head.dir[0], self.snake.head.grid[1] + self.snake.head.dir[1],
                self.number['grid'][0], self.number['grid'][1], self.dir2i())

    def step(self, action, mode='manual'):
        if not self.alive:
            self.start_game()
        action_taken, event = self.snake.update_move(action, self.number['grid'], mode=mode)
        self.update_screen()
        self.check_delay(pygame.key.get_pressed())
        pygame.time.delay(self.delay)
        reward, state = self.check_game_event(event)
        return action_taken, reward, state, self.alive

    def check_delay(self, key):
        for event in pygame.event.get():
            if event.type == KEYDOWN:# and event.key in [K_PAGEUP, K_PAGEDOWN]:
                if event.key == K_PAGEUP:
                    self.delay += 1
                if event.key == K_PAGEDOWN and self.delay >= 1:
                    self.delay -= 1
                print("Delay: %i ms" % self.delay)
                pygame.time.delay(100)

    def plot_msg(self, msg):
        x = (self.screen.get_width() - self.font.size(msg)[0]) / 2
        y = (self.screen.get_height() - self.font.size(msg)[1]) / 2
        msg_surface = self.font.render(msg, True, (250, 0, 0))
        self.screen.blit(msg_surface, (x, y))
        pygame.display.update()

    def check_quit_event(self):
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                return True
            elif event.type == QUIT:
                return True
        return False

    def check_continue_event(self):
        for event in pygame.event.get():
            if event.type == KEYDOWN and event.key == K_SPACE:
                return True
        return False

    def exit(self):
        self.plot_msg("Press Esc. to quit", self.screen, self.font)
        pygame.display.update()
        while not(self.check_quit_event()):
            pass
        pygame.display.quit()