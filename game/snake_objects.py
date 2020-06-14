import pygame
from pygame.locals import *
import numpy as np

black = 0, 0, 0
white = 255, 255, 255
blue = 0, 0, 255

class SnakePart(pygame.sprite.Sprite):
    def __init__(self, grid, dir, style, size):
        super(SnakePart, self).__init__()
        self.dir = dir
        self.grid = grid
        self.size = size
        self.body = pygame.Surface(size)
        self.rect = self.body.get_rect()
        if style == "square":
            self.body.fill(white)
            pygame.draw.rect(self.body, black, self.rect, 1)
        if style == "round":
            pygame.draw.circle(self.body, (0, 255, 0), size//2 , min(size)//2)

        self.pos = self.grid2pos()
        self.rect.topleft = self.pos

    def grid2pos(self):
        return (self.grid * self.size).astype('float64')

    def pos2grid(self):
        return self.rect.topleft // self.size

    def step(self, speed):
        self.pos += speed * self.dir
        self.rect.topleft = self.pos

    def on_grid(self):
        # head or tail are on grid if they fall exactly on a full grid square
        return (self.rect.topleft %  self.size == [0, 0]).all()

    def update_grid(self):
        if self.on_grid():
            self.grid = self.rect.topleft // self.size

    def copy(self, other):
        self.dir = other.dir
        self.grid = other.grid
    # def __eq__(self, other):
    #     pass

class Snake():
    def __init__(self, style, screen_size, grid_size, speed=0.05):
        self.grid = grid_size//2 - [[2, 0], [1, 0], [0, 0]]
        self.direction = np.array([[1, 0], [1, 0], [1, 0]])
        self.body = pygame.sprite.Group()
        self.screen_size = screen_size
        self.sq_size = screen_size // grid_size
        self.style = style
        self.length = 3
        self.length_increase = 3
        self.speed = speed
        self.dir_buffer = self.direction[0]
        # Generate snake parts
        self.tail = SnakePart(self.grid[0], self.direction[0], self.style, self.sq_size)
        for i in range(1, len(self.grid)-1):
            self.body.add(SnakePart(self.grid[i], self.direction[i], self.style, self.sq_size))
        self.head = SnakePart(self.grid[-1], self.direction[-1], self.style, self.sq_size)

    def step(self):
        self.head.step(self.speed)
        if len(self.body) >= self.length:
            self.tail.step(self.speed)

    def out_of_screen(self):
        if self.head.rect.top < 0 or self.head.rect.bottom > self.screen_size[1]\
                or self.head.rect.left < 0 or self.head.rect.right > self.screen_size[0]:
            return True
        return False

    def check_state(self, number_pos):
        # Check dying conditions for new position
        if self.out_of_screen():# or len(pygame.sprite.spritecollide(self.head, self.body, False)) > 1:
            return "dead"
        if ((self.head.grid + self.head.dir) == number_pos).all():
            self.length += self.length_increase
            return "just_ate"
        return ""

    def update_move(self, pressed_keys, number_pos, mode='manual'):
        take_action = self.head.on_grid()   # if head is on grid perform action
        if take_action:
            if mode == 'AI':
                self.head.dir = self.update_dir(pressed_keys, mode=mode) # check input to decide into which square to move
            if mode == 'manual':
                self.dir_buffer = self.update_dir(pressed_keys, mode=mode)
                self.head.dir = self.dir_buffer
            self.body.add(SnakePart(self.head.grid, self.head.dir, self.style, self.sq_size))
            self.grid = np.array([p.grid for p in self.body])
            self.step()                     # Move
        else:
            self.dir_buffer = self.update_dir(pressed_keys, mode=mode)
            self.step()                     # Move
            self.head.update_grid()         # Update head grid if it reaches a new full square
        # Check if tail has reached a new grid square and remove that part of the body
        if self.tail.on_grid() and (self.tail.pos2grid() != self.tail.grid).any():
            overlap = [sp for sp in self.body if sp.rect.colliderect(self.tail.rect)][0]
            self.tail.copy(overlap)
            self.body.remove(overlap)
            self.grid = np.array([p.grid for p in self.body])
        return take_action, self.check_state(number_pos)

    def update_dir(self, pressed_keys, mode='manual'):
        if mode == 'manual':
            if pressed_keys[K_LEFT] and self.head.dir[0] != 1:
                return np.array([-1, 0])
            if pressed_keys[K_RIGHT] and self.head.dir[0] != -1:
                return np.array([1, 0])
            if pressed_keys[K_UP] and self.head.dir[1] != 1:
                return np.array([0, -1])
            if pressed_keys[K_DOWN] and self.head.dir[1] != -1:
                return np.array([0, 1])
            return self.dir_buffer
        if mode == 'AI':
            if pressed_keys == 0 and self.head.dir[0] != 1:
                return np.array([-1, 0])
            if pressed_keys == 1 and self.head.dir[0] != -1:
                return np.array([1, 0])
            if pressed_keys == 2 and self.head.dir[1] != 1:
                return np.array([0, -1])
            if pressed_keys == 3 and self.head.dir[1] != -1:
                return np.array([0, 1])
            return self.head.dir

    def plot(self, screen):
        screen.blit(self.head.body, self.head.rect)
        screen.blit(self.tail.body, self.tail.pos)
        for sp in self.body:
            screen.blit(sp.body, sp.pos)
