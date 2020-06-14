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

    def update(self, speed):
        self.pos += speed * self.dir
        self.rect.topleft = self.pos

    def on_grid(self):
        # head or tail are on grid if they fall exactly on a full grid square
        return (self.rect.topleft %  self.size == [0, 0]).all()

    def update_grid(self):
        if self.onGrid():
            self.grid = self.rect.topleft // self.size

    # def __eq__(self, other):
    #     pass

class Snake():
    def __init__(self, style, screen_size, grid_size, speed=0.05):
        # super(Snake, self).__init__()
        self.grid = grid_size//2 - [[2, 0], [1, 0], [0, 0]]
        self.direction = np.array([[1, 0], [1, 0], [1, 0]])
        self.body = pygame.sprite.Group()
        self.screen_size = screen_size
        self.sq_size = screen_size // grid_size
        self.style = style
        self.next_dir = self.direction[-1]
        self.length = 3
        self.length_increase = 3
        self.speed = speed
        self.state = "alive"
        # Generate snake parts
        self.tail = SnakePart(self.grid[0], self.direction[0], self.style, self.sq_size)
        for i in range(1, len(self.grid)-1):
            self.body.add(SnakePart(self.grid[i], self.direction[i], self.style, self.sq_size))
        self.head = SnakePart(self.grid[-1], self.direction[-1], self.style, self.sq_size)

    def new_square(self, part):
        return part.on_grid() and \
                (part.rect.topleft // self.sq_size != part.grid).any()

    def step(self):
        self.head.update(self.speed)
        if len(self.body) >= self.length:
            self.tail.update(self.speed)

    def out_of_screen(self):
        if self.head.rect.top < 0 or self.head.rect.bottom > self.screen_size[1]\
                or self.head.rect.left < 0 or self.head.rect.right > self.screen_size[0]:
            return True
        return False

    def update_move(self, pressed_keys, number_pos, mode='manual'):
        self.state = "alive"
        action_taken = False
        # Check if head is in the middle of a square. If so just perform its move
        if not self.head.on_grid():
            self.step()                     # Move
            # Check if we have reached a new full square
            if self.new_square(self.head):
                self.head.grid = self.head.rect.topleft // self.sq_size
        else:
            # If head is in a full square, check input to decide into which square to move next
            self.update_dir(pressed_keys, mode=mode)
            action_taken = True
            # apply new direction to head and add new body part
            self.head.dir = self.next_dir
            self.body.add(SnakePart(self.head.grid, self.head.dir, self.style, self.sq_size))
            self.grid = np.append(self.grid, self.head.grid)
            # Move
            self.step()
            # Check dying conditions for new position
            if self.out_of_screen():
                self.state = "dead"
            if len(pygame.sprite.spritecollide(self.head, self.body, False)) > 1:
                self.state = "dead"
            if ((self.head.grid + self.head.dir) == number_pos).all():
                self.length += self.length_increase
                self.state = "just_ate"
        # Check if tail has reached a new grid square and remove that part of the body
        if self.new_square(self.tail):
            to_del = [sp for sp in self.body if sp.rect.colliderect(self.tail.rect)][0]
            self.tail.dir = to_del.dir
            self.tail.grid = to_del.grid
            self.body.remove(to_del)
            self.grid = np.array([p.grid for p in self.body])
        return action_taken


    def old_update_move(self, pressed_keys, number_pos, mode='manual'):
        # self.state = "alive"
        # Move the head and, if the length is ok, the tail
        self.head.update(self.speed)
        if len(self) >= self.length:
            self.tail.update(self.speed)
        # Check if head has reached a new grid square and add a new part to body
        if self.new_square(self.head):
            # Check user input
            self.update_dir(pressed_keys, mode=mode)

            self.head.grid = self.head.rect.topleft // self.sq_size
            self.head.dir = self.next_dir
            self.add(SnakePart(self.head.grid, self.head.dir, self.style, self.sq_size))
            self.grid = np.append(self.grid, [self.head.grid], axis=0)
            if (self.head.grid == number_pos).all():
                self.length += self.length_increase
                self.state = "just_ate"
        # Check if tail has reached a new grid square and remove that part of the body
        if self.new_square(self.tail):
            to_del = [sp for sp in self if (sp.grid == self.tail.rect.topleft // self.sq_size).all()][0]
            self.grid = self.grid[[(g != self.tail.grid).any() for g in self.grid]]
            self.tail.dir = to_del.dir
            self.tail.grid = to_del.grid
            self.remove(to_del)

        # Check dying conditions
        if self.head.rect.top < 0 or self.head.rect.bottom > self.screen_size[1]:
            self.state = "dead"
        if self.head.rect.left < 0 or self.head.rect.right > self.screen_size[0]:
            self.state = "dead"
        if len(pygame.sprite.spritecollide(self.head, self, False)) > 1:
            self.state = "dead"


    def update_dir(self, pressed_keys, mode='manual'):
        if mode == 'manual':
            if pressed_keys[K_LEFT] and self.head.dir[0] != 1:
                self.next_dir = np.array([-1, 0])
            if pressed_keys[K_RIGHT] and self.head.dir[0] != -1:
                self.next_dir = np.array([1, 0])
            if pressed_keys[K_UP] and self.head.dir[1] != 1:
                self.next_dir = np.array([0, -1])
            if pressed_keys[K_DOWN] and self.head.dir[1] != -1:
                self.next_dir = np.array([0, 1])
        if mode == 'AI':
            if pressed_keys == 0 and self.head.dir[0] != 1:
                self.next_dir = np.array([-1, 0])
            if pressed_keys == 1 and self.head.dir[0] != -1:
                self.next_dir = np.array([1, 0])
            if pressed_keys == 2 and self.head.dir[1] != 1:
                self.next_dir = np.array([0, -1])
            if pressed_keys == 3 and self.head.dir[1] != -1:
                self.next_dir = np.array([0, 1])

    def plot(self, screen):
        screen.blit(self.head.body, self.head.rect)
        screen.blit(self.tail.body, self.tail.pos)
        for sp in self.body:
            screen.blit(sp.body, sp.pos)
