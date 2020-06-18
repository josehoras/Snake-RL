from game.snake_engine import GameSession
from performance_mon import *
import os.path
import pickle
import random
import pygame
from pygame.locals import *

class TabTD:
    def __init__(self, grid, alpha=0.4, gamma=0.99, r=15, policy='', update_rule='sarsa'):
        self.q = np.full((grid[0], grid[1], grid[0], grid[1], 4, 5), 0.2)
        self.q2 = np.full((grid[0], grid[1], grid[0], grid[1], 4, 5), 0.2)
        self.alpha = alpha
        self.gamma = gamma
        self.r = r
        self.T = 1
        self.e = 0
        if policy=='e-greedy':
            self.policy = self.e_greedy_policy
        else:
            self.policy = self.sofmax_policy
        if update_rule=='sarsa':
            self.update = self.sarsa

        self.performance = {'score':[], 'smooth_score':[], 'moves':[], 'smooth_moves':[]}

    def load(self, file_name):
        self.q, self.alpha, self.gamma, self.r, self.performance = pickle.load(open(file_name, 'rb'), encoding='latin1')

    def save(self, file_name):
        pickle.dump([self.q, self.alpha, self.gamma, self.r, self.performance], open(file_name, 'wb'))

    def terminal(self, state):
        if state[0] < 0 or state[0] > 19 or state[1] < 0 or state[1] > 19:
            return True
        return False

    def sofmax_policy(self, state):
        p = np.exp(self.q[state]/self.T) / sum(np.exp(self.q[state]/self.T))
        a = np.random.choice(range(5), p=p)
        return a

    def e_greedy_policy(self, state):
        if self.e < random.random():
            a = np.argmax(self.q[state])
        else:
            a = random.randint(0, len(self.q[state]))
        return a

    def sarsa(self, s):
        a = self.policy(s)
        return self.q[s][a]

    def expected_sarsa(self, s):
        p = np.exp(self.q[s] / self.T) / sum(np.exp(self.q[s] / self.T))
        return np.sum(p * self.q[s])

    def q_learning(self, s):
        return np.max(self.q[s])

    def update_q(self, s, a, r, s_p):
        r *= self.r
        if not self.terminal(s_p):
            v_p = self.update(s_p)
            self.q[s][a] += self.alpha * (r + self.gamma * v_p - self.q[s][a])
        else:
            self.q[s][a] += self.alpha * r

    def double_q_learning_update(self, s, a, r, s_p):
        r *= self.r
        if not self.terminal(s_p):
            if random.randint(0,1) == 0:
                self.q[s][a] += self.alpha * (r + self.gamma * self.q2[s_p][np.argmax(self.q[s_p])] - self.q[s][a])
            else:
                self.q2[s][a] += self.alpha * (r + self.gamma * self.q[s_p][np.argmax(self.q2[s_p])] - self.q2[s][a])
        else:
            if random.randint(0, 1) == 0:
                self.q[s][a] += self.alpha * r
            else:
                self.q2[s][a] += self.alpha * r


def debug_msg(before='', rew=0, a=False, s=''):
    if before:
        print('_'*80)
        print("Before move")
    else:
        print("After move")

    print("   State:  ",  s, env.snake.head.rect.topleft)
    print("   On grid: ", env.snake.head.on_grid())
    if not before:
        if a: print("   Action taken")
        print("   Reward: ", rew)
        print('_' * 80)
    while not check_continue_event():
        pass


def check_continue_event():
    for event in pygame.event.get():
        if event.type == KEYDOWN and event.key == K_SPACE:
            return True
    return False

# MAIN FUNCTION
screen_size = [400, 400]
grid_size = [20, 20]
fix_pos = [5, 10]
env = GameSession(screen_size, grid_size, fix_number=fix_pos, delay=0, render=False)

# Model name
file_dir = "results/"
file_base = "e-sarsa"
file_post = "_a04g099r20"
file_ext = ".td"
file_name = file_dir + file_base + file_post + file_ext
restart = False
# Maybe load previous model
if os.path.isfile(file_name) and not restart:
    model = TabTD(grid_size)
    model.load(file_name)
else:
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    model = TabTD(grid_size, alpha=0.4, gamma=0.99, r=1, policy='softmax', update_rule='sarsa')

# Main loop
for i in range(len(model.performance['moves']) + 1, len(model.performance['moves']) + 200001):
    _, _, alive = env.start_game()
    level_moves = 0
    number_moves = 0
    score = 1
    while alive:
        level_moves += 1
        number_moves += 1

        # Take action depending on policy
        old_state = env.get_state()
        action = model.policy(old_state)
        # debug_msg(before=True, s=old_state)

        action_taken, reward, new_state, alive = env.step(action, mode='AI')

        # debug_msg(before=False, rew=reward, a=action_taken, s=new_state)
        # Assign reward and update q function
        if action_taken:
            if env.number['n'] > score:
                score = env.number['n']
                number_moves = 0
            # next_state = env.get_next_state()
            # next_state = env.get_state()

            # model.double_q_learning_update(old_state, action, reward, next_state)
            model.update_q(old_state, action, reward, new_state)

        if number_moves == 1000 or env.number['n'] > 50:
            break
    # Book-keeping
    model.performance = update_performance(model.performance, env.number['n'], level_moves)
    print(i, " - score: %i (%.1f), moves: %i (%.0f)" %
          (model.performance['score'][-1], model.performance['smooth_score'][-1],
           model.performance['moves'][-1], model.performance['smooth_moves'][-1]))
    print()
    if i % 50 == 0:
        plot_performance(model.performance, file_dir + file_base + file_post)
        # plot_value(model.q, 5, 10, file_name=file_dir + 'v_map' + file_post + '.png',
        #            alpha=alpha, gamma=gamma, r=r, it=i)
        plot_probs(model, fix_pos[0], fix_pos[1], T=model.T, file_name=file_dir + file_base + file_post)
        model.save(file_name)

env.exit()
