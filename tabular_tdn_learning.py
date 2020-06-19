from game.snake_engine import GameSession
from performance_mon import *
import os.path
import pickle
import random


class TabTD_n:
    def __init__(self, grid, alpha=0.4, gamma=0.99, n=1, policy='', update_rule='sarsa'):
        self.q = np.full((grid[0], grid[1], grid[0], grid[1], 4, 5), 0, dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self.T = 1
        self.it = 2
        self.e = 0.5
        self.policy = self.e_greedy_policy
        if update_rule == 'expected_sarsa':
            self.update_q = self.update
            self.update_rule = self.expected_sarsa
        elif update_rule == 'q-learning':
            self.update_q = self.update
            self.update_rule = self.q_learning
        elif update_rule == 'double-q-learning':
            self.q2 = np.full((grid[0], grid[1], grid[0], grid[1], 4, 5), 0.2)
            self.update_q = self.double_q_learning
        else:
            self.update_q = self.update
            self.update_rule = self.sarsa
        self.performance = {'score':[], 'smooth_score':[], 'moves':[], 'smooth_moves':[]}

    def load(self, file_name):
        self.q, self.alpha, self.gamma, self.n, \
        self.it, self.e, self.policy, self.update_q,\
        self.update_rule,self.performance = pickle.load(open(file_name, 'rb'), encoding='latin1')
        print('E-greedy: %.3f' % self.e)

    def save(self, file_name):
        pickle.dump([self.q, self.alpha, self.gamma, self.n,
                     self.it, self.e, self.policy, self.update_q,
                     self.update_rule, self.performance], open(file_name, 'wb'))

    def terminal(self, state):
        if state[0] < 0 or state[0] > 19 or state[1] < 0 or state[1] > 19:
            return True
        return False

    def it_plus(self):
        self.it += 1
        self.e = 1 / self.it
        if self.e < 0.01: self.e = 0
        print('E-greedy: %.3f' % self.e)

    def e_greedy_policy(self, state):
        if self.e < random.random():
            a = np.argmax(self.q[state])
        else:
            a = random.randint(0, len(self.q[state])-1)
        return a

    def e_expected(self, state):
        p = np.full_like(state, self.e * (1/len(state)), dtype=float)
        p[np.argmax(self.q[state])] += 1 - self.e
        return p

    def sarsa(self, s):
        a = self.policy(s)
        return self.q[s][a]

    def expected_sarsa(self, s):
        if self.policy == self.e_greedy_policy:
            p = self.e_expected(s)
        else:
            p = np.exp(self.q[s] / self.T) / sum(np.exp(self.q[s] / self.T))
        return np.sum(p * self.q[s])

    def q_learning(self, s):
        return np.max(self.q[s])

    def update(self, S, A, R, tau, T):
        G = 0
        for i in range(tau + 1, min(tau + self.n, T) + 1):
            G += self.gamma ** (i - tau - 1) * R[i]
            # print(i,': ', self.gamma**(i-tau-1), '  R: ', R[i])
        if not self.terminal(S[tau + self.n]): #tau + N < T+1:
            # print(tau + N, ': ', self.gamma**N)
            G += self.gamma ** self.n * self.update_rule(S[tau + self.n])
        # print('G: %.2f' % G)
        # print("Inc.: %.2f at state %s, action %i" % (self.alpha * (G - self.q[S[tau]][A[tau]]), S[tau], A[tau] ))
        self.q[S[tau]][A[tau]] += self.alpha * (G - self.q[S[tau]][A[tau]])

    def double_q_learning(self, s, a, r, s_p):
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


def debug_msg(before=True, rew=0):
    if before:
        print('_'*80)
        print("Before move")
    else:
        print("After move")
    print("   State:  ",  s, env.snake.head.rect.topleft)
    # print("   On grid: ", env.snake.head.on_grid())
    if not before:
        # if a: print("   Action taken")
        print("   Reward: ", rew)
        print('_' * 80)


# MAIN FUNCTION
screen_size = [400, 400]
grid_size = [20, 20]
fix_pos = [5, 10]
env = GameSession(screen_size, grid_size, fix_number='', delay=0, render=False)

# Model name
file_dir = "results/"
file_base = "sarsa"
file_post = "_a04g09e10_full"
file_ext = ".tdn"
file_name = file_dir + file_base + file_post + file_ext
restart = True
batch = 1000
# Maybe load previous model
if os.path.isfile(file_name) and not restart:
    model = TabTD_n(grid_size)
    model.load(file_name)
else:
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    model = TabTD_n(grid_size, alpha=0.4, gamma=0.8, n=10, policy='e-greedy', update_rule='sarsa')


# Main loop
for i in range(len(model.performance['moves']) + 1, len(model.performance['moves']) + 200001):
    _, _, alive = env.start_game()
    if i % batch == 0:
        model.it_plus()
    level_moves = 0
    number_moves = 0
    score = 1
    T = 1
    S = [env.get_state()]
    A = []
    R = [0]

    t = 0
    tau = t - model.n + 1
    while tau < (T-1):
        action_taken = True
        if t < T:
            # Take action depending on policy
            old_state = env.get_state()
            action = model.policy(old_state)
            action_taken, reward, new_state, alive = env.step(action, mode='AI')
        if action_taken:
            level_moves += 1
            number_moves += 1
            # debug_msg(before=True, s=old_state)
            # debug_msg(before=False, rew=reward, a=action_taken, s=new_state)
            S.append(new_state)
            A.append(action)
            R.append(reward)
            tau = t - model.n + 1
            if tau >= 0:
                model.update_q(S, A, R, tau, T)
            if env.number['n'] > score:
                score = env.number['n']
                number_moves = 0
            t += 1
            if alive and env.number['n'] <= 50 and number_moves < 250:
                T += 1
            # while not env.check_continue_event():
            #     pass

    # Book-keeping
    # print("%06.3f, %06.3f" % (np.min(model.q[:, :, 5, 10]), np.max(model.q[:, :, 5, 10])))
    model.performance = update_performance(model.performance, env.number['n'], level_moves)
    print(i, " - score: %i (%.1f), moves: %i (%.0f)" %
          (model.performance['score'][-1], model.performance['smooth_score'][-1],
           model.performance['moves'][-1], model.performance['smooth_moves'][-1]))
    print()
    if i % 500 == 0:
        plot_performance(model.performance, file_dir + file_base + file_post)
        # plot_value(model.q, 5, 10, file_name=file_dir + 'v_map' + file_post + '.png',
        #            alpha=alpha, gamma=gamma, r=r, it=i)
        plot_probs(model, fix_pos[0], fix_pos[1], file_name=file_dir + file_base + file_post)
        model.save(file_name)

env.exit()
