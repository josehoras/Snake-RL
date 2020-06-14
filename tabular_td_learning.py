from game.snake_engine import GameSession
from performance_mon import *
import os.path
import pickle


class TabTD:
    def __init__(self, grid, alpha = 0.4, gamma = .99, r = 15):
        self.q = np.full((grid[0], grid[1], grid[0], grid[1], 4, 5), 0.2)
        self.alpha = alpha
        self.gamma = gamma
        self.r = r
        self.performance = {'score':[], 'smooth_score':[], 'moves':[], 'smooth_moves':[]}

    def load(self, file_name):
        self.q, self.alpha, self.gamma, self.r, self.performance = pickle.load(open(file_name, 'rb'), encoding='latin1')

    def save(self, file_name):
        pickle.dump([self.q, self.alpha, self.gamma, self.r, self.performance], open(file_name, 'wb'))

    def update_q(self, r, s, a, s_p):
        try:
            self.q[s][a] += self.alpha * (r + self.gamma * np.mean(model.q[s_p]) - model.q[s][a])
        except:
            self.q[s][a] += self.alpha * r


def debug_msg_1(old_state, probs_td, action, new_state):
    d_name = ['left', 'right', 'up', 'down']
    a_name = ['left', 'right', 'up', 'down', 'no']
    print("At ", old_state[0], old_state[1], "going ", d_name[old_state[-1]])
    print("value: ", model.q[old_state], "on ", old_state)
    print("probs: ", probs_td)
    print("action taken: %s (%i)" % (a_name[action], action))
    # print("new state mean: ", np.mean(model.q[new_state]))


def debug_msg_2(old_state, new_state):
    d_name = ['left', 'right', 'up', 'down']
    print("new value: ", model.q[old_state])
    print("new dir: ", d_name[new_state[-1]], "on ", new_state)


# MAIN FUNCTION
screen_size = np.array([400, 400])
grid_size = np.array([20, 20])
fix_pos = [5, 10]
env = GameSession(screen_size, grid_size, delay=0, fix_number=fix_pos, render=False)

# Model name
file_dir = "results/"
file_base = "v3"
file_post = "_a04_g099r20_mean_f"
file_ext = ".td"
file_name = file_dir + file_base + file_post + file_ext

# Maybe load previous model
if os.path.isfile(file_name):
    model = TabTD(grid_size)
    model.load(file_name)
else:
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    model = TabTD(grid_size, alpha=0.4, gamma=0.99, r=15)

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
        probs_td = np.exp(model.q[old_state]) / sum(np.exp(model.q[old_state]))
        action = np.random.choice(range(5), p=probs_td)

        action_taken, reward, alive = env.step(action, mode='AI')

        # Assign reward and update q function
        if action_taken:
            if env.number['n'] > score:
                score = env.number['n']
                number_moves = 0
            next_state = env.get_next_state()
            model.update_q(reward*20, old_state, action, next_state)
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
        plot_probs(model, fix_pos[0], fix_pos[1], file_name=file_dir + 'p_map' + file_post + '.png')
        model.save(file_name)

env.exit()
