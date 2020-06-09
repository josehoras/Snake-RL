from game.snake_objects import Snake
from game.snake_functions import *
from performance_mon import *
import os.path
import pickle
import torch


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


# MAIN FUNCTION
black = 0, 0, 0
white = 255, 255, 255
blue = 0, 0, 255
screen_size = np.array([400, 400])
grid_size = np.array([20, 20])
sq_size = screen_size // grid_size
render = False
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
# Define model

fix_pos = [5, 10]
# Maybe load previous model
file_dir = "td/"
file_base = "v2"
file_post = "_a04_g099r20_mean_f"
file_ext = ".td"
file_name = file_dir + file_base + file_post + file_ext

if not os.path.isdir(file_dir):
    os.mkdir(file_dir)

if os.path.isfile(file_name):
    model = TabTD(grid_size)
    model.load(file_name)
else:
    model = TabTD(grid_size, alpha=0.4, gamma=0.99, r=15)

# Start pygame screen
pygame.init()
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Snake')
if render==False: pygame.display.iconify()
# Define fonts for screen messages
font_size = int(sq_size[1]*1.5)
font = pygame.font.SysFont("ubuntumono",  font_size)
delay = 0

record_dir = "/video/"

print(model.q.shape)
directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
d_name = ['left', 'right', 'up', 'down']
a_name = ['left', 'right', 'up', 'down', 'no']

# Main loop
for i in range(len(model.performance['moves']) + 1, len(model.performance['moves']) + 200001):
    # Create snake and first number
    snake = Snake("square", screen_size, grid_size, speed=5)  # style "square" or "round"
    number, number_grid, number_txt = generate_number(0, snake.grid, grid_size,
                                                      font, white, set=fix_pos)
    update_game_screen(screen, snake, number, number_grid * sq_size , number_txt, font, render=render)

    level_moves = 0
    number_moves = 0
    while not(check_quit_event()):
        level_moves += 1
        number_moves += 1
        # features = calculate_features(snake, number_grid, grid_size).to(device)
        old_dir = [i for i in range(4) if (directions[i]==snake.head.dir).all()][0]
        old_state = (snake.head.grid[0], snake.head.grid[1], number_grid[0], number_grid[1], old_dir)
        probs_td = np.exp(model.q[old_state])/sum(np.exp(model.q[old_state]))

        # Take action depending on prob of each option.
        action = np.random.choice(range(5), p=probs_td)  # policy
        action_taken = snake.update_move(action, number_grid, mode='AI')
        # snake.update_move(pygame.key.get_pressed(), number_grid)  # Manual play
        new_dir = [i for i in range(4) if (directions[i]==snake.head.dir).all()][0]
        new_state = (snake.head.grid[0] + snake.head.dir[0], snake.head.grid[1] + snake.head.dir[1],
                     number_grid[0], number_grid[1], new_dir)
        if action_taken:
            reward = torch.tensor([0], dtype=torch.float)
            if snake.state == "just_ate":
                number_moves = 0
                reward += model.r
                number, number_grid, number_txt = generate_number(number, snake.grid, grid_size,
                                                                  font, white, set=fix_pos)
                # print("At ", snake.head.grid[0], snake.head.grid[1], "going ", d_name[old_dir])
                # print("value: ", value[old_state], "on ", old_state)
                # print("probs: ", probs_td)
                # print("action taken: %s (%i)" % (a_name[action], action))
                # print("new state mean: ", np.mean(value[new_state]))

                model.update_q(reward, old_state, action, new_state)

                # print("new value: ", value[old_state])
                # print("new dir: ", d_name[new_dir], "on ", new_state)
            if snake.state == "dead":
                reward += -model.r
                # print("At ", snake.head.grid[0], snake.head.grid[1], "going ", d_name[old_dir])
                # print("old probs: ", probs_td, "on ", old_state)
                # print("action taken: %s (%i)" % (a_name[action], action))
                # update this value
                model.update_q(reward, old_state, action, new_state)
                # print("new probs: ", np.exp(value[old_state])/sum(np.exp(value[old_state])), "on ", old_state)
                # print("new dir: ", d_name[new_dir], "on ", new_state)
                break
            model.update_q(reward, old_state, action, new_state)
        if number_moves == 1000 or number > 50:
            break

        update_game_screen(screen, snake, number, number_grid * sq_size , number_txt, font, render=render)
        delay = check_delay(delay, pygame.key.get_pressed())

        pygame.time.delay(delay)

    model.performance = update_performance(model.performance, number, level_moves)
    print(i, " - score: %i (%.1f), moves: %i (%.0f)" %
          (model.performance['score'][-1], model.performance['smooth_score'][-1],
           model.performance['moves'][-1], model.performance['smooth_moves'][-1]))
    print()
    if i % 50 == 0:
        plot_performance(model.performance, file_dir + file_base + file_post)
        # plot_value(value, 5, 10, file_name=file_dir + 'v_map' + file_post + '.png',
        #            alpha=alpha, gamma=gamma, r=r, it=i)
        plot_probs(model.q, fix_pos[0], fix_pos[1], file_name=file_dir + 'p_map' + file_post + '.png',
                   alpha=model.alpha, gamma=model.gamma, r=model.r, it=i)
        model.save(file_name)


plot_msg("Press Esc. to quit", screen, font)
pygame.display.update()
while not(check_quit_event()):
    pass
pygame.display.quit()
