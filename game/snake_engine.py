from game.snake_objects import Snake
from game.snake_functions import *

# white = 255, 255, 255
class GameSession:
    def __init__(self, screen_size, grid_size, delay=0, render=True, fix_number=''):
        self.screen_size = screen_size
        self.grid_size = grid_size
        self.sq_size = screen_size // grid_size
        self.delay = delay
        self.render = render
        self.number = {'n': 0, 'grid': np.array([0,0]), 'txt': 0}
        self.fix_number = fix_number
        self.state = "live"
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
        if self.fix_number != '':
            self.number['grid'] = np.array(self.fix_number)
        else:
            c = [0]
            while c:
                self.number['grid'] = np.array(
                    [random.randint(gap, self.grid_size[0] - 1 - gap), random.randint(gap, self.grid_size[1] - 1 - gap)])
                c = [g for g in self.snake.grid if (self.number['grid'] == g).all()]
        self.number['txt'] = self.font.render(str(self.number['n']), True, color, grey)

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
        self.snake = Snake("square", self.screen_size, self.grid_size, speed=5)  # style "square" or "round"
        self.gen_number(0, white)
        self.update_screen()

    def check_game_event(self):
        reward = 0
        if self.snake.state == "just_ate":
            self.gen_number(self.number['n'], white)
            reward = 1
        if self.snake.state == "dead":
            reward = -1
            self.state = "dead"
        return reward

    def step(self, action, mode='manual'):
        action_taken = self.snake.update_move(action, self.number['grid'], mode=mode)
        self.update_screen()
        self.delay = check_delay(self.delay, pygame.key.get_pressed())
        pygame.time.delay(self.delay)
        reward = self.check_game_event()
        print(self.state, reward)
        return action_taken, reward, self.state

    def exit(self):
        plot_msg("Press Esc. to quit", self.screen, self.font)
        pygame.display.update()
        while not(check_quit_event()):
            pass
        pygame.display.quit()