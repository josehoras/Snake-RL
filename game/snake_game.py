from game.snake_objects import Snake
from game.snake_functions import *


# MAIN FUNCTION
black = 0, 0, 0
white = 255, 255, 255
blue = 0, 0, 255
screen_size = np.array([400, 400])
grid_size = np.array([20, 20])
sq_size = screen_size // grid_size

# Start pygame screen
pygame.init()
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Snake')
# Define fonts for screen messages
font_size = int(sq_size[1]*1.5)
font = pygame.font.SysFont("ubuntumono",  font_size)

# Create snake and first number
snake = Snake("square", screen_size, grid_size, speed=5)  # style "square" or "round"
number, number_grid, number_txt = generate_number(0, snake.grid, grid_size, font, white)

# plot welcome screen
update_game_screen(screen, snake, number, number_grid * sq_size , number_txt, font)
plot_msg("Press Space to start", screen, font)
while not(check_continue_event()):
    pass

# record_dir = "video/"

# Main loop
i = 0
while not(check_quit_event()):
    i += 1
    snake.update_move(pygame.key.get_pressed(), number_grid)

    if snake.state == "just_ate":
        number, number_grid, number_txt = generate_number(number, snake.grid, grid_size, font, white)
    if snake.state == "dead":
        break

    update_game_screen(screen, snake, number, number_grid * sq_size , number_txt, font)
    # if i % 3 == 0:
    #     pygame.image.save(screen, record_dir + "screenshot_" + '{:04d}'.format(i) + ".jpeg")
    pygame.time.delay(60)


plot_msg("Press Esc. to quit", screen, font)
while not(check_quit_event()):
    pass

pygame.display.quit()
