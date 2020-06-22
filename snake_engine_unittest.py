import unittest
from game.snake_engine import GameSession


class TestTabularTDn(unittest.TestCase):
    def setUp(self):
        grid_size = [20, 20]
        screen_size = [400, 400]
        env = GameSession(screen_size, grid_size, delay=40, fix_number=[12, 10])
        env.start_game()
        # currentTest = self.id().split('.')[-1]
        # if currentTest == 'test_2':
        #     self.skipTest('bla')

    def test_1(self):
        pass

if __name__ == "__main__":
    unittest.main()
