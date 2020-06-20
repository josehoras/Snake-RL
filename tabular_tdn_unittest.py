import unittest
from tabular_tdn_learning import TabularTDn


class TestTabularTDn(unittest.TestCase):
    def setUp(self):
        grid_size = [20, 20]
        self.model = TabularTDn(grid_size, alpha=0.5, gamma=0.8, n=3, policy='e-greedy', update_rule='sarsa')
        # currentTest = self.id().split('.')[-1]
        # if currentTest == 'test_2':
        #     self.skipTest('bla')

    def test_1(self):
        # This test shows that the reward present at state S[3] gets accounted into
        # the value of Q[S[0]][A[0]], as n=3, at timestamp 3 (t = 0, 1, 2, ...)
        # The expected result is the reward discounted by alpha and gamma^2
        # At this test all Q = 0. As the value of the next state is not relevent (0)
        # all algortithms work with this test (sarsa, expected sarsa, q-learning,...)
        S = [(10, 10, 13, 10, 1), (11, 10, 13, 10, 1), (12, 10, 13, 10, 1), (13, 10, 13, 10, 1)]
        A = [0, 0, 0]
        R = [0, 0, 0, 1]
        t = 2
        T = 3
        tau = t - self.model.n + 1
        if tau >= 0:
            self.model.update_q(S, A, R, tau, T)
        self.assertEqual(self.model.q[S[tau]][A[tau]], self.model.alpha * self.model.gamma ** 2 * R[3])

    def test_2(self):
        # This test shows that the value of state S[3][A3] gets accounted into
        # the value of Q[S[0]][A[0]], as n=3, at timestamp 3 (t = 0, 1, 2, ...)
        # A3 is calculated on policy in sarsa algorithm, choosing
        # the action with highest value that I previously set
        # To eliminate randomness I use an e-greedy policy with e=0
        # On e=0 sarsa is equal to q-learning
        # The expected result is the value at S[3][A3] discounted by alpha and gamma^3
        S = [(10, 10, 13, 10, 1), (11, 10, 13, 10, 1), (12, 10, 13, 10, 1), (13, 10, 13, 10, 1)]
        A = [0, 0, 0]
        R = [0, 0, 0, 0]
        self.model.q[S[3]][0] = 0.8
        self.model.e = 0
        t = 2
        T = 3
        tau = t - self.model.n + 1
        # print('Tau: ', tau)
        # print('Q at S[3]: ', self.model.q[S[3]])
        # print('Q at S[tau]: ', self.model.q[S[tau]])
        # print('testing...')
        if tau >= 0:
            self.model.update_q(S, A, R, tau, T)
        # print('Q at S[tau]: ', self.model.q[S[tau]])
        self.assertEqual(self.model.q[S[tau]][A[tau]], self.model.alpha * self.model.gamma ** 3 * self.model.q[S[3]][0])


if __name__ == "__main__":
    unittest.main()
