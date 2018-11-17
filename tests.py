import unittest
import numpy as np
from game import Game
from mcts import MCTS
from neural_network import NeuralNetwork

# I have invented a game called "guess-it", which is extremely simple.
# You have a NxN grid and must place a piece in the bottom right corner to win.
# I can work out the game trees for this simple game by hand.

class OnePlayerGuessIt(Game):
    def __init__(self): self.n = 2
    def get_initial_state(self): return np.zeros((self.n, self.n), dtype=np.float32)
    def get_available_actions(self, s): return s == 0
    def check_winner(self, s): return None if s[1,1] == 0 else 0
    def take_action(self, s, a): return s + a.astype(np.float32)
    def get_player(self, s): return 0

class TwoPlayerGuessIt(Game):
    def __init__(self): self.n = 2
    def get_initial_state(self): return np.zeros((self.n, self.n, 3), dtype=np.float32)
    def get_available_actions(self, s): return s[:, :, :2].sum(axis=-1) == 0
    def check_winner(self, s):
        if s[1,1,0] == 1:
            return 0
        if s[1,1,1] == 1:
            return 1
    def take_action(self, s, a):
        p = int(s[0,0,2])
        s = s.copy()
        s[:,:,p] += a.astype(np.float32) # Next move
        s[:,:,2] = (s[:,:,2] + 1) % 2 # Toggle player
        return s
    def get_player(self, s): return int(s[0,0,2])


# Always predicts a value of 0.5 for a node and a uniform distribution over actions.
class DumbNet(NeuralNetwork):
    def __init__(self, game): self.game = game
    def predict(self, s): return np.ones_like(self.game.get_available_actions(self.game.get_initial_state())), 0.5
        

class GuessItTest(unittest.TestCase):

    # Check that statistics of out-edges from expanded node are correct.
    def assertExpanded(self, state, mcts):
        hashed_s = mcts.np_hash(state)
        num_actions = len(mcts.tree[hashed_s])
        for k in mcts.tree[hashed_s]:
            v = mcts.tree[hashed_s][k]
            self.assertListEqual(v[1:], [0, 0, 1/num_actions])
            self.assertEqual(k, mcts.np_hash(v[0]))

    # Allows you to check edge statistics (N, Q, P)
    def assertEdge(self, state, action, mcts, statistics, heuristic=None):
        self.assertListEqual(mcts.tree[mcts.np_hash(state)][mcts.np_hash(action)][1:], statistics)
        if heuristic != None:
            n_total = 0
            for stat in mcts.tree[mcts.np_hash(state)].values():
                n_total += stat[1]
            self.assertEqual(statistics[1] + statistics[2]*(n_total**.5/(1+statistics[0])), heuristic)

    def time(self):
        import time
        gi = OnePlayerGuessIt()
        d = DumbNet(gi)
        m = MCTS(gi, d)
        start = time.clock()
        s = gi.get_initial_state()
        for _ in range(1600):
            m.simulate(s)
        stop = time.clock()
        print(stop - start)


    # Test that our MCTS behaves as expected for a one-player game of guess-it
    def test_one_player_guess_it(self):
        gi = OnePlayerGuessIt()
        d = DumbNet(gi)
        m = MCTS(gi, d)
        self.assertEqual(m.tree, {})
        init = gi.get_initial_state()

        # First simulation
        m.simulate(init) # Adds root and outward edges
        self.assertIn(m.np_hash(init), m.tree) # Root added to tree
        self.assertEqual(len(m.tree), 1)
        self.assertExpanded(init, m)

        # Second simulation
        m.simulate(init)
        s = gi.take_action(init, np.array([[1,0],[0,0]])) # Takes first action since uniform
        self.assertIn(m.np_hash(s), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 2)
        self.assertExpanded(s, m)
        self.assertEdge(init, np.array([0,0]), m, [1,.5,.25])
        self.assertEdge(init, np.array([0,1]), m, [0,0,.25])

        # Third simulation
        m.simulate(init)
        s_prev = s
        s = gi.take_action(s_prev, np.array([[0,1],[0,0]])) # Takes first action since uniform
        self.assertIn(m.np_hash(s), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 3)
        self.assertExpanded(s, m)
        self.assertEdge(init, np.array([0,0]), m, [2,.5,.25])
        self.assertEdge(init, np.array([0,1]), m, [0,0,.25])
        self.assertEdge(s_prev, np.array([0,1]), m, [1,.5, 1/3])
        self.assertEdge(s_prev, np.array([1,1]), m, [0,0,1/3])

        # Fourth simulation
        m.simulate(init)
        s_prev = s
        s = gi.take_action(s_prev, np.array([[0,0],[1,0]])) # Takes first action since uniform
        self.assertIn(m.np_hash(s), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 4)
        self.assertExpanded(s, m)
        self.assertEdge(init, np.array([0,0]), m, [3,.5,.25])
        self.assertEdge(s_prev, np.array([1,0]), m, [1,.5,1/2])
        self.assertEdge(s_prev, np.array([1,1]), m, [0,0,1/2])

        # Fifth simulation
        m.simulate(init)
        self.assertEqual(len(m.tree), 4) # Make sure it did not expand, terminal node.
        self.assertEdge(init, np.array([0,0]), m, [4,.625,.25])
        self.assertEdge(s_prev, np.array([1,0]), m, [2,.75,1/2])
        self.assertEdge(s_prev, np.array([1,1]), m, [0,0,1/2])

        # Run alot. This test is here to ensure future changes don't brake this seemingly correct implementation.
        for _ in range(10000):
            m.simulate(init)
            
        self.assertEdge(init, np.array([0,0]), m, [2384, 0.9991610738255033, 0.25])
        self.assertEdge(init, np.array([0,1]), m, [2488, 0.9995980707395499, 0.25])
        self.assertEdge(init, np.array([1,0]), m, [2540, 0.9998031496062992, 0.25])
        self.assertEdge(init, np.array([1,1]), m, [2592, 1.0, 0.25])


    # Test that our MCTS behaves as expected for a two-player game of guess-it
    def test_two_player_guess_it(self):
        gi = TwoPlayerGuessIt()
        d = DumbNet(gi)
        m = MCTS(gi, d)
        self.assertEqual(m.tree, {})
        init = gi.get_initial_state()

        # First simulation
        m.simulate(init) # Adds root and outward edges
        self.assertIn(m.np_hash(init), m.tree) # Root added to tree
        self.assertEqual(len(m.tree), 1)
        self.assertExpanded(init, m)

        # Second simulation
        m.simulate(init)
        s = gi.take_action(init, np.array([[1,0],[0,0]])) # Takes first action since uniform
        self.assertIn(m.np_hash(s), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 2)
        self.assertExpanded(s, m)
        self.assertEdge(init, np.array([0,0]), m, [1,-.5,.25], heuristic=-.375)
        self.assertEdge(init, np.array([0,1]), m, [0,0,.25], heuristic=.25)

        # Third simulation
        m.simulate(init)
        s = gi.take_action(init, np.array([[0,1],[0,0]])) # Now takes the second action
        self.assertIn(m.np_hash(s), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 3)
        self.assertExpanded(s, m)
        self.assertEdge(init, np.array([0,0]), m, [1,-.5,.25], heuristic=-0.32322330470336313)
        self.assertEdge(init, np.array([0,1]), m, [1,-.5,.25], heuristic=-0.32322330470336313)
        self.assertEdge(init, np.array([1,0]), m, [0,0,.25], heuristic=0.3535533905932738)

        # Fourth simulation
        m.simulate(init)
        s = gi.take_action(init, np.array([[0,0],[1,0]])) # Now takes the third action
        self.assertIn(m.np_hash(s), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 4)
        self.assertExpanded(s, m)
        self.assertEdge(init, np.array([0,0]), m, [1,-.5,.25], heuristic=-0.28349364905389035)
        self.assertEdge(init, np.array([0,1]), m, [1,-.5,.25], heuristic=-0.28349364905389035)
        self.assertEdge(init, np.array([1,0]), m, [1,-.5,.25], heuristic=-0.28349364905389035)
        self.assertEdge(init, np.array([1,1]), m, [0,0,.25], heuristic=0.4330127018922193)

        # Fifth simulation
        m.simulate(init)
        s = gi.take_action(init, np.array([[0,0],[0,1]])) # Takes fourth action since uniform
        self.assertEqual(len(m.tree), 4)
        self.assertEdge(init, np.array([0,0]), m, [1,-.5,.25], heuristic=-.25)
        self.assertEdge(init, np.array([0,1]), m, [1,-.5,.25], heuristic=-.25)
        self.assertEdge(init, np.array([1,0]), m, [1,-.5,.25], heuristic=-.25)
        self.assertEdge(init, np.array([1,1]), m, [1,1,.25], heuristic=1.25)

        # Run a few times until heuristic about to cross.
        for _ in range(145):
            m.simulate(init)
        self.assertEdge(init, np.array([0,0]), m, [1,-.5,.25], heuristic=1.0258194519667128)
        self.assertEdge(init, np.array([0,1]), m, [1,-.5,.25], heuristic=1.0258194519667128)
        self.assertEdge(init, np.array([1,0]), m, [1,-.5,.25], heuristic=1.0258194519667128)
        self.assertEdge(init, np.array([1,1]), m, [146,1,.25], heuristic=1.0207594483260778)

        # Heuristic crosses over
        m.simulate(init)
        s_prev = gi.take_action(init, np.array([[1,0],[0,0]]))
        s = gi.take_action(s_prev, np.array([[0,1],[0,0]]))
        self.assertIn(m.np_hash(s), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 5)
        self.assertEdge(s_prev, np.array([0,1]), m, [1,-.5,1/3], heuristic=-0.33333333333333337)
        self.assertEdge(init, np.array([0,0]), m, [2, 0,.25], heuristic=1.0206207261596576)
        self.assertEdge(init, np.array([0,1]), m, [1,-.5,.25], heuristic=1.0309310892394863)
        self.assertEdge(init, np.array([1,0]), m, [1,-.5,.25], heuristic=1.0309310892394863)
        self.assertEdge(init, np.array([1,1]), m, [146,1,.25], heuristic=1.0208289944114215)

        # Run alot. This test is here to ensure future changes don't brake this seemingly correct implementation.
        for _ in range(10000):
            m.simulate(init)

        # We have learned that player 0's optimal strategy is to place the cross in the bottom right box
        self.assertEdge(init, np.array([0,0]), m,  [14, -0.75, 0.25], 0.9291201399674904)
        self.assertEdge(init, np.array([0,1]), m, [14, -0.75, 0.25], 0.9291201399674904)
        self.assertEdge(init, np.array([1,0]), m, [14, -0.75, 0.25], 0.9291201399674904)
        self.assertEdge(init, np.array([1,1]), m, [10108, 1.0, 0.25], 1.0024915226134645)

        # We have indirectly learn that player 1's optimal strategy, should player 0 fail on the first turn,
        # is the place the circle in the bottom right box
        self.assertEdge(s_prev, np.array([0,1]), m, [1, -0.5, 1/3], 0.10092521257733145)
        self.assertEdge(s_prev, np.array([1,0]), m, [1, -0.5, 1/3], 0.10092521257733145)
        self.assertEdge(s_prev, np.array([1,1]), m, [11, 1.0, 1/3], 1.1001542020962218)

        



if __name__ == '__main__':
    unittest.main()