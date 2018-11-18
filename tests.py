import unittest
import numpy as np
from games.guessit import OnePlayerGuessIt, TwoPlayerGuessIt
from games.leapfrog import ThreePlayerLeapFrog, ThreePlayerLinearLeapFrog
from mcts import MCTS
from neural_network import NeuralNetwork


# Always predicts a value of 0.5 for a node and a uniform distribution over actions.
class DumbNet(NeuralNetwork):
    def __init__(self, game): self.game = game
    def predict(self, s): return np.ones_like(self.game.get_available_actions(self.game.get_initial_state())), 0.5


class MCTSTest(unittest.TestCase):

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

class GuessItTest(MCTSTest):

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



class LeapFrogTest(MCTSTest):

    def test_three_player_linear_leap_frog(self):
        lf = ThreePlayerLinearLeapFrog()
        d = DumbNet(lf)
        m = MCTS(lf, d)
        self.assertEqual(m.tree, {})
        init = lf.get_initial_state()

        # First simulation
        m.simulate(init) # Adds root and outward edges
        self.assertIn(m.np_hash(init), m.tree) # Root added to tree
        self.assertEqual(len(m.tree), 1)
        self.assertExpanded(init, m)

        # Second simulation
        m.simulate(init)
        s1 = lf.take_action(init, np.array([0,1,0,0,0,0])) # Takes first action since uniform
        self.assertIn(m.np_hash(s1), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 2)
        self.assertExpanded(s1, m)
        self.assertEdge(init, np.array([1]), m, [1,-.5,1], heuristic=0)
        self.assertEdge(s1, np.array([2]), m, [0,0,1], heuristic=0)

        # Third simulation
        m.simulate(init)
        s2 = lf.take_action(s1, np.array([0,0,1,0,0,0])) # Takes first action since uniform
        self.assertIn(m.np_hash(s2), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 3)
        self.assertExpanded(s2, m)
        self.assertEdge(init, np.array([1]), m, [2,-.5,1], heuristic=-0.028595479208968266)
        self.assertEdge(s1, np.array([2]), m, [1,-.5,1], heuristic=0)
        self.assertEdge(s2, np.array([3]), m, [0,0,1], heuristic=0)

        # Fourth simulation
        m.simulate(init)
        s3 = lf.take_action(s2, np.array([0,0,0,1,0,0])) # Takes first action since uniform
        self.assertIn(m.np_hash(s3), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 4)
        self.assertExpanded(s3, m)
        self.assertEdge(init, np.array([1]), m, [3,-1/6,1], heuristic=0.26634603522555267)
        self.assertEdge(s1, np.array([2]), m, [2,-.5,1], heuristic=-0.028595479208968266)
        self.assertEdge(s2, np.array([3]), m, [1,-.5,1], heuristic=0)
        self.assertEdge(s3, np.array([4]), m, [0,0,1], heuristic=0)

        # Fifth simulation
        m.simulate(init)
        s4 = lf.take_action(s3, np.array([0,0,0,0,1,0])) # Takes first action since uniform
        self.assertIn(m.np_hash(s4), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 5)
        self.assertExpanded(s4, m)
        self.assertEdge(init, np.array([1]), m, [4,-.25,1], heuristic=0.15000000000000002)
        self.assertEdge(s1, np.array([2]), m, [3,-1/6,1], heuristic=0.26634603522555267)
        self.assertEdge(s2, np.array([3]), m, [2,-.5,1], heuristic=-0.028595479208968266)
        self.assertEdge(s3, np.array([4]), m, [1,-.5,1], heuristic=0)
        self.assertEdge(s4, np.array([5]), m, [0,0,1], heuristic=0)

        # Sixth simulation - terminal state
        m.simulate(init)
        self.assertEqual(len(m.tree), 5)
        self.assertEdge(init, np.array([1]), m, [5,-.4,1], heuristic=-0.027322003750035073)
        self.assertEdge(s1, np.array([2]), m, [4,.125,1], heuristic=0.525)
        self.assertEdge(s2, np.array([3]), m, [3,-2/3,1], heuristic=-0.23365396477444733)
        self.assertEdge(s3, np.array([4]), m, [2,-.75,1], heuristic=-0.27859547920896827)
        self.assertEdge(s4, np.array([5]), m, [1,1,1], heuristic=1.5)

        # Simulate from end of chain
        m.simulate(s4)
        self.assertEqual(len(m.tree), 5)
        self.assertEdge(init, np.array([1]), m, [5,-.4,1], heuristic=-0.027322003750035073)
        self.assertEdge(s1, np.array([2]), m, [4,.125,1], heuristic=0.525)
        self.assertEdge(s2, np.array([3]), m, [3,-2/3,1], heuristic=-0.23365396477444733)
        self.assertEdge(s3, np.array([4]), m, [2,-.75,1], heuristic=-0.27859547920896827)
        self.assertEdge(s4, np.array([5]), m, [2,1,1], heuristic=1.4714045207910318)

        # Simulate from middle of chain
        m.simulate(s3)
        self.assertEqual(len(m.tree), 5)
        self.assertEdge(init, np.array([1]), m, [5,-.4,1], heuristic=-0.027322003750035073)
        self.assertEdge(s1, np.array([2]), m, [4,.125,1], heuristic=0.525)
        self.assertEdge(s2, np.array([3]), m, [3,-2/3,1], heuristic=-0.23365396477444733)
        self.assertEdge(s3, np.array([4]), m, [3,-5/6,1], heuristic=-0.40032063144111407)
        self.assertEdge(s4, np.array([5]), m, [3,1,1], heuristic=1.4330127018922192)


    def test_three_player_leap_frog(self):
        lf = ThreePlayerLeapFrog()
        d = DumbNet(lf)
        m = MCTS(lf, d)
        self.assertEqual(m.tree, {})
        init = lf.get_initial_state()

        # First simulation
        m.simulate(init) # Adds root and outward edges
        self.assertIn(m.np_hash(init), m.tree) # Root added to tree
        self.assertEqual(len(m.tree), 1)
        self.assertExpanded(init, m)

        # Second simulation
        m.simulate(init)
        s = lf.take_action(init, np.array([0,1,0,0,0,0,0,0,0,0])) # Takes first action since uniform
        self.assertIn(m.np_hash(s), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 2)
        self.assertExpanded(s, m)
        self.assertEdge(init, np.array([1]), m, [1,-.5,1/3], heuristic=-0.33333333333333337)
        self.assertEdge(init, np.array([2]), m, [0,0,1/3], heuristic=1/3)

        # Third simulation
        m.simulate(init)
        s = lf.take_action(init, np.array([0,0,1,0,0,0,0,0,0,0])) # Takes first action since uniform
        self.assertIn(m.np_hash(s), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 3)
        self.assertExpanded(s, m)
        self.assertEdge(init, np.array([1]), m, [1,-.5,1/3], heuristic=-0.26429773960448416)
        self.assertEdge(init, np.array([2]), m, [1,-.5,1/3], heuristic=-0.26429773960448416)
        self.assertEdge(init, np.array([3]), m, [0,0,1/3], heuristic=0.4714045207910317)

        # Fourth simulation
        m.simulate(init)
        s = lf.take_action(init, np.array([0,0,0,1,0,0,0,0,0,0])) # Takes first action since uniform
        self.assertIn(m.np_hash(s), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 4)
        self.assertExpanded(s, m)
        self.assertEdge(init, np.array([1]), m, [1,-.5,1/3], heuristic=-0.21132486540518713)
        self.assertEdge(init, np.array([2]), m, [1,-.5,1/3], heuristic=-0.21132486540518713)
        self.assertEdge(init, np.array([3]), m, [1,-.5,1/3], heuristic=-0.21132486540518713)

        # Fifth simulation
        m.simulate(init)
        s_prev = lf.take_action(init, np.array([0,1,0,0,0,0,0,0,0,0]))
        s = lf.take_action(s_prev, np.array([0,0,1,0,0,0,0,0,0,0])) # Takes first action since uniform
        self.assertIn(m.np_hash(s), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 5)
        self.assertExpanded(s, m)
        self.assertEdge(s_prev, np.array([2]), m, [1,-.5,1/3], heuristic=-0.33333333333333337)
        self.assertEdge(init, np.array([1]), m, [2,-.5,1/3], heuristic=-0.2777777777777778)

        # Sixth simulation
        m.simulate(init)
        s_prev = lf.take_action(init, np.array([0,0,1,0,0,0,0,0,0,0]))
        s = lf.take_action(s_prev, np.array([0,0,0,1,0,0,0,0,0,0])) # Takes first action since uniform
        self.assertIn(m.np_hash(s), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 6)
        self.assertExpanded(s, m)
        self.assertEdge(s_prev, np.array([3]), m, [1,-.5,1/3], heuristic=-0.33333333333333337)
        self.assertEdge(init, np.array([1]), m, [2,-.5,1/3], heuristic=-0.2515480025000234)
        self.assertEdge(init, np.array([2]), m, [2,-.5,1/3], heuristic=-0.2515480025000234)

        # Seventh simulation
        m.simulate(init)
        s_prev = lf.take_action(init, np.array([0,0,0,1,0,0,0,0,0,0]))
        s = lf.take_action(s_prev, np.array([0,0,0,0,1,0,0,0,0,0])) # Takes first action since uniform
        self.assertIn(m.np_hash(s), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 7)
        self.assertExpanded(s, m)
        self.assertEdge(s_prev, np.array([4]), m, [1,-.5,1/3], heuristic=-0.33333333333333337)
        self.assertEdge(init, np.array([1]), m, [2,-.5,1/3], heuristic=-0.22783447302409138)
        self.assertEdge(init, np.array([2]), m, [2,-.5,1/3], heuristic=-0.22783447302409138)
        self.assertEdge(init, np.array([3]), m, [2,-.5,1/3], heuristic=-0.22783447302409138)

        # Eigth simulation - Q value should now be positive.
        m.simulate(init)
        s_prev = lf.take_action(init, np.array([0,1,0,0,0,0,0,0,0,0]))
        s_prev = lf.take_action(s_prev, np.array([0,0,0,1,0,0,0,0,0,0]))
        s = lf.take_action(s_prev, np.array([0,0,0,0,1,0,0,0,0,0]))
        self.assertIn(m.np_hash(s), m.tree) # Node added to tree
        self.assertEqual(len(m.tree), 8)
        self.assertExpanded(s, m)
        self.assertEdge(init, np.array([1]), m, [3,-1/6,1/3], heuristic=0.05381260925538256)

        # Something new: simulate starting from a node other than the root.
        # We should receive -.5 during propagtion because we switched perspectives.
        self.assertEdge(s_prev, np.array([4]), m, [1,-.5,1/3], heuristic=-0.33333333333333337)
        m.simulate(s_prev) # We are now player 3.
        self.assertEdge(s_prev, np.array([5]), m, [1,-.5,1/3], heuristic=-0.26429773960448416)

        # Run alot. This test is here to ensure future changes don't brake this seemingly correct implementation.
        for _ in range(1000):
            m.simulate(init)

        # We have learned that player 0's optimal strategy is to take a step of size 3
        self.assertEdge(init, np.array([1]), m, [28, -0.26785714285714285, 0.3333333333333333], 0.09689301007670609)
        self.assertEdge(init, np.array([2]), m, [60, -0.05000000000000002, 0.3333333333333333], 0.12340581041117407)
        self.assertEdge(init, np.array([3]), m, [919, 0.12459194776931437, 0.3333333333333333], 0.13608950693788135)

        





if __name__ == '__main__':
    unittest.main()