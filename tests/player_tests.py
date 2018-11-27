import unittest
import sys
import numpy as np
sys.path.append("..")
from players.uninformed_mcts_player import UninformedMCTSPlayer
from players.deep_mcts_player import DeepMCTSPlayer
from games.guessit import TwoPlayerGuessIt
from neural_network import NeuralNetwork
from models.priornet import PriorNet


class UninformedMCTSPlayerPlayerTest(unittest.TestCase):

    def test_one_sim_guess_it(self):
        gi = TwoPlayerGuessIt()
        p = UninformedMCTSPlayer(gi, 1)
        init = gi.get_initial_state()
        s = p.update_state(init)
        s_gt = init
        s_gt[0,0,0] = 1
        s_gt[:,:,-1] = 1
        self.assertTrue((s == s_gt).all())

    def test_two_sim_guess_it(self):
        gi = TwoPlayerGuessIt()
        p = UninformedMCTSPlayer(gi, 2)
        init = gi.get_initial_state()
        s = p.update_state(init)
        s_gt = init
        s_gt[0,0,0] = 1
        s_gt[:,:,-1] = 1
        self.assertTrue((s == s_gt).all())

    def test_six_sim_guess_it(self):
        for _ in range(1000): # There should be no stochasticity (temp=0)
            gi = TwoPlayerGuessIt()
            p = UninformedMCTSPlayer(gi, 6)
            init = gi.get_initial_state()
            s = p.update_state(init)
            s_gt = init
            s_gt[-1,-1,0] = 1
            s_gt[:,:,-1] = 1
            self.assertTrue((s == s_gt).all())
    
    # Tests that the tree persists between calls to update_state
    def test_memory_guess_it(self):
        gi = TwoPlayerGuessIt()
        p = UninformedMCTSPlayer(gi, 2)
        init = gi.get_initial_state()
        s = p.update_state(init)
        s_gt = init.copy()
        s_gt[0,0,0] = 1
        s_gt[:,:,-1] = 1
        self.assertTrue((s == s_gt).all())
        for _ in range(100):
            s = p.update_state(init)
        s_gt = init.copy()
        s_gt[1,1,0] = 1
        s_gt[:,:,-1] = 1
        self.assertTrue((s == s_gt).all())

    # Test that resets are effective.
    def test_reset_guess_it(self):
        gi = TwoPlayerGuessIt()
        p = UninformedMCTSPlayer(gi, 2)
        init = gi.get_initial_state()
        s = p.update_state(init)
        s_gt = init.copy()
        s_gt[0,0,0] = 1
        s_gt[:,:,-1] = 1
        self.assertTrue((s == s_gt).all())
        for _ in range(100):
            p.reset()
            s = p.update_state(init)
        self.assertTrue((s == s_gt).all())



class DeepMCTSPlayerPlayerTest(unittest.TestCase):

    def test_two_sim_guess_it(self):
        gi = TwoPlayerGuessIt()
        pr = NeuralNetwork(gi, PriorNet)
        p = DeepMCTSPlayer(gi, pr, 2)
        init = gi.get_initial_state()
        s = p.update_state(init)
        s_gt = init
        s_gt[0,1,0] = 1
        s_gt[:,:,-1] = 1
        self.assertTrue((s == s_gt).all())

    def test_three_sim_guess_it(self):
        for _ in range(100): # There should be no stochasticity (temp=0)
            gi = TwoPlayerGuessIt()
            pr = NeuralNetwork(gi, PriorNet)
            p = DeepMCTSPlayer(gi, pr, 3)
            init = gi.get_initial_state()
            s = p.update_state(init)
            s_gt = init.copy()
            s_gt[0,1,0] = 1
            s_gt[:,:,-1] = 1
            self.assertTrue((s == s_gt).all())

    # Tests that the tree persists between calls to update_state
    def test_memory_guess_it(self):
        gi = TwoPlayerGuessIt()
        pr = NeuralNetwork(gi, PriorNet)
        p = DeepMCTSPlayer(gi, pr, 2)
        init = gi.get_initial_state()
        s = p.update_state(init)
        s_gt = init.copy()
        s_gt[0,1,0] = 1
        s_gt[:,:,-1] = 1
        self.assertTrue((s == s_gt).all())
        for _ in range(100):
            s = p.update_state(init)
        s_gt = init.copy()
        s_gt[1,1,0] = 1
        s_gt[:,:,-1] = 1
        self.assertTrue((s == s_gt).all())

    # Test that resets are effective.
    def test_reset_guess_it(self):
        gi = TwoPlayerGuessIt()
        pr = NeuralNetwork(gi, PriorNet)
        p = DeepMCTSPlayer(gi, pr, 2)
        init = gi.get_initial_state()
        s = p.update_state(init)
        s_gt = init.copy()
        s_gt[0,1,0] = 1
        s_gt[:,:,-1] = 1
        self.assertTrue((s == s_gt).all())
        for _ in range(100):
            p.reset()
            s = p.update_state(init)
        self.assertTrue((s == s_gt).all())





if __name__ == '__main__':
    unittest.main()