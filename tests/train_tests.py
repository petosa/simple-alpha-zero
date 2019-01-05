import unittest
import sys
import numpy as np
import torch
sys.path.append("..")
from games.guessit import TwoPlayerGuessIt
from games.tictactoe import TicTacToe
from games.leapfrog import ThreePlayerLinearLeapFrog
from models.dumbnet import DumbNet
from models.priornet import PriorNet
from trainer import Trainer
from neural_network import NeuralNetwork



class TrainTests(unittest.TestCase):

    def test_dumbnet_guessit_self_play(self):
        gi = TwoPlayerGuessIt()
        nn = NeuralNetwork(gi, DumbNet)

        t = Trainer(gi, nn, num_simulations=2, num_games=1, num_updates=0, buffer_size_limit=None, cpuct=1, num_threads=1)
        data = t.self_play(temperature=0)

        np.testing.assert_equal(data[:,-1], np.array([-1, 1, -1, 1]))
        s = gi.get_initial_state()
        np.testing.assert_equal(data[0,0], s)
        np.testing.assert_equal(data[0,1], np.array([1, 0, 0, 0]))
        s = gi.take_action(s, np.array([[1,0],[0,0]]))
        np.testing.assert_equal(data[1,0], s)
        np.testing.assert_equal(data[1,1], np.array([1, 0, 0]))
        s = gi.take_action(s, np.array([[0,1],[0,0]]))
        np.testing.assert_equal(data[2,0], s)
        np.testing.assert_equal(data[2,1], np.array([1, 0]))
        s = gi.take_action(s, np.array([[0,0],[1,0]]))
        np.testing.assert_equal(data[3,0], s)
        np.testing.assert_equal(data[3,1], np.array([1]))


    def test_priornet_tictactoe_self_play(self):
        ttt = TicTacToe()
        nn = NeuralNetwork(ttt, PriorNet)

        t = Trainer(ttt, nn, num_simulations=2, num_games=1, num_updates=0, buffer_size_limit=None, cpuct=1, num_threads=4)
        data = t.self_play(temperature=0)

        np.testing.assert_equal(data[:,-1], np.array([1, -1, 1, -1, 1, -1, 1]))
        s = ttt.get_initial_state()
        np.testing.assert_equal(data[0,0], s)
        np.testing.assert_equal(data[0,1], np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]))
        s = ttt.take_action(s, np.array([[0,1,0],[0,0,0],[0,0,0]])) # Top-middle X
        np.testing.assert_equal(data[1,0], s) 
        np.testing.assert_equal(data[1,1], np.array([1, 0, 0, 0, 0, 0, 0, 0]))
        s = ttt.take_action(s, np.array([[1,0,0],[0,0,0],[0,0,0]])) # Top-left O
        np.testing.assert_equal(data[2,0], s)
        np.testing.assert_equal(data[2,1], np.array([1, 0, 0, 0, 0, 0, 0]))
        s = ttt.take_action(s, np.array([[0,0,1],[0,0,0],[0,0,0]])) # Top-right X
        np.testing.assert_equal(data[3,0], s)
        np.testing.assert_equal(data[3,1], np.array([1, 0, 0, 0, 0, 0]))
        s = ttt.take_action(s, np.array([[0,0,0],[1,0,0],[0,0,0]])) # Mid-left O
        np.testing.assert_equal(data[4,0], s)
        np.testing.assert_equal(data[4,1], np.array([1, 0, 0, 0, 0]))
        s = ttt.take_action(s, np.array([[0,0,0],[0,1,0],[0,0,0]])) # Mid-mid X
        np.testing.assert_equal(data[5,0], s)
        np.testing.assert_equal(data[5,1], np.array([1, 0, 0, 0]))
        s = ttt.take_action(s, np.array([[0,0,0],[0,0,1],[0,0,0]])) # Mid-right O
        np.testing.assert_equal(data[6,0], s)
        np.testing.assert_equal(data[6,1], np.array([1, 0, 0]))


    def test_dumbnet_linearleapfrog_self_play(self):
        # This test makes sure things still work in the multiplayer case
        llf = ThreePlayerLinearLeapFrog()
        nn = NeuralNetwork(llf, DumbNet)

        t = Trainer(llf, nn, num_simulations=2, num_games=1, num_updates=0, buffer_size_limit=None, cpuct=1, num_threads=1)
        data = t.self_play(temperature=0)
        np.testing.assert_equal(data[:,-1], np.array([-1, 1, -1, -1, 1]))


    def test_policy_iteration(self):
        ttt = TicTacToe()
        nn = NeuralNetwork(ttt, PriorNet)
        t = Trainer(ttt, nn, num_simulations=2, num_games=100, num_updates=0, buffer_size_limit=None, cpuct=1, num_threads=4)
        t.policy_iteration()
        states = t.training_data[:,0]
        inits = 0
        for s in states:
            if (s.astype(np.float32) == ttt.get_initial_state()).all():
                inits += 1
        self.assertEqual(inits, 100)


    # This test verifies that the training data buffer is properly managed when a limit is set.
    def test_buffer_size_limit_100(self):
        ttt = TicTacToe()
        nn = NeuralNetwork(ttt, PriorNet)
        t = Trainer(ttt, nn, num_simulations=2, num_games=100, num_updates=0, buffer_size_limit=100, cpuct=1, num_threads=4)
        t.policy_iteration()
        self.assertEqual(len(t.training_data), 100)

   
if __name__ == '__main__':
    unittest.main()