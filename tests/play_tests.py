import unittest
import sys
import numpy as np
sys.path.append("..")
from play import play_match
from players.uninformed_mcts_player import UninformedMCTSPlayer
from games.tictactoe import TicTacToe
from games.guessit import TwoPlayerGuessIt
from games.leapfrog import ThreePlayerLeapFrog
from neural_network import NeuralNetwork
from models.dumbnet import DumbNet


class PlayTest(unittest.TestCase):

    def test_simple(self):
        game = TicTacToe()
        p1, p2 = UninformedMCTSPlayer(game, 1), UninformedMCTSPlayer(game, 1)
        scores, outcomes = play_match(game, [p1, p2])
        self.assertEqual(scores, {p1: 1.0, p2: 0.0})
        self.assertEqual(outcomes, {p1: "Win", p2: "Lose"})

    def test_permutation(self):
        game = TicTacToe()
        p1, p2 = UninformedMCTSPlayer(game, 1), UninformedMCTSPlayer(game, 1)
        scores, outcomes = play_match(game, [p1, p2], permute=True)
        self.assertEqual(scores, {p1: 0.5, p2: 0.5})
        self.assertEqual(outcomes, {p1: "Tie", p2: "Tie"})

    def test_self_play(self):
        game = TwoPlayerGuessIt()
        p1 = UninformedMCTSPlayer(game, 4)
        scores, outcomes = play_match(game, [p1, p1], permute=True)
        self.assertEqual(len(scores), 1)
        self.assertEqual(len(outcomes), 1)

    def test_resets_work(self):
        game = TicTacToe()
        p1, p2 = UninformedMCTSPlayer(game, 1), UninformedMCTSPlayer(game, 1)
        for _ in range(100):
            scores, outcomes = play_match(game, [p1, p2])
            self.assertEqual(scores, {p1: 1.0, p2: 0.0})
            self.assertEqual(outcomes, {p1: "Win", p2: "Lose"})
        for _ in range(100):
            scores, outcomes = play_match(game, [p1, p2], permute=True)
            self.assertEqual(scores, {p1: 0.5, p2: 0.5})
            self.assertEqual(outcomes, {p1: "Tie", p2: "Tie"})

    def test_three_simple(self):
        game = ThreePlayerLeapFrog()
        p1, p2, p3 = UninformedMCTSPlayer(game, 1), UninformedMCTSPlayer(game, 1), UninformedMCTSPlayer(game, 1)
        scores, outcomes = play_match(game, [p1, p2, p3])
        self.assertEqual(scores, {p1: 0.0, p2: 0.0, p3: 1.0})
        self.assertEqual(outcomes, {p1: "Lose", p2: "Lose", p3: "Win"})

    def test_three_permutation(self):
        game = ThreePlayerLeapFrog()
        p1, p2, p3 = UninformedMCTSPlayer(game, 1), UninformedMCTSPlayer(game, 1), UninformedMCTSPlayer(game, 1)
        scores, outcomes = play_match(game, [p1, p2, p3], permute=True)
        self.assertEqual(scores, {p1: 1/3, p2: 1/3, p3: 1/3})
        self.assertEqual(outcomes, {p1: "Tie", p2: "Tie", p3: "Tie"})

if __name__ == '__main__':
    unittest.main()