import sys
import numpy as np
sys.path.append("..")
from neural_network import NeuralNetwork


class DumbNet(NeuralNetwork):
    def __init__(self, game): self.game = game
    def train(self, data): pass
    def predict(self, s): return np.ones_like(self.game.get_available_actions(self.game.get_initial_state())), 0.0
