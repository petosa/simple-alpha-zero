import numpy as np
import sys
sys.path.append("..")
from game import Game

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
    def friendly_print(self, s):
        board = np.ones((2,2)).astype(np.object)
        board[:,:] = " "
        board[s == 1] = 'a'
        print(board)

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
    def friendly_print(self, s):
        board = np.ones((2,2)).astype(np.object)
        board[:,:] = " "
        board[s[:,:,0] == 1] = 'a'
        board[s[:,:,1] == 1] = 'b'
        print(board)

