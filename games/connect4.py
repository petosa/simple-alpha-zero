import numpy as np
import sys
from scipy.signal import correlate2d
sys.path.append("..")
from game import Game

# Implementation for Connect 4.
class Connect4(Game):

    # Returns a blank Connect 4 board.
    # There are extra layers to represent the red and yellow pieces, as well as a turn indicator layer.
    def get_initial_state(self):
        return np.zeros((6, 7, 2 + 1), dtype=np.float32) # Final plane is a turn indicator

    # Returns a 7-item boolean array indicating open slots. 
    def get_available_actions(self, s):
        pieces = s[:, :, :2].sum(axis=-1)
        counts = pieces.sum(axis=0)
        return counts != 6
        
    # Drop a red or yellow piece in a slot.
    def take_action(self, s, a):
        p = self.get_player(s)
        s = s.copy()
        pieces = s[:, :, :2].sum(axis=-1)
        col = np.argwhere(a.astype(np.float32) == 1.).item()
        row = np.argwhere(pieces[:,col] == 0).max()
        s[row, col, p] = 1.
        s[:,:,2] = (s[:,:,2] + 1) % 2 # Toggle player
        return s

    # Check all possible 4-in-a-rows for a win.
    def check_winner(self, s):
        if self.get_available_actions(s).sum() == 0: # Full board, draw
            return -1
        for p in [0, 1]:
            board = s[:,:,p]
            if np.isin(4, correlate2d(board, np.ones((1, 4)), mode="valid")): return p # Horizontal
            if np.isin(4, correlate2d(board, np.ones((4, 1)), mode="valid")): return p # Vertical
            i = np.eye(4)
            if np.isin(4, correlate2d(board, i, mode="valid")): return p # Downward diagonol
            if np.isin(4, correlate2d(board, np.fliplr(i), mode="valid")): return p # Upward diagonol

    # Return 0 for red's turn or 1 for yellow's turn.
    def get_player(self, s):
        return int(s[0,0,2])

    # Fixed constant for Tic-Tac-Toe
    def get_num_players(self):
        return 2

    # Print a human-friendly visualization of the board.
    def visualize(self, s):
        board = np.ones((6,7)).astype(np.object)
        board[:,:] = "_"
        board[s[:,:,0] == 1] = 'x'
        board[s[:,:,1] == 1] = 'o'
        last_line = np.array([str(x) for x in np.arange(7)])
        print(np.concatenate([board, last_line.reshape(1,-1)], axis=0))

