import numpy as np
import sys
from scipy.signal import correlate2d
sys.path.append("..")
from game import Game


class Connect4(Game):

    def get_initial_state(self):
        return np.zeros((6, 7, 2 + 1), dtype=np.float32) # Final plane is a turn indicator

    def get_available_actions(self, s):
        pieces = s[:, :, :2].sum(axis=-1)
        counts = pieces.sum(axis=0)
        return counts != 6
        
    # Step from state s with action a
    def take_action(self, s, a):
        p = self.get_player(s)
        s = s.copy()
        pieces = s[:, :, :2].sum(axis=-1)
        col = np.argwhere(a.astype(np.float32) == 1.).item()
        row = np.argwhere(pieces[:,col] == 0).max()
        s[row, col, p] = 1.
        s[:,:,2] = (s[:,:,2] + 1) % 2 # Toggle player
        return s

    # Check the board for a winner (0/1), draw (-1), or incomplete (None)
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

        

    def get_player(self, s):
        return int(s[0,0,2])

    # Print a human-friendly visualization of the board.
    def friendly_print(self, s):
        board = np.ones((6,7)).astype(np.object)
        board[:,:] = "_"
        board[s[:,:,0] == 1] = 'x'
        board[s[:,:,1] == 1] = 'o'
        last_line = np.array([str(x) for x in np.arange(7)])
        print(np.concatenate([board, last_line.reshape(1,-1)], axis=0))

