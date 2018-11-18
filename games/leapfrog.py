import numpy as np
import sys
sys.path.append("..")
from game import Game

# I have invented a game called "leap-frog" which is pretty simple.
# You have a single array which is N spaces long.
# Each player takes turns hopping 1, 2, or 3 spaces forward.
# You hop starting from where the last player left off.
# The winner is whichever player lands in the last space.

class ThreePlayerLeapFrog(Game):
    def __init__(self): self.n = 10
    def get_initial_state(self):
        arr = np.zeros((self.n, 1, 2), dtype=np.float32)
        arr[0,0,0] = 1 # Start all the way on the left.
        return arr
    def get_available_actions(self, s):
        s = s.copy()
        path = s[:,0,0]
        i = np.argwhere(path == 1)[0,0]
        path[i] = 0
        path[i+1:i+4] = 1
        return path.astype(np.bool)
    def check_winner(self, s):
        return None if s[-1,0,0] == 0 else (int(s[0,0,1]) - 1) % 3
    def take_action(self, s, a):
        s = s.copy()
        s[:,:,0] = 0
        s[:,0,0] += a.astype(np.float32) # Next move
        s[:,:,1] = (s[:,:,1] + 1) % 3 # Toggle player
        return s
    def get_player(self, s):
        return int(s[0,0,1])
