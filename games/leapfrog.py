import numpy as np
import sys
sys.path.append("..")
from game import Game

# I have invented a game called "leap-frog" which is pretty simple.
# You have a single array which is N spaces long.
# Each player takes turns hopping 1, 2, or 3 spaces forward.
# You hop starting from where the last player left off.
# The winner is whichever player lands in the last space.

# This version of leapfrog only supports hopping 1 space forward.
class ThreePlayerLinearLeapFrog(Game):
    def __init__(self): self.n = 6
    def get_initial_state(self):
        arr = np.zeros((self.n, 1, 4), dtype=np.float32)
        arr[0,0,0] = 1 # Start all the way on the left.
        arr[:,:,1] = 1 # Player 1 starts
        return arr
    def get_available_actions(self, s):
        path = s[:,0,0]
        i = np.argwhere(path == 1)[0,0]
        spots_left = len(path) - i - 1
        available = np.zeros(1)
        if spots_left > 0:
            available[0] = 1
        return available.astype(np.bool)
    def check_winner(self, s):
        return None if s[-1,0,0] == 0 else (self.get_player(s) - 1) % 3
    def take_action(self, s, a):
        s = s.copy()
        path = s[:,0,0]
        i = np.argwhere(path == 1)[0]
        path = s[:,0,0] = 0
        s[i+1,0,0] = a.astype(np.float32) # Next move
        p = self.get_player(s) + 1
        s[:,:,p] = 0
        s[:,:,(p%3) + 1] = 1 # Toggle player
        return s
    def get_player(self, s):
        vec = s[0,0,1:]
        return int(np.where(vec==1)[0][0])
    def get_num_players(self): return 3
    def visualize(self, s):
        path = s[:,0, 0]
        board = np.zeros_like(path, dtype=np.object)
        board[:] = "_"
        if s[0,0,0] == 1:
            token = 'start'
        elif self.get_player(s) == 1:
            token = 'a'
        elif self.get_player(s) == 2:
            token = 'b'
        elif self.get_player(s) == 0:
            token = 'c'
        board[path == 1] = token
        print(board)

# This version of leapfrog supports hopping 1, 2 or 3 spaces forward.
class ThreePlayerLeapFrog(Game):
    def __init__(self): self.n = 10
    def get_initial_state(self):
        arr = np.zeros((self.n, 1, 4), dtype=np.float32)
        arr[0,0,0] = 1 # Start all the way on the left.
        arr[:,:,1] = 1 # Player 1 starts
        return arr
    def get_available_actions(self, s):
        path = s[:,0,0]
        i = np.argwhere(path == 1)[0,0]
        spots_left = len(path) - i - 1
        available = np.zeros(3)
        available[:spots_left] = 1
        return available.astype(np.bool)
    def check_winner(self, s):
        return None if s[-1,0,0] == 0 else (self.get_player(s) - 1) % 3
    def take_action(self, s, a):
        s = s.copy()
        path = s[:,0,0]
        i = np.argwhere(path == 1)[0,0]
        j = np.argwhere(a == 1)[0] + 1
        path = s[:,0,0] = 0
        s[i+j,0,0] = 1.0 # Next move
        p = self.get_player(s) + 1
        s[:,:,p] = 0
        s[:,:,(p%3) + 1] = 1 # Toggle player
        return s
    def get_player(self, s):
        vec = s[0,0,1:]
        return int(np.where(vec==1)[0][0])
    def get_num_players(self): return 3
    def visualize(self, s):
        path = s[:,0, 0]
        board = np.zeros_like(path, dtype=np.object)
        board[:] = "_"
        if s[0,0,0] == 1:
            token = 'start'
        elif self.get_player(s) == 1:
            token = 'a'
        elif self.get_player(s) == 2:
            token = 'b'
        elif self.get_player(s) == 0:
            token = 'c'
        board[path == 1] = token
        print(board)
