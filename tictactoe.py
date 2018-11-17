import numpy as np
from game import Game


class TicTacToe(Game):

    def get_initial_state(self):
        return np.zeros((3, 3, 2 + 1), dtype=np.float32) # Final plane is a turn indicator

    def get_available_actions(self, s):
        return s[:, :, :2].sum(axis=-1) == 0
        
    # Step from state s with action a
    def take_action(self, s, a):
        p = int(s[0,0,2])
        s = s.copy()
        s[:,:,p] += a.astype(np.float32) # Next move
        s[:,:,2] = (s[:,:,2] + 1) % 2 # Toggle player
        return s

    # Check the board for a winner (0/1), draw (-1), or incomplete (None)
    def check_winner(self, s):
        for p in [0, 1]:
            board = s[:,:,p]
            if np.isin(3, board.sum(axis=0)) or np.isin(3, board.sum(axis=1)): # Verticals & horizontals
                return p
            elif board[np.eye(3).astype(np.bool)].sum() == 3 or board[np.fliplr(np.eye(3)).astype(np.bool)].sum() == 3: # Diagonals
                return p
        if self.get_available_actions(s).sum() == 0: # Full board, draw
            return -1

    def get_player(self, s):
        return int(s[0,0,2])

    # Print a human-friendly visualization of the board.
    def friendly_print(self, s):
        board = np.ones((3,3)).astype(np.object)
        board[:,:] = " "
        board[s[:,:,0] == 1] = 'x'
        board[s[:,:,1] == 1] = 'o'
        print(board)

        

if __name__ == "__main__":

    def play(hvh=False):         
        t = TicTacToe()
        state = t.get_initial_state()
        t.friendly_print(state)

        turn = 0
        winner = None
        while winner is None:

            p = t.get_player(state)
            print("Turn {} (Player {})".format(turn, p))

            if hvh:
                # Human vs human play
                loop = True
                while loop:
                    action = input()
                    idx = action.split(" ")
                    template = np.zeros((3,3))
                    action = (int(idx[0]) - 1, int(idx[1]) - 1)
                    loop = ~t.get_available_actions(state)[action]
            else:
                # Sample action from distribution
                mask = t.get_available_actions(state)
                action_dist = np.random.rand(3,3)
                action_dist[~mask] = 0
                action_dist[mask] /= action_dist[mask].sum()
                flat_action = np.random.choice(np.arange(action_dist.size), p=action_dist.reshape(-1))
                action = np.unravel_index(flat_action, action_dist.shape)
                # Perhaps do argmax instead? Comparative study.

            # Publish action
            template = np.zeros((3,3))
            template[action] = 1
            state = t.take_action(state, template)

            # See outcome
            t.friendly_print(state)
            winner = t.check_winner(state)
            turn += 1

        print("And the winner is Player {}!".format(winner))
        return winner


    score = {0:0, 1:0, -1:0}
    for _ in range(100):
        score[play()] += 1
    print(score)

    #play(True)
