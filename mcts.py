import math
import numpy as np

# Concerns: Add epsilon amount to UCB evaluation to ensure probability is considered
# Caveat: Q in heuristic might obviate this.
# Concerns: No Dir noise being added. If it is added, tests would break.
# Caveat: Make Dir a switch, write tests that use Dir with fixed seed.

class MCTS():

    def __init__(self, game, nn):
        self.game = game
        self.nn = nn
        self.tree = {}

    def np_hash(self, data):
        return data.tostring()

    # Run a MCTS simulation starting from state s of the tree
    def simulate(self, s, cpuct=1):
        hashed_s = self.np_hash(s) # Key for state in dictionary
        current_player = self.game.get_player(s)
        if hashed_s in self.tree: # Not at leaf; select.
            stats = np.array(list(self.tree[hashed_s].values()))
            u = cpuct*stats[:,3]*math.sqrt(stats[:,1].sum())/(1 + stats[:,1])
            heuristic = stats[:,2] + u # Q + U
            best_a = stats[np.argmax(heuristic), 0] # Pick best action to take
            template = np.zeros_like(self.game.get_available_actions(s)) # Submit action to get s'
            template[tuple(best_a)] = 1
            s_prime = self.game.take_action(s, template)
            v, winning_player = self.simulate(s_prime) # Forward simulate with this action
            edge = self.tree[hashed_s][self.np_hash(best_a)]
            N, Q = edge[1], edge[2]
            adj_v = v if current_player == winning_player else -v
            self.tree[hashed_s][self.np_hash(best_a)][2] = (N*Q+adj_v)/(N + 1)
            self.tree[hashed_s][self.np_hash(best_a)][1] += 1
            return v, winning_player


        else: # Expand
            w = self.game.check_winner(s)
            if w is not None: # Reached a terminal node
                return 1 if w is not -1 else 0, w # Someone won, or tie
            self.tree[hashed_s] = {} # Empty dictionary of children
            available_actions = self.game.get_available_actions(s)
            idx = np.stack(np.where(available_actions)).T
            p, v = self.nn.predict(s)
            norm = p[available_actions].sum()
            for a in idx:
                self.tree[hashed_s][self.np_hash(a)] = [a, 0, 0, p[tuple(a)]/norm] # S', N, Q, P
            return v, current_player

