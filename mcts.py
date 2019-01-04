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
    def simulate(self, s, cpuct=1, epsilon_fix=True):
        hashed_s = self.np_hash(s) # Key for state in dictionary
        current_player = self.game.get_player(s)
        if hashed_s in self.tree: # Not at leaf; select.
            stats = self.tree[hashed_s]
            N, Q, P = stats[:,1], stats[:,2], stats[:,3]
            U = cpuct*P*math.sqrt(N.sum() + (1e-6 if epsilon_fix else 0))/(1 + N)
            heuristic = Q + U
            best_a_idx = np.argmax(heuristic)
            best_a = stats[best_a_idx, 0] # Pick best action to take
            template = np.zeros_like(self.game.get_available_actions(s)) # Submit action to get s'
            template[tuple(best_a)] = 1
            s_prime = self.game.take_action(s, template)
            v, winning_player = self.simulate(s_prime) # Forward simulate with this action
            n, q = N[best_a_idx], Q[best_a_idx]
            adj_v = v if current_player == winning_player else -v
            stats[best_a_idx, 2] = (n*q+adj_v)/(n + 1)
            stats[best_a_idx, 1] += 1
            return v, winning_player


        else: # Expand
            w = self.game.check_winner(s)
            if w is not None: # Reached a terminal node
                return 1 if w is not -1 else 0, w # Someone won, or tie
            available_actions = self.game.get_available_actions(s)
            idx = np.stack(np.where(available_actions)).T
            p, v = self.nn.predict(s)
            stats = np.zeros((len(idx), 4), dtype=np.object)
            stats[:,-1] = p
            stats[:,0] = list(idx)
            self.tree[hashed_s] = stats
            return v, current_player


    # Returns the MCTS policy distribution for the given state
    def get_distribution(self, s, temperature):
        hashed_s = self.np_hash(s)
        stats = self.tree[hashed_s][:,:2].copy()
        N = stats[:,1]
        try:
            raised = np.power(N, 1/temperature)
        except (ZeroDivisionError, OverflowError):
            raised = np.zeros_like(N)
            raised[N.argmax()] = 1
        
        total = raised.sum()
        if total == 0:
            raised[:] = 1
            total = raised.sum()
        dist = raised/total
        stats[:,1] = dist
        return stats
        

