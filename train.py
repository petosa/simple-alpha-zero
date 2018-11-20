import numpy as np
from mcts import MCTS
from play import play_match
from players.uninformed_mcts_player import UninformedMCTS
from players.deep_mcts_player import DeepMCTS

class Trainer:

    def __init__(self, game, nn, simulations, verbose=False):
        self.game = game
        self.nn = nn
        self.simulations = simulations
        self.verbose = verbose


    # Does one game of self play and generates training samples
    def self_play(self, temperature):
        s = self.game.get_initial_state()
        tree = MCTS(self.game, self.nn)
        #if self.verbose: self.game.friendly_print(s)

        data = []
        w = None
        while w is None:

            # Think
            for _ in range(self.simulations):
                tree.simulate(s)

            # Fetch action distribution and append training example template.
            dist = tree.get_distribution(s, temperature=temperature)
            data.append([self.game.get_player(s), s, dist[:,1], None]) # player, state, prob, outcome

            # Sample an action
            idx = np.random.choice(len(dist), p=dist[:,1].astype(np.float))
            a = tuple(dist[idx, 0])

            # Apply action
            available = self.game.get_available_actions(s)
            template = np.zeros_like(available)
            template[a] = 1
            s = self.game.take_action(s, template)

            # Check winner
            w = self.game.check_winner(s)
            #if self.verbose: self.game.friendly_print(s)

        # Update training examples with outcome
        data = np.array(data)
        if w == -1:
            data[:,-1] = 0
        else:
            data[data[:,0] == w, -1] = 1
            data[data[:,0] != w, -1] = -1

        return data[:,1:]


    def policy_iteration(self):
        
        temperature = .5
        training_data = []
        for _ in range(10): # Self-play games
            training_data.append(self.self_play(temperature))
        training_data = np.concatenate(training_data, axis=0)
        print(training_data.shape)
        self.nn.train(training_data)
        uninformed, informed = UninformedMCTS(self.game, self.simulations), DeepMCTS(self.game, self.nn, self.simulations)
        
        first = play_match(self.game, [informed, uninformed])
        second = play_match(self.game, [uninformed, informed])
        if self.verbose:
            print("When I play first: {}     When I play second: {}".format(first, second))
            print(self.nn.predict(self.game.get_initial_state()))




if __name__=="__main__":
    from networks.mlp import MLP
    from games.tictactoe import TicTacToe

    t = TicTacToe()
    d = MLP(t)
    
    pi = Trainer(t, d, 100, True)
    for _ in range(100):
        pi.policy_iteration()


