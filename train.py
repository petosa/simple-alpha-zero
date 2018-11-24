import numpy as np
from mcts import MCTS
from play import play_match
from players.uninformed_mcts_player import UninformedMCTSPlayer
from players.deep_mcts_player import DeepMCTSPlayer

class Trainer:

    def __init__(self, game, nn, simulations, cpuct=1):
        self.game = game
        self.nn = nn
        self.simulations = simulations
        self.training_data = np.zeros((0,3))
        self.cpuct = cpuct


    # Does one game of self play and generates training samples
    def self_play(self, temperature):
        s = self.game.get_initial_state()
        tree = MCTS(self.game, self.nn)

        data = []
        w = None
        while w is None:

            # Think
            for _ in range(self.simulations):
                tree.simulate(s, cpuct=self.cpuct)

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

        # Update training examples with outcome
        data = np.array(data)
        if w == -1:
            data[:,-1] = 0
        else:
            data[data[:,0] == w, -1] = 1
            data[data[:,0] != w, -1] = -1

        return data[:,1:]


    def policy_iteration(self):
        temperature = 1
        for _ in range(100): # Self-play games
            new_data = self.self_play(temperature)
            self.training_data = np.concatenate([self.training_data, new_data], axis=0)
        #self.training_data = self.training_data[-200000:,:]
        self.nn.train(self.training_data)


    def evaluate_against_uninformed(self, uninformed_simulations):
        uninformed, informed = UninformedMCTSPlayer(self.game, uninformed_simulations), DeepMCTSPlayer(self.game, self.nn, self.simulations)
        first = play_match(self.game, [informed, uninformed])
        first = ["Win", "Lose", "Tie"][first]
        second = play_match(self.game, [uninformed, informed])
        second = ["Lose", "Win", "Tie"][second]
        print("Opponent strength: {}     When I play first: {}     When I play second: {}".format(uninformed_simulations,first, second))

