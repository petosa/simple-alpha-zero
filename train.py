import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from mcts import MCTS
from play import play_match
from players.uninformed_mcts_player import UninformedMCTSPlayer
from players.deep_mcts_player import DeepMCTSPlayer
import time

class Trainer:

    def __init__(self, game, nn, num_simulations,  num_games, num_updates, cpuct, num_threads):
        self.game = game
        self.nn = nn
        self.num_simulations = num_simulations
        self.num_updates = num_updates
        self.num_games = num_games
        self.training_data = np.zeros((0,3))
        self.cpuct = cpuct
        self.num_threads = num_threads
        self.error_log = []


    # Does one game of self play and generates training samples
    def self_play(self, temperature):
        s = self.game.get_initial_state()
        tree = MCTS(self.game, self.nn)

        data = []
        w = None
        while w is None:
            
            # Think
            for _ in range(self.num_simulations):
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


    # Performs one iteration of policy improvement.
    # Creates some number of games, then updates network parameters some number of times.
    def policy_iteration(self, verbose=False):
        temperature = 1   

        if verbose:
            print("SIMULATING " + str(self.num_games) + " games")
            start = time.clock()
        if self.num_threads > 1:
            jobs = [temperature]*self.num_games
            pool = ThreadPool(self.num_threads) 
            new_data = pool.map(self.self_play, jobs)
            pool.close() 
            pool.join() 
            self.training_data = np.concatenate([self.training_data] + new_data, axis=0)
        else:
            for _ in range(self.num_games): # Self-play games
                new_data = self.self_play(temperature)
                self.training_data = np.concatenate([self.training_data, new_data], axis=0)
        if verbose:
            print("Simulating took " + str(int(time.clock()-start)) + " seconds")


        # self.training_data = self.training_data[-200000:,:]
        if verbose:
            print("TRAINING")
            start = time.clock()
        mean_loss = None
        count = 0
        for _ in range(self.num_updates):
            self.nn.train(self.training_data)
            new_loss = self.nn.latest_loss.item()
            if mean_loss is None:
                mean_loss = new_loss
            else:
                (mean_loss*count + new_loss)/(count+1)
            count += 1
        self.error_log.append(mean_loss)

        if verbose:
            print("Training took " + str(int(time.clock()-start)) + " seconds")
            print("Average train error:", mean_loss)


    def evaluate_against_uninformed(self, uninformed_simulations):
        num_opponents = self.game.get_num_players() - 1
        uninformeds = [UninformedMCTSPlayer(self.game, uninformed_simulations) for _ in range(num_opponents)]
        informed = DeepMCTSPlayer(self.game, self.nn, self.num_simulations)
        scores, outcomes = play_match(self.game, [informed] + uninformeds, permute=True)
        score, outcome = scores[informed], outcomes[informed]
        print("Opponent strength: {}     My win rate: {} ({})".format(uninformed_simulations, round(score, 3), outcome))

