
import os
from models.smallvgg import SmallVGG
from models.minivgg import MiniVGG
from models.mlp import MLP
from neural_network import NeuralNetwork
from games.connect4 import Connect4
from games.tictactoe import TicTacToe
from games.leapfrog import ThreePlayerLeapFrog
from players.deep_mcts_player import DeepMCTSPlayer
from players.uninformed_mcts_player import UninformedMCTSPlayer
from play import play_match



# Tracks the current best checkpoint across all checkpoints
def rank_checkpoints(game, model_class, sims):
    winning_model = NeuralNetwork(game, model_class)
    contending_model = NeuralNetwork(game, model_class)
    ckpts = winning_model.list_checkpoints()
    num_opponents = game.get_num_players() - 1
    current_winner = ckpts[0]

    for contender in ckpts:

        # Load contending player
        contending_model.load(contender)
        contending_player = DeepMCTSPlayer(game, contending_model, sims)

        # Load winning player
        winning_model.load(current_winner)
        winners = [DeepMCTSPlayer(game, winning_model, sims) for _ in range(num_opponents)]
        
        scores, outcomes = play_match(game, [contending_player] + winners, verbose=False, permute=True)
        score, outcome = scores[contending_player], outcomes[contending_player]
        if outcome == "Win":
            current_winner = contender
        print("Current Champion:", current_winner, "Challenger:", contender, "Outcome:", outcome, score)


# Plays the given checkpoint against all other checkpoints and logs upsets.
def one_vs_all(checkpoint, game, model_class, sims):
    my_model = NeuralNetwork(game, model_class)
    my_model.load(checkpoint)
    contending_model = NeuralNetwork(game, model_class)
    ckpts = my_model.list_checkpoints()
    num_opponents = game.get_num_players() - 1

    for contender in ckpts:
        contending_model.load(contender)
        my_player = DeepMCTSPlayer(game, my_model, sims)
        contenders = [DeepMCTSPlayer(game, contending_model, sims) for _ in range(num_opponents)]
        scores, outcomes = play_match(game, [my_player] + contenders, verbose=False, permute=True)
        score, outcome = scores[my_player], outcomes[my_player]
        print("Challenger:", contender, "Outcome:", outcome, score)


# Finds the effective MCTS strength of a checkpoint
# Also presents a control at each checkpoint - that is, the result
# if you had used no heuristic but the same num_simulations.
def effective_model_power(checkpoint, game, model_class, sims):
    my_model = NeuralNetwork(game, model_class)
    my_model.load(checkpoint)
    my_player = DeepMCTSPlayer(game, my_model, sims)
    strength = 10
    num_opponents = game.get_num_players() - 1
    lost = False

    while not lost: 
        contenders = [UninformedMCTSPlayer(game, strength) for _ in range(num_opponents)]

        # Play main game
        scores, outcomes = play_match(game, [my_player] + contenders, verbose=False, permute=True)
        score, outcome = scores[my_player], outcomes[my_player]
        if outcome == "Lose":
            lost = True
        print("{} <{}>      Opponent strength: {}".format(outcome, round(score, 3), strength), end="")

        # Play control game
        control_player = UninformedMCTSPlayer(game, sims)
        scores, outcomes = play_match(game, [control_player] + contenders, verbose=False, permute=True)
        score, outcome = scores[control_player], outcomes[control_player]
        print("      (Control: {} <{}>)".format(outcome, round(score, 3)))

        strength *= 2 # Opponent strength doubles every turn


if __name__ == "__main__":
    checkpoint = 76
    game = TicTacToe()
    model_class = MiniVGG
    sims = 50
    
    #rank_checkpoints(game, model_class, sims)
    #one_vs_all(checkpoint, game, model_class, sims)
    effective_model_power(checkpoint, game, model_class, sims)
