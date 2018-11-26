
import os
from models.smallvgg import SmallVGG
from models.minivgg import MiniVGG
from neural_network import NeuralNetwork
from games.connect4 import Connect4
from games.tictactoe import TicTacToe
from players.deep_mcts_player import DeepMCTSPlayer
from players.uninformed_mcts_player import UninformedMCTSPlayer
from play import play_match



# Tracks the current best checkpoint across all checkpoints
def rank_checkpoints(game, model_class, sims):
    winning_model = NeuralNetwork(game, model_class)
    contending_model = NeuralNetwork(game, model_class)
    path = "checkpoints/{}-{}/".format(game.__class__.__name__, model_class.__name__)
    ckpts = sorted([int(filename.split(".ckpt")[0]) for filename in os.listdir(path) if filename.endswith(".ckpt")])
    current_winner = ckpts[0]

    for contender in ckpts:
        winning_model.load(current_winner)
        contending_model.load(contender)
        winning_player = DeepMCTSPlayer(game, winning_model, sims)
        contending_player = DeepMCTSPlayer(game, contending_model, sims)
        outcome = play_match(game, [contending_player, winning_player], verbose=False, permute=True)[contending_player]
        if outcome >= 0:
            current_winner = contender
        print("Current Champion:", current_winner, "Challenger:", contender, "Outcome:", outcome)


# Plays the given checkpoint against all other checkpoints and logs upsets.
def one_vs_all(checkpoint, game, model_class, sims):
    my_model = NeuralNetwork(game, model_class)
    my_model.load(checkpoint)
    contending_model = NeuralNetwork(game, model_class)
    path = "checkpoints/{}-{}/".format(game.__class__.__name__, model_class.__name__)
    ckpts = sorted([int(filename.split(".ckpt")[0]) for filename in os.listdir(path) if filename.endswith(".ckpt")])

    for contender in ckpts:
        contending_model.load(contender)
        my_player = DeepMCTSPlayer(game, my_model, sims)
        contending_player = DeepMCTSPlayer(game, contending_model, sims)
        outcome = play_match(game, [my_player, contending_player], verbose=False, permute=True)[my_player]

        if outcome < 0:
            print("LOSE", contender, outcome)
        elif outcome == 0:
            print("TIE", contender, outcome)
        else:
            print("WIN", contender, outcome)


# Finds the effective MCTS strength of a checkpoint
# Also presents a control at each checkpoint - that is, the result
# if you had used no heuristic but the same power.
def effective_model_power(checkpoint, game, model_class, sims):
    my_model = NeuralNetwork(game, model_class)
    my_model.load(checkpoint)
    my_player = DeepMCTSPlayer(game, my_model, sims)
    strength = 10
    lost = False

    while not lost: 

        # Play main game
        contending_player = UninformedMCTSPlayer(game, strength)
        main = play_match(game, [my_player, contending_player], verbose=False, permute=True)[my_player]
        if main < 0:
            print("LOSE", strength, end=" ")
            lost = True
        elif main == 0:
            print("TIE", strength, end=" ")
        else:
            print("WIN", strength, end=" ")

        # Play control game
        control_player = UninformedMCTSPlayer(game, sims)
        control = play_match(game, [control_player, contending_player], verbose=False, permute=True)[control_player]
        if control < 0:
            print("(Control: LOSE)")
        elif control == 0:
            print("(Control: TIE)")
        else:
            print("(Control: WIN)")

        strength *= 2


if __name__ == "__main__":
    checkpoint = 1810
    game = Connect4()
    model_class = SmallVGG
    sims = 100
    
    #rank_checkpoints(game, model_class, sims)
    #one_vs_all(checkpoint, game, model_class, sims)
    effective_model_power(checkpoint, game, model_class, sims)