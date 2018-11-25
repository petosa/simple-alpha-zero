
from models.smallvgg import SmallVGG
from neural_network import NeuralNetwork
from games.connect4 import Connect4
from players.deep_mcts_player import DeepMCTSPlayer
from players.uninformed_mcts_player import UninformedMCTSPlayer
from play import play_match



# Tracks the current best checkpoint across all checkpoints
def rank_checkpoints(game, model_class, sims):
    winning_model = NeuralNetwork(game, model_class)
    contending_model = NeuralNetwork(game, model_class)
    current_winner = 10
    contender = current_winner
    while True:  
        winning_model.load(current_winner)
        contending_model.load(contender)
        winning_player = DeepMCTSPlayer(game, winning_model, sims)
        contending_player = DeepMCTSPlayer(game, contending_model, sims)
        game1 = [-1, 1, 0][play_match(game, [winning_player, contending_player], verbose=False)]
        game2 = [1, -1, 0][play_match(game, [contending_player, winning_player], verbose=False)]
        if game1 + game2 > 0:
            current_winner = contender
        print(current_winner, game1, game2)
        contender += 10


# Plays the given checkpoint against all other checkpoints and logs upsets.
def one_vs_all(checkpoint, game, model_class, sims):
    my_model = NeuralNetwork(game, model_class)
    my_model.load(checkpoint)
    contending_model = NeuralNetwork(game, model_class)
    contender = 10
    while True:
        contending_model.load(contender)
        my_player = DeepMCTSPlayer(game, my_model, sims)
        contending_player = DeepMCTSPlayer(game, contending_model, sims)
        game1 = [1, -1, 0][play_match(game, [my_player, contending_player], verbose=False)]
        game2 = [-1, 1, 0][play_match(game, [contending_player, my_player], verbose=False)]

        outcome = game1 + game2
        if outcome < 0:
            print("LOSE", contender, game1, game2)
        elif outcome == 0:
            print("TIE", contender, game1, game2)
        else:
            print("WIN", contender, game1, game2)

        contender += 10


# Finds the effective MCTS strength of a checkpoint
# Also presents a control at each checkpoint - that is, the result
# if you had used no heuristic but the same power.
def effective_model_power(checkpoint, game, model_class, sims):
    my_model = NeuralNetwork(game, model_class)
    my_model.load(checkpoint)
    my_player = DeepMCTSPlayer(game, my_model, sims)
    strength = 10
    while True: 

        # Play main game
        contending_player = UninformedMCTSPlayer(game, strength)
        main1 = [1, -1, 0][play_match(game, [my_player, contending_player], verbose=False)]
        main2 = [-1, 1, 0][play_match(game, [contending_player, my_player], verbose=False)]        
        main_outcome = main1 + main2
        if main_outcome < 0:
            print("LOSE", strength, end=" ")
        elif main_outcome == 0:
            print("TIE", strength, end=" ")
        else:
            print("WIN", strength, end=" ")

        # Play control game
        control_player = UninformedMCTSPlayer(game, sims)
        control1 = [1, -1, 0][play_match(game, [control_player, contending_player], verbose=False)]
        control2 = [-1, 1, 0][play_match(game, [contending_player, control_player], verbose=False)]
        control_outcome = control1 + control2
        if control_outcome < 0:
            print("(Control: LOSE)")
        elif control_outcome == 0:
            print("(Control: TIE)")
        else:
            print("(Control: WIN)")

        strength *= 2


#rank_checkpoints()
#one_vs_all(1140)
effective_model_power(1140, game=Connect4(), model_class=SmallVGG, sims=100)