import numpy as np
from models.mlp import MLP
from models.minivgg import MiniVGG
from games.tictactoe import TicTacToe
from games.guessit import TwoPlayerGuessIt
from neural_network import NeuralNetwork
from train import Trainer


tictactoe_config = {
    "game": TicTacToe,
    "model": MiniVGG,
    "ckpt_frequency": 10,
    "num_updates": 100,
    "num_games": 100,
    "weight_decay": 1e-4,
    "lr": 1e-3,
    "cpuct": 3,
    "num_simulations": 15,
    "batch_size": 64
}

# Please select your config
#################################
config = tictactoe_config
#################################

# Instantiate
game = config["game"]()
nn = NeuralNetwork(game=game, model_class=config["model"], lr=config["lr"],
    weight_decay=config["weight_decay"], batch_size=config["batch_size"])
pi = Trainer(game=game, nn=nn, num_simulations=config["num_simulations"],
num_games=config["num_games"], num_updates=config["num_updates"], cpuct=config["cpuct"])

# Training loop
iteration = 0
while True:
    for _ in range(config["ckpt_frequency"]):
        pi.policy_iteration() # One iteration of PI
        iteration += 1
    
    nn.save(name=iteration)
    pi.evaluate_against_uninformed(15)
    pi.evaluate_against_uninformed(100)
    pi.evaluate_against_uninformed(500)
    pi.evaluate_against_uninformed(1000)
    #pi.evaluate_against_uninformed(10000)

    template = np.zeros_like(game.get_available_actions(game.get_initial_state()))
    template[0,0] = 1
    second = game.take_action(game.get_initial_state(), template)
    pred = nn.predict(second)
    # Report statistics
    print(pred[0][3], nn.latest_loss, len(pi.training_data))
    


