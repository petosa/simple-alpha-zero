import numpy as np
from models.mlp import MLP
from models.minivgg import MiniVGG
from models.smallvgg import SmallVGG
from models.bigvgg import BigVGG
from models.resnet import ResNet
from models.senet import SENet
from games.tictactoe import TicTacToe
from games.connect4 import Connect4
from games.guessit import TwoPlayerGuessIt
from neural_network import NeuralNetwork
from train import Trainer


tictactoe_config = {
    "game": TicTacToe,
    "model": MiniVGG,
    "ckpt_frequency": 1,
    "num_updates": 1000,
    "num_games": 30,
    "weight_decay": 1e-4,
    "lr": 1e-3,
    "cpuct": 3,
    "num_simulations": 50,
    "batch_size": 64,
    "num_threads": 4,
    "cuda": False,
    "verbose": True,
    "num_opponents": 1,
    "resume": False,
}

connect4_config = {
    "game": Connect4,
    "model": SmallVGG,
    "ckpt_frequency": 10,
    "num_updates": 100,
    "num_games": 30,
    "weight_decay": 1e-4,
    "lr": 1e-3,
    "cpuct": 3,
    "num_simulations": 50,
    "batch_size": 64,
    "num_threads": 2,
    "cuda": False,
    "verbose": False,
    "num_opponents": 1,
    "resume": False,
}

connect4_config_cuda = {
    "game": Connect4,
    "model": SENet,
    "ckpt_frequency": 5,
    "num_updates": 100,
    "num_games": 30,
    "weight_decay": 1e-4,
    "lr": 1e-6,
    "cpuct": 3,
    "num_simulations": 50,
    "batch_size": 64,
    "num_threads": 1,
    "cuda": True,
    "verbose": False,
    "num_opponents": 1,
    "resume": False,
}

# Please select your config
#################################
config = connect4_config_cuda
#################################

# Instantiate
game = config["game"]()
nn = NeuralNetwork(game=game, model_class=config["model"], lr=config["lr"],
    weight_decay=config["weight_decay"], batch_size=config["batch_size"], cuda=config["cuda"])
pi = Trainer(game=game, nn=nn, num_simulations=config["num_simulations"],
num_games=config["num_games"], num_updates=config["num_updates"], cpuct=config["cpuct"],
num_threads=config["num_threads"])

# Logic for resuming training
checkpoints = nn.list_checkpoints()
if config["resume"]:
    if len(checkpoints) == 0:
        print("No existing checkpoints to resume.")
        quit()
    iteration = checkpoints[-1]
    pi.training_data, pi.error_log = nn.load(iteration, load_supplementary_data=True)
else:
    if len(checkpoints) != 0:
        print("Please delete the existing checkpoints for this game+model combination, or change resume flag to True.")
        quit()
    iteration = 0

# Training loop
while True:
    for _ in range(config["ckpt_frequency"]):
        if config["verbose"]: print("Iteration:",iteration)
        pi.policy_iteration(verbose=config["verbose"]) # One iteration of PI
        iteration += 1
        if config["verbose"]: print("Training examples:", len(pi.training_data))
    
    nn.save(name=iteration, training_data=pi.training_data, error_log=pi.error_log)
    pi.evaluate_against_uninformed(10, config["num_opponents"])
    pi.evaluate_against_uninformed(20, config["num_opponents"])
    pi.evaluate_against_uninformed(40, config["num_opponents"])
    pi.evaluate_against_uninformed(80, config["num_opponents"])
    pi.evaluate_against_uninformed(160, config["num_opponents"])
    pi.evaluate_against_uninformed(320, config["num_opponents"])
    pi.evaluate_against_uninformed(640, config["num_opponents"])
    pi.evaluate_against_uninformed(1280, config["num_opponents"])
