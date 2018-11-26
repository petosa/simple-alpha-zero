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
import torch
import time


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
    "batch_size": 64,
    "num_threads": 4,
    "cuda": False,
    "verbose": False
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
    "verbose": False
}

connect4_config_cuda = {
    "game": Connect4,
    "model": SENet,
    "ckpt_frequency": 5,
    "num_updates": 100,
    "num_games": 10,
    "weight_decay": 1e-4,
    "lr": 1e-6,
    "cpuct": 3,
    "num_simulations": 50,
    "batch_size": 64,
    "num_threads": 1,
    "cuda": True,
    "verbose": False
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

# Training loop
iteration = 0
while True:
    for _ in range(config["ckpt_frequency"]):
        pi.policy_iteration(verbose=config["verbose"]) # One iteration of PI
        iteration += 1
        if config["verbose"]:
            print(nn.latest_loss, len(pi.training_data))
    
    nn.save(name=iteration)
    pi.evaluate_against_uninformed(10)
    pi.evaluate_against_uninformed(20)
    pi.evaluate_against_uninformed(40)
    pi.evaluate_against_uninformed(80)
    pi.evaluate_against_uninformed(160)
    pi.evaluate_against_uninformed(320)
    pi.evaluate_against_uninformed(640)
    pi.evaluate_against_uninformed(1280)