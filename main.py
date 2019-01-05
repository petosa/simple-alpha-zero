import json
import sys
import numpy as np
from models.mlp import MLP
from models.minivgg import MiniVGG
from models.smallvgg import SmallVGG
from models.bigvgg import BigVGG
from models.resnet import ResNet
from models.senet import SENet
from games.tictactoe import TicTacToe
from games.leapfrog import ThreePlayerLeapFrog
from games.connect4 import Connect4
from games.guessit import TwoPlayerGuessIt
from neural_network import NeuralNetwork
from train import Trainer


# Load in a run configuration
with open(sys.argv[1], "r") as f:
    config = json.loads(f.read())

# Instantiate
game = globals()[config["game"]]()
model_class = globals()[config["model"]]
nn = NeuralNetwork(game=game, model_class=model_class, lr=config["lr"],
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
    pi.evaluate_against_uninformed(10)
    pi.evaluate_against_uninformed(20)
    pi.evaluate_against_uninformed(40)
    pi.evaluate_against_uninformed(80)
    pi.evaluate_against_uninformed(160)
    pi.evaluate_against_uninformed(320)
    pi.evaluate_against_uninformed(640)
    pi.evaluate_against_uninformed(1280)
