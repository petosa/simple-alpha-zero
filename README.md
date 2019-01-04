# What is this
An independent, general implementation of DeepMind's AlphaZero algorithm.
AlphaZero is a deep reinforcement learning algorithm which can learn to master a certain class of adversarial games through self-play.
Strategies are learned *tabula rasa* and, with enough time and computation, achieve super-human performance.
The canonical AlphaZero algorithm is intended for 2-player games like Chess and Go, though this project supports multiplayer games as well.

# Benefits

### Clean & Simple
Clear and concise - a no-frills AlphaZero implementation written with Python 3 and PyTorch.
Extensively commented and easy to extend.
Support for CPU and GPU training, as well as pausing and resuming.

### Fully Tested
A suite of comprehensive unit tests validate the correctness of each AlphaZero component.

### Modular & Extensible
Easily plug in your own games or neural networks by implementing the Game and Model interface. Currently, only PyTorch models are supported. Example games and networks included with this repo are listed below.

#### Example Games
- Tic-Tac-Toe
- Connect 4

#### Example Networks
- SENet

### Novel Multiplayer Support
The first of its kind; support for games with more than 2 players.
