# What is this
An independent implementation and of DeepMind's AlphaZero algorithm with novel multiplayer support.
AlphaZero is a deep reinforcement learning algorithm which can learn to master a certain class of adversarial games through self-play. Games must be discrete, turn-based, 2-player, and have perfect information.
Strategies are learned *tabula rasa* and, with enough time and computation, achieve super-human performance.
The canonical AlphaZero algorithm is intended for 2-player games like Chess and Go.
The alpha-zero-multiplayer project extends the functionality of DeepMind's AlphaZero to support multiplayer games as well. Some examples of discrete, multiplayer games with perfect information are Blokus, Quatrochess, and Battle Sheep.

# Benefits

### Clean & Simple
Clear and concise - a no-frills AlphaZero implementation powered by Python 3 and PyTorch.
Extensively documented and easy to extend.

### Fully Tested
A suite of comprehensive unit tests validate the correctness of each AlphaZero component.

### Modular & Extensible
Easily plug in your own games or neural networks by implementing the Game and Model abstract classes. Currently, only PyTorch models are supported. Example games and networks included with this repo are listed below.

#### Example Games
- Tic-Tac-Toe
- Connect 4

#### Example Networks
- SENet

### Novel Multiplayer Support
The first of its kind; support for games with more than 2 players. See examples below.

