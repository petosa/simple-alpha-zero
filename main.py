
from general.TicTacToe import TicTacToeGame as Game
from general.NNetWrapper import NNetWrapper as nn
from general.MCTS import MCTS
from general.utils import *
from general.Coach import Coach

args = dotdict({
    'numIters': 1,
    'numEps': 100,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__=="__main__":
    ttt = Game(3)
    net = nn(ttt)

    c = Coach(ttt, net, args)

    c.learn()