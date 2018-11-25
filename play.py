

def play_match(game, players, verbose=False):
    s = game.get_initial_state()
    if verbose: game.friendly_print(s)
    turn = 0
    while game.check_winner(s) is None:
        p_num = turn % len(players)
        p = players[p_num]
        if verbose: print("Player {}'s turn.".format(p_num))

        s = p.update_state(s)
        if verbose: game.friendly_print(s)
        turn += 1
    return game.check_winner(s)


if __name__ == "__main__":
    from players.human_player import HumanPlayer
    from players.uninformed_mcts_player import UninformedMCTSPlayer
    from players.deep_mcts_player import DeepMCTSPlayer
    from games.tictactoe import TicTacToe
    from games.guessit import OnePlayerGuessIt, TwoPlayerGuessIt
    from games.leapfrog import ThreePlayerLeapFrog, ThreePlayerLinearLeapFrog
    from games.connect4 import Connect4
    from neural_network import NeuralNetwork
    from models.mlp import MLP
    from models.minivgg import MiniVGG
    from models.smallvgg import SmallVGG

    t = TicTacToe()
    gi1 = OnePlayerGuessIt()
    gi2 = TwoPlayerGuessIt()
    llf3 = ThreePlayerLinearLeapFrog()
    lf3 = ThreePlayerLeapFrog()
    c4 = Connect4()
    
    #######################
    # Human v Human games #
    #######################

    #h1, h2 = HumanPlayer(t), HumanPlayer(t)
    #r = play_match(t, [h1, h2], verbose=True)

    #h1 = HumanPlayer(gi1)
    #r = play_match(gi1, [h1], verbose=True)

    #h1, h2 = HumanPlayer(gi2), HumanPlayer(gi2)
    #r = play_match(gi2, [h1, h2], verbose=True)

    #h1, h2, h3 = HumanPlayer(llf3), HumanPlayer(llf3), HumanPlayer(llf3)
    #r = play_match(llf3, [h1, h2, h3], verbose=True)

    #h1, h2, h3 = HumanPlayer(lf3), HumanPlayer(lf3), HumanPlayer(lf3)
    #r = play_match(lf3, [h1, h2, h3], verbose=True)

    #h1, h2 = HumanPlayer(c4), HumanPlayer(c4)
    #r = play_match(c4, [h1, h2], verbose=True)

    #######################
    # Human v UMCTS games #
    #######################

    #h1, u1 = HumanPlayer(t), UninformedMCTSPlayer(t, 10000)
    #r = play_match(t, [h1, u1], verbose=True)

    #h1, u1, u2 = HumanPlayer(lf3), UninformedMCTSPlayer(llf3, 10), UninformedMCTSPlayer(lf3, 10)
    #r = play_match(lf3, [h1, u1, u2], verbose=True)

    #h1, u1= HumanPlayer(c4), UninformedMCTSPlayer(c4, 50)
    #r = play_match(c4, [u1, h1], verbose=True)

    #######################
    # Human v DMCTS games #
    #######################

    #nn = NeuralNetwork(t, MiniVGG)
    #nn.load("500")
    #h1, d1 = HumanPlayer(t), DeepMCTSPlayer(t, nn, 15)
    #r = play_match(t, [h1, d1], verbose=True)

    nn = NeuralNetwork(c4, SmallVGG)
    nn.load("1140")
    h1, d1 = HumanPlayer(c4), DeepMCTSPlayer(c4, nn, 50)
    r = play_match(c4, [d1, h1], verbose=True)


    #######################
    # UMCTS v UMCTS games #
    #######################

    #h1, h2 = UninformedMCTSPlayer(c4, 50), UninformedMCTSPlayer(c4, 1000)
    #r = play_match(c4, [h1, h2], verbose=True)

    '''
    import numpy as np
    import matplotlib.pyplot as plt
    log = []
    for i in range(10, 1000, 100):
        for j in range(10, 1000, 100):
            t = TicTacToe()
            u1, u2 = UninformedMCTSPlayer(t, i), UninformedMCTSPlayer(t, j)
            r = play_match(t, [u1, u2], verbose=False)
            log.append(r)
            print(r)
    log = np.array(log)
    n = int(len(log)**.5)
    log = log.reshape(n, n)
    log += 1
    plt.imshow(log, cmap="gray", origin="lower")
    plt.xlabel("O's AI strength")
    plt.ylabel("X's AI strength")
    plt.show()
    print(log.shape)
    '''

    if r == -1:
        print("Tie game.")
    else:
        print("Player {} wins!".format(r))



    