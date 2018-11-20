

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
    from players.human_player import Human
    from players.uninformed_mcts_player import UninformedMCTS
    from games.tictactoe import TicTacToe
    from games.guessit import OnePlayerGuessIt, TwoPlayerGuessIt
    from games.leapfrog import ThreePlayerLeapFrog, ThreePlayerLinearLeapFrog

    t = TicTacToe()
    gi1 = OnePlayerGuessIt()
    gi2 = TwoPlayerGuessIt()
    llf3 = ThreePlayerLinearLeapFrog()
    lf3 = ThreePlayerLeapFrog()
    
    #######################
    # Human v Human games #
    #######################

    #h1, h2 = Human(t), Human(t)
    #r = play_match(t, [h1, h2], verbose=True)

    #h1 = Human(gi1)
    #r = play_match(gi1, [h1], verbose=True)

    #h1, h2 = Human(gi2), Human(gi2)
    #r = play_match(gi2, [h1, h2], verbose=True)

    #h1, h2, h3 = Human(llf3), Human(llf3), Human(llf3)
    #r = play_match(llf3, [h1, h2, h3], verbose=True)

    #h1, h2, h3 = Human(lf3), Human(lf3), Human(lf3)
    #r = play_match(lf3, [h1, h2, h3], verbose=True)

    #######################
    # Human v UMCTS games #
    #######################

    h1, u1 = UninformedMCTS(t, 100), UninformedMCTS(t, 10000)
    r = play_match(t, [u1, h1], verbose=True)

    #h1, u1, u2 = Human(lf3), UninformedMCTS(llf3, 10), UninformedMCTS(lf3, 10)
    #r = play_match(lf3, [h1, u1, u2], verbose=True)

    #######################
    # UMCTS v UMCTS games #
    #######################

    '''
    import numpy as np
    import matplotlib.pyplot as plt
    log = []
    for i in range(10, 1000, 100):
        for j in range(10, 1000, 100):
            t = TicTacToe()
            u1, u2 = UninformedMCTS(t, i), UninformedMCTS(t, j)
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



    