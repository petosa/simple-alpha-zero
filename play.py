from itertools import permutations

# Runs a match with the given game and list of players.
# Returns a dictionary of points. It will map each player object to its score.
# For each match, a player gains a point if it wins, loses a point if it loses,
# and gains no points if it ties.
def play_match(game, players, verbose=False, permute=False):

    # You can use permutations to break the dependence on player order in measuring strength.
    matches = list(permutations(players)) if permute else [players]
    
    # Initialize scoreboard
    scores = {}
    for p in players:
        scores[p] = 0
        p.reset() # Reset incoming players as a precaution.

    # Run the matches (there will be multiple if permute=True)
    for m in matches:
        s = game.get_initial_state()
        if verbose: game.friendly_print(s)
        winner = game.check_winner(s)
        while winner is None:
            p_num = game.get_player(s)
            p = m[p_num]
            if verbose: print("Player {}'s turn.".format(p_num))
            s = p.update_state(s)
            if verbose: game.friendly_print(s)
            winner = game.check_winner(s)
        for i, p in enumerate(m):
            if winner == -1:
                scores[p] += .5/len(matches)
            elif winner == i:
                scores[p] += 1/len(matches)
            p.reset() # Clear our tree to make the next match fair

    # Assign an outcome to each player (Win, Lose, or Tie)
    outcomes = {}
    num_players = game.get_num_players()
    for p in scores:
        if scores[p] < 1/num_players:
            outcomes[p] = "Lose"
        elif scores[p] == 1/num_players:
            outcomes[p] = "Tie"
        else:
            outcomes[p] = "Win"

    return scores, outcomes



if __name__ == "__main__":
    from players.human_player import HumanPlayer
    from neural_network import NeuralNetwork
    from models.minivgg import MiniVGG
    from models.smallvgg import SmallVGG
    from models.senet import SENet
    from players.uninformed_mcts_player import UninformedMCTSPlayer
    from players.deep_mcts_player import DeepMCTSPlayer
    from games.connect4 import Connect4
    from games.tictactoe import TicTacToe
    from games.leapfrog import ThreePlayerLeapFrog


    # Change these variable 
    game = Connect4()
    ckpt = 775
    nn = NeuralNetwork(game, SENet, cuda=True)
    nn.load(ckpt)
    
    # HumanPlayer(game),
    # UninformedMCTSPlayer(game, simulations=1000)
    # DeepMCTSPlayer(game, nn, simulations=50)
    
    players = [HumanPlayer(game), DeepMCTSPlayer(game, nn, simulations=50)]
    for _ in range(5):
        print(play_match(game, players, verbose=True, permute=True))
    