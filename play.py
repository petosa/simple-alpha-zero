from itertools import permutations

# Runs a match with the given game and list of players.
# Returns a dictionary of points. It will map each player object to its score.
# For each match, a player gains a point if it wins, loses a point if it loses,
# and gains no points if it ties.
def play_match(game, players, verbose=False, permute=False):

    matches = list(permutations(players)) if permute else [players]
    scores = {}
    for p in players:
        scores[p] = 0

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
        if winner != -1:
            for i, p in enumerate(m):
                scores[p] += 1 if i == winner else -1

    return scores


if __name__ == "__main__":
    from players.human_player import HumanPlayer
    from neural_network import NeuralNetwork
    from models.minivgg import MiniVGG
    from models.smallvgg import SmallVGG
    from players.uninformed_mcts_player import UninformedMCTSPlayer
    from players.deep_mcts_player import DeepMCTSPlayer
    from games.connect4 import Connect4
    from games.tictactoe import TicTacToe

    game = Connect4()
    ckpt = 1800
    choices = [
        HumanPlayer(game),
        UninformedMCTSPlayer(game, simulations=1000),
        DeepMCTSPlayer(game, NeuralNetwork(game, SmallVGG), simulations=800)
    ]
    choices[-1].tree.nn.load(ckpt)
    players = [choices[2], choices[0]]
    play_match(game, players, verbose=True)
    