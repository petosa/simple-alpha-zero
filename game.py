class Game:

    def get_initial_state(self):
        raise NotImplementedError

    def get_available_actions(self, s):
        raise NotImplementedError

    def check_winner(self, s):
        raise NotImplementedError()
        
    # Taking an action. Make sure this does NOT modify the given s. Return a new array instead.
    def take_action(self, s, a):
        raise NotImplementedError()

    def get_player(self, s):
        raise NotImplementedError()
