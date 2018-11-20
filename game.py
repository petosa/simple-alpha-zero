class Game:

    def get_initial_state(self):
        raise NotImplementedError

    def get_available_actions(self, s):
        raise NotImplementedError

    # Return None if there is no winner yet.
    # Return -1 if there is a tie.
    # Otherwise return the player number that won.
    def check_winner(self, s):
        raise NotImplementedError()
        
    # Taking an action. Make sure this does NOT modify the given s. Return a new array instead.
    def take_action(self, s, a):
        raise NotImplementedError()

    # Return the player number whose turn it is.
    def get_player(self, s):
        raise NotImplementedError()

    # Visualizes the given state.
    def friendly_print(self, s):
        raise NotImplementedError()
