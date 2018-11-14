class Game:

    def get_available_actions(self, s):
        raise NotImplementedError

    def check_winner(self, s):
        raise NotImplementedError()
        
    # Taking an action 
    def take_action(self, s, a):
        raise NotImplementedError()
