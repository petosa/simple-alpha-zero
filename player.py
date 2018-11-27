class Player:

    # Player is given a state s and returns a mutated state.
    def update_state(self, s):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
