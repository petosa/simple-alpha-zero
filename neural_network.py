class NeuralNetwork():

    def train(self, data): # data is a NX3 ndarray with columns [s, p, v]
        raise NotImplementedError

    def predict(self, s): # Returns P(s), V(s) 
        raise NotImplementedError