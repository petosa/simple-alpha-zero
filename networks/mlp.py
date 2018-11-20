import sys
import torch
import numpy as np
sys.path.append("..")
from neural_network import NeuralNetwork

class MLP(NeuralNetwork):

    def __init__(self, game):
        self.game = game
        input_shape = game.get_initial_state().shape
        p_shape = game.get_available_actions(game.get_initial_state()).shape
        self.model = Network(input_shape, p_shape)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def loss(self, states, p_pred, v_pred, p_gt, v_gt):
        v_loss = ((v_pred - v_gt)**2).sum()/v_gt.size()[0]
        p_loss = 0
        for i, gt in enumerate(p_gt):
            gt = torch.from_numpy(gt.astype(np.float32))
            s = states[i]
            pred = p_pred[i]
            mask = torch.from_numpy(self.game.get_available_actions(s).astype(np.uint8))
            pred = torch.masked_select(pred, mask)
            p_loss += (-gt * torch.log(pred)).sum()
        p_loss /= len(p_gt)
        return p_loss + v_loss

    def train(self, data):
        self.model.train()
        states = np.stack(data[:,0])
        x = torch.from_numpy(states)
        p_pred, v_pred = self.model(x)
        p_gt, v_gt = data[:,1], torch.from_numpy(data[:,2].astype(np.float32))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=1e-4)
        optimizer.zero_grad()
        loss = self.loss(states, p_pred, v_pred.view(-1), p_gt, v_gt)
        loss.backward()
        optimizer.step()
        print(loss)
    
    def predict(self, s):
        s = np.array([s])
        with torch.no_grad():
            s = torch.from_numpy(s)
            self.model.eval()
            p, v = self.model(s)
            p, v = p.numpy().squeeze(), v.numpy()
        return p, v


class Network(torch.nn.Module):
  def __init__(self, input_shape, output_shape):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(Network, self).__init__()
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.linear1 = torch.nn.Linear(np.prod(input_shape), 10)
    self.p_head = torch.nn.Linear(10, np.prod(output_shape))
    self.v_head = torch.nn.Linear(10, 1)

  def forward(self, x):
    """
    In the forward function we accept a Tensor of input data and we must return
    a Tensor of output data. We can use Modules defined in the constructor as
    well as arbitrary (differentiable) operations on Tensors.
    """
    n = len(x)
    x = x.view(n, -1)
    this_output_shape = tuple([n] + list(self.output_shape))
    h_relu = self.linear1(x).clamp(min=0)
    p = torch.nn.functional.softmax(self.p_head(h_relu))
    p = p.view(this_output_shape)
    v = torch.tanh(self.v_head(h_relu))
    return p, v
