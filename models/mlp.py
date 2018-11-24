import sys
import torch
import torch.nn.functional as F
import numpy as np
sys.path.append("..")
from model import Model


class MLP(Model):

  def __init__(self, input_shape, output_shape, **kwargs):
    super(MLP, self).__init__(input_shape, output_shape, *kwargs)
    num_hidden_units = 200
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.linear1 = torch.nn.Linear(np.prod(input_shape), num_hidden_units)
    self.p_head = torch.nn.Linear(num_hidden_units, np.prod(output_shape))
    self.v_head = torch.nn.Linear(num_hidden_units, 1)

  def forward(self, x):
    batch_size = len(x)
    this_output_shape = tuple([batch_size] + list(self.output_shape))

    # Network
    x = x.view(batch_size, -1) # Flatten high-dimensional samples
    h_relu = F.relu(self.linear1(x)) # Input -> hidden

    # Outputs
    p_logits = self.p_head(h_relu).view(this_output_shape)
    v = torch.tanh(self.v_head(h_relu))

    return p_logits, v
