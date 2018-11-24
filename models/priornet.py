import sys
import torch
import numpy as np
sys.path.append("..")
from model import Model



class PriorNet(Model):


  def forward(self, x):
    batch_size = len(x)
    this_output_shape = tuple([batch_size] + list(self.output_shape))

    p_logits = torch.zeros(this_output_shape)
    p_logits[:,0,1] = 10
    v = torch.zeros((batch_size, 1))

    return p_logits, v
