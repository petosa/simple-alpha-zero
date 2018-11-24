import torch


class Model(torch.nn.Module):

  def __init__(self, input_shape, output_shape):
    super(Model, self).__init__()
    self.input_shape = input_shape
    self.output_shape = output_shape

  def forward(self, x):
    raise NotImplementedError
