import torch

# Interface for defining a pytorch model.
class Model(torch.nn.Module):

  def __init__(self, input_shape, output_shape):
    super(Model, self).__init__()
    self.input_shape = input_shape
    self.output_shape = output_shape

  # Simply define the forward pass.
  # Your output must have both a policy and value head, so you must return 2 tensors p, v in that order.
  def forward(self, x):
    raise NotImplementedError
