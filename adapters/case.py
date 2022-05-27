import torch
from torch import nn
from collections import OrderedDict

class CaSE(nn.Module):
  def __init__(self, cin, reduction=64, min_units=16, standardize=True, out_mul=2.0, device=None, dtype=None):
      """
      Initialize a CaSE adaptive block.
  
      Parameters:
      cin (int): number of input channels.
      reduction (int): divider for computing number of hidden units.
      min_units (int): clip hidden units to this value (if lower).
      standardize (bool): standardize the input for the MLP.
      out_mul (float): multiply the MLP output by this value.
      """
      factory_kwargs = {'device': device, 'dtype': dtype}
      super(CaSE, self).__init__()
      self.cin = cin
      self.standardize = standardize
      self.out_mul = out_mul

      # Gamma-generator
      hidden_features = max(min_units, cin // reduction)
      self.gamma_generator = nn.Sequential(OrderedDict([
          ('gamma_lin1', nn.Linear(cin, hidden_features, bias=True, **factory_kwargs)),
          ('gamma_silu1', nn.SiLU()),
          ('gamma_lin2', nn.Linear(hidden_features, hidden_features, bias=True, **factory_kwargs)),
          ('gamma_silu2', nn.SiLU()),
          ('gamma_lin3', nn.Linear(hidden_features, cin, bias=True, **factory_kwargs)),
          ('gamma_sigmoid', nn.Sigmoid()),
        ]))

      self.gamma = torch.tensor([1.0]) # Set to one for the moment
      self.reset_parameters()

  def reset_parameters(self):      
      torch.nn.init.zeros_(self.gamma_generator.gamma_lin3.weight)
      torch.nn.init.zeros_(self.gamma_generator.gamma_lin3.bias)

  def forward(self, x):
      # Adaptive mode
      if(self.training):
          self.gamma = torch.mean(x, dim=[0,2,3]) # spatial + context pooling
          if(self.standardize):
                  self.gamma = (self.gamma - torch.mean(self.gamma)) / torch.sqrt(torch.var(self.gamma, unbiased=False)+1e-5)
          self.gamma = self.gamma.unsqueeze(0) #-> [1,channels]
          self.gamma = self.gamma_generator(self.gamma) * self.out_mul
          self.gamma = self.gamma.reshape([1,-1,1,1])
          return self.gamma * x # Apply gamma to the input and return
      # Inference Mode
      else:
          self.gamma = self.gamma.to(x.device)
          return self.gamma * x # Use previous gamma

  def extra_repr(self) -> str:
        return 'cin={}'.format(self.cin)

