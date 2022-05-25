# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Bottleneck ResNet v2 with GroupNorm and Weight Standardization."""

import math
from collections import OrderedDict  # pylint: disable=g-importing-member

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class StdConv2d(nn.Conv2d):

  def forward(self, x):
    w = self.weight
    v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    w = (w - m) / torch.sqrt(v + 1e-10)
    return F.conv2d(x, w, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
  return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                   padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
  return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                   padding=0, bias=bias)


def tf2th(conv_weights):
  """Possibly convert HWIO to OIHW."""
  if conv_weights.ndim == 4:
    conv_weights = conv_weights.transpose([3, 2, 0, 1])
  return torch.from_numpy(conv_weights)


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


class PreActBottleneck(nn.Module):
  """Pre-activation (v2) bottleneck block.

  Follows the implementation of "Identity Mappings in Deep Residual Networks":
  https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

  Except it puts the stride on 3x3 conv when available.
  """

  def __init__(self, cin, cout=None, cmid=None, stride=1, use_adapter=False):
    super().__init__()
    cout = cout or cin
    cmid = cmid or cout//4
    self.use_adapter = use_adapter

    self.gn1 = nn.GroupNorm(32, cin)
    self.conv1 = conv1x1(cin, cmid)
    self.gn2 = nn.GroupNorm(32, cmid)
    if(use_adapter): self.adapter2 = CaSE(cmid) #TODO change if needed
    self.conv2 = conv3x3(cmid, cmid, stride)
    self.gn3 = nn.GroupNorm(32, cmid)
    if(use_adapter): self.adapter3 = CaSE(cmid) #TODO change if needed
    self.conv3 = conv1x1(cmid, cout)
    self.relu = nn.ReLU(inplace=True)

    if (stride != 1 or cin != cout):
      # Projection also with pre-activation according to paper.
      self.downsample = conv1x1(cin, cout, stride)

  def forward(self, x):
    out = self.relu(self.gn1(x))

    # Residual branch
    residual = x
    if hasattr(self, 'downsample'):
      residual = self.downsample(out)

    # Unit's branch
    out = self.conv1(out)
    
    if(self.use_adapter):
        out = self.conv2(self.adapter2(self.relu(self.gn2(out))))
    else:
        out = self.conv2(self.relu(self.gn2(out)))

    if(self.use_adapter):
        out = self.conv3(self.adapter3(self.relu(self.gn3(out))))
    else:
        out = self.conv3(self.relu(self.gn3(out)))

    return out + residual

  def load_from(self, weights, prefix=''):
    convname = 'standardized_conv2d'
    with torch.no_grad():
      self.conv1.weight.copy_(tf2th(weights[f'{prefix}a/{convname}/kernel']))
      self.conv2.weight.copy_(tf2th(weights[f'{prefix}b/{convname}/kernel']))
      self.conv3.weight.copy_(tf2th(weights[f'{prefix}c/{convname}/kernel']))
      self.gn1.weight.copy_(tf2th(weights[f'{prefix}a/group_norm/gamma']))
      self.gn2.weight.copy_(tf2th(weights[f'{prefix}b/group_norm/gamma']))
      self.gn3.weight.copy_(tf2th(weights[f'{prefix}c/group_norm/gamma']))
      self.gn1.bias.copy_(tf2th(weights[f'{prefix}a/group_norm/beta']))
      self.gn2.bias.copy_(tf2th(weights[f'{prefix}b/group_norm/beta']))
      self.gn3.bias.copy_(tf2th(weights[f'{prefix}c/group_norm/beta']))
      if hasattr(self, 'downsample'):
        w = weights[f'{prefix}a/proj/{convname}/kernel']
        self.downsample.weight.copy_(tf2th(w))


class ResNetV2(nn.Module):
  """Implementation of Pre-activation (v2) ResNet mode."""

  def __init__(self, block_units, width_factor, use_adapter=False):
    super().__init__()
    wf = width_factor  # shortcut 'cause we'll use it a lot.

    # The following will be unreadable if we split lines.
    # pylint: disable=line-too-long
    self.root = nn.Sequential(OrderedDict([
        ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
        ('pad', nn.ConstantPad2d(1, 0)),
        ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
        # The following is subtly not the same!
        # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]))

    self.body = nn.Sequential(OrderedDict([
        ('block1', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf, use_adapter=use_adapter))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf, use_adapter=use_adapter)) for i in range(2, block_units[0] + 1)],
        ))),
        ('block2', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2, use_adapter=use_adapter))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf, use_adapter=use_adapter)) for i in range(2, block_units[1] + 1)],
        ))),
        ('block3', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2, use_adapter=use_adapter))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf, use_adapter=use_adapter)) for i in range(2, block_units[2] + 1)],
        ))),
        ('block4', nn.Sequential(OrderedDict(
            [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2, use_adapter=use_adapter))] +
            [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf, use_adapter=use_adapter)) for i in range(2, block_units[3] + 1)],
        ))),
    ]))
    # pylint: enable=line-too-long

    #self.head_size = head_size
    self.embedding_size = 2048*wf # wf=width_factor

    #self.head = nn.Sequential(OrderedDict([
    #    ('gn', nn.GroupNorm(32, 2048*wf)),
    #    ('relu', nn.ReLU(inplace=True)),
    #    ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
    #    #('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True)), # removing this, as we will use a matrix instead
    #]))
    
    # Adding a dict to better manage the head
    #self.zero_head = zero_head
    head_dict = OrderedDict()
    head_dict['gn'] = nn.GroupNorm(32, 2048*wf)
    head_dict['relu'] = nn.ReLU(inplace=True)
    if use_adapter: head_dict['adapter'] = CaSE(2048*wf)
    head_dict['avg'] =nn.AdaptiveAvgPool2d(output_size=1)
    self.head = nn.Sequential(head_dict)

  def set_mode(self, adapter: str, backbone: str, verbose: bool = False):
        assert adapter in ["eval", "train"]
        assert backbone in ["eval", "train"]
        for name, module in self.named_modules():
            if(type(module) is CaSE):
                if(adapter=="eval"): module.eval()
                elif(adapter=="train"): module.train()
                if(verbose): print(f"adapter-layer ...... name: {name}; train: {module.training}")
            else:
                if(backbone=="eval"): module.eval()
                elif(backbone=="train"): module.train()
                if(verbose): print(f"Backbone-layer .. name: {name}; train: {module.training}")

  def count_parameters(self):
      params_backbone = 0
      params_adapters = 0
      for name, parameter in self.named_parameters():
          if("set_encoder" in name):
              params_adapters += parameter.numel()
          elif("gamma_generator" in name):
              params_aadapters += parameter.numel()
          else:
              params_backbone += parameter.numel()
      # Done, printing
      info_str = f"params-backbone .... {params_backbone} ({(params_backbone/1e6):.2f} M)\n" \
                   f"params-adapters .... {params_adapters} ({(params_adapters/1e6):.2f} M)\n" \
                   f"params-total ....... {params_backbone+params_adapters} ({((params_backbone+params_adapters)/1e6):.2f} M)\n"
      print(info_str)
        
  def reset(self):
      for name, module in self.named_modules():
          if(type(module) is CaSE):
              module.reset_parameters()

  def forward(self, x):
    x = self.head(self.body(self.root(x)))
    assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
    return x[...,0,0]
        
  def load_from(self, weights, prefix='resnet/'):
    with torch.no_grad():
      self.root.conv.weight.copy_(tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel']))  # pylint: disable=line-too-long
      self.head.gn.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
      self.head.gn.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
      #if self.zero_head:
      #  #nn.init.zeros_(self.head.conv.weight)
      #  #nn.init.zeros_(self.head.conv.bias)
      #  #nn.init.zeros_(self.out.weight)
      #  #nn.init.zeros_(self.out.bias)
      #  pass
      #else:
      #  self.head.conv.weight.copy_(tf2th(weights[f'{prefix}head/conv2d/kernel']))  # pylint: disable=line-too-long
      #  self.head.conv.bias.copy_(tf2th(weights[f'{prefix}head/conv2d/bias']))

      for bname, block in self.body.named_children():
        for uname, unit in block.named_children():
          unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')


KNOWN_MODELS = OrderedDict([
    ('BiT-M-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-M-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-M-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-M-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-M-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-M-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
    ('BiT-S-R50x1', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 1, *a, **kw)),
    ('BiT-S-R50x3', lambda *a, **kw: ResNetV2([3, 4, 6, 3], 3, *a, **kw)),
    ('BiT-S-R101x1', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 1, *a, **kw)),
    ('BiT-S-R101x3', lambda *a, **kw: ResNetV2([3, 4, 23, 3], 3, *a, **kw)),
    ('BiT-S-R152x2', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 2, *a, **kw)),
    ('BiT-S-R152x4', lambda *a, **kw: ResNetV2([3, 8, 36, 3], 4, *a, **kw)),
])
