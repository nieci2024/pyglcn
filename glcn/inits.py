import torch
import torch.nn as nn
import numpy as np

def glorot_init(tensor):
    if tensor.is_sparse:
        # For sparse tensors, initialize non-zero values
        values = tensor.values()
        nn.init.xavier_uniform_(values.unsqueeze(0) if values.dim() == 1 else values) # xavier_uniform_ expects at least 2D
    else:
      if tensor.dim() < 2:
          stdv = 1. / np.sqrt(tensor.size(0))
          tensor.data.uniform_(-stdv, stdv) # Fallback for 1D, or use kaiming for relu
      else:
        nn.init.xavier_uniform_(tensor)

def zeros_init(tensor):
    if tensor.is_sparse:
        # Cannot directly fill sparse tensor values with zeros in place like this easily,
        # typically sparse tensors are constructed with non-zero values.
        # If a parameter is meant to be zero and sparse, it's an unusual case.
        # For dense, it's:
        nn.init.zeros_(tensor.values()) # if it makes sense to zero out existing values
    else:
        nn.init.zeros_(tensor)

def ones_init(tensor):
    if tensor.is_sparse:
         nn.init.ones_(tensor.values()) # if it makes sense to make existing values one
    else:
        nn.init.ones_(tensor)