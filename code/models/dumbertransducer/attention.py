from data.alphabets import Alphabet

import utils.rnns as BRNN
import utils.seq as BS

import torch
import numpy as _np

import torch.nn.utils.rnn as _rnn_utils

class KeyValueAttention(torch.nn.Module):
    def __init__(self, ff_layer,scale=True):
        super().__init__()
        self.ff = ff_layer
        self.softmax = torch.nn.Softmax(-1)
        self.scale = scale
    
    def forward(self, q, k, v):
        """
        queries, keys, and values are stored in rows.
        """
        x = self.ff(q)
        scale_factor = 1.0
        if self.scale:
            scale_factor = _np.power(x.shape[-1],-0.5)
        return self.softmax((x@k.T)*scale_factor)@v


__all__ = ["KeyValueAttention"]
