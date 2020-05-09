from data.alphabets import Alphabet

import utils.rnns as BRNN
import utils.seq as BS

import torch
import numpy as _np

import torch.nn.utils.rnn as _rnn_utils

class KeyValueAttention(torch.nn.Module):
    def __init__(self, ff_q, ff_k=None,scale=True):
        super().__init__()
        self.ff_q = ff_q
        self.ff_k = ff_k if ff_k is not None else torch.nn.Identity()
        self.softmax = torch.nn.Softmax(-1)
        self.scale = scale
    
    def forward(self, q, k, v):
        """
        queries, keys, and values are stored in rows.
        """
        x = self.ff_q(q)
        kk = self.ff_k(k)
        scale_factor = 1.0
        if self.scale:
            scale_factor = _np.power(x.shape[-1],-0.5)
        return self.softmax((x@kk.T)*scale_factor)@v

class MultiHeadAttention(torch.nn.Module):
    pass

__all__ = ["KeyValueAttention"]
