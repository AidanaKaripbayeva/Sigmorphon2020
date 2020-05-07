from data.alphabets import Alphabet

import utils.rnns as BRNN
import utils.seq as BS

import torch

import torch.nn.utils.rnn as _rnn_utils

class KeyValueAttention(torch.nn.Module):
    def __init__(self, ff_layer):
        super().__init__()
        self.ff = ff_layer
        self.softmax = torch.nn.Softmax(-1)
    
    def forward(self, q, k, v):
        """
        queries, keys, and values are stored in rows.
        """
        x = self.ff(q)
        return self.softmax(x@k.T)@v


__all__ = ["KeyValueAttention"]
