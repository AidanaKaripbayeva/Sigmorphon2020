from data.alphabets import Alphabet

import utils.rnns as BRNN
import utils.seq as BS

import torch

import torch.nn.utils.rnn as _rnn_utils

from .attention import *


class EncodingWindows(torch.nn.Module):
    def __init__(self, before, after):
        super().__init__()
        self.before = before
        self.after = after
    
    def forward(self, padded_data):
        #TODO: This is probably quite slow
        
        is_packed = isinstance(padded_data,_rnn_utils.PackedSequence)
        if is_packed:
            padded_data, lens = _rnn_utils.pad_packed_sequence(padded_data,padding_value=Alphabet.stop_integer)
        
        y = torch.cat([padded_data[0:1] for i in range(self.before)] + [padded_data] + [padded_data[-1:] for i in range(self.after)])
        y = y.unfold(0,self.before+1+self.after,1)
        if padded_data.ndim == 1:
            pass
        elif padded_data.ndim == 2:
            y = y.reshape(y.shape[0],-1)
        elif padded_data.ndim >= 3:
            newshape = [int(i) for i in y.shape[:-2]] + [-1]
            y = y.reshape(*newshape)
        
        if is_packed:
            return _rnn_utils.pack_padded_sequence(y, lens, batch_first=False, enforce_sorted=False)
        return y
        #return y.unfold(0,self.before+1+self.after,1)

class OffsetWindows(torch.nn.Module):
    def __init__(self, offset,inclusive=False):
        super().__init__()
        self.offset = offset
        self.inclusive=inclusive
    
    def forward(self, padded_data):
        #TODO: This is probably quite slow
        
        is_packed = isinstance(padded_data,_rnn_utils.PackedSequence)
        if is_packed:
            padded_data, lens = _rnn_utils.pad_packed_sequence(padded_data,padding_value=Alphabet.stop_integer)
        
        cat_list = list()
        
        stacking_dim = 1
        if padded_data.ndim >= 3:
            stacking_dim = 2
            
        if self.offset < 0:
            y = torch.cat([padded_data[0:1]]*(-1*self.offset) + [padded_data[:self.offset]], dim=0)
            if self.inclusive and self.offset < -1:
                stack_these = [y] + [torch.cat([padded_data[0:1]]*(-1*o) + [padded_data[:o]], dim=0) for o in range(self.offset+1,0)]
                y = torch.cat(stack_these, dim=stacking_dim)
        elif self.offset > 0:
            y = torch.cat([padded_data[self.offset:]] + [padded_data[-1:]]*self.offset, dim=0)
            if self.inclusive and self.offset > 1:
                stack_these = [torch.cat([padded_data[o:]] + [padded_data[-1:]]*o, dim=0) for o in range(1,self.offset)] + [y]
                y = torch.cat(stack_these, dim=stacking_dim)
        else:
            y = padded_data
        
        
        if is_packed:
            return _rnn_utils.pack_padded_sequence(y, lens, batch_first=False, enforce_sorted=False)
        return y
        #return y.unfold(0,self.before+1+self.after,1)


__all__ = ["EncodingWindows","OffsetWindows"]
