import torch as _torch
import torch.nn as _nn
import torch.nn.utils.rnn as _rnn_utils
import numpy as _np

class RNN_Absolute_Position(_nn.Module):
    def __init__(self, num_freqs=0, dtype=_torch.float16):
        super().__init__()
        self.num_freqs = num_freqs
        self.dtype = dtype
    
    def create_matrix(self, L,out=None):
        if out is None:
            out = _torch.zeros(L,self.num_freqs)
        
        _torch.linspace(0.01,  L*0.01 + 0.01,steps=L,out=out[:L,0])
        out[L:,0] = L
        for i in range(1,self.num_freqs):
            _torch.sin(_np.pi * out[:,0] * _np.power(i,1.0),out=out[:,i])
        
        #return out
        return out.detach()
    
    def forward(self, in_data):
        raise NotImplementedError("sorry")

class RNN_Relative_Position(_nn.Module):
    def __init__(self, num_freqs=0, dtype=_torch.float16):
        super().__init__()
        self.num_freqs = num_freqs
        self.dtype = dtype
        self.whitening = 0.00001
    
    def create_matrix(self, L, t,out=None):
        if out is None:
            out = _torch.zeros(L,self.num_freqs)
        
        _torch.linspace(self.whitening,  1.0-self.whitening,steps=t,out=out[0:t,0])
        out[t:,0] = 1.0
        for i in range(1,self.num_freqs):
            #The first chebyshev polynomial is just y = x, which we already have.
            _torch.cos((i+1)*_torch.acos(out[:,0]), out=out[:,i])
        
        #return out
        return out.detach()
    
    def forward(self, in_data):
        raise NotImplementedError("sorry")


#DEPRICATED
class RNN_Position_Encoding(_nn.Module):
    def __init__(self,fourier_freqs=0,chebyshev_freqs=0):
        super(RNN_Position_Encoding,self).__init__()
        self.fourier_freqs = fourier_freqs
        self.chebyshev_freqs = chebyshev_freqs
    
    def forward(self,in_data,lengths=None):
        is_packed = isinstance(in_data,_rnn_utils.PackedSequence)
        assert is_packed or lengths is not None, "Either provide a PackedSequence, or the lengths are needed"
        
        x = None
        if is_packed:
            x, lengths = _rnn_utils.pad_packed_sequence(in_data)
        
        d_total_pos_dims = 2 + self.fourier_freqs + self.chebyshev_freqs
        
        my_output = _torch.zeros(x.shape[0],x.shape[1],d_total_pos_dims)
        #relative and absolute coordinates
        for i,l in enumerate(lengths):
            _torch.linspace(0.0,1.0,steps=l,out=my_output[0:l,i,0])
            _torch.linspace(0.0,  l,steps=l,out=my_output[0:l,i,1])
        
        #fourier basis will be the same for all. It is absolute.
        #TODO: Rewrite this without the temporary storage for the fourier_basis
        fourier_basis = _torch.zeros(x.shape[0],self.fourier_freqs)
        full_linspace = _torch.linspace(0.0,x.shape[0],steps=x.shape[0])#as long as the longest sequence
        for i in range(fourier_basis.shape[1]):
            fourier_basis[:,i] = _torch.cos(_np.pi * full_linspace * _np.power(i+1,-1.0) )
        my_output[:,:,2:2+self.fourier_freqs] = fourier_basis[:,None,:]
        
        for i in range(self.chebyshev_freqs):#Could probably also eliminate this loop
            my_output[:,:,2+self.fourier_freqs+i] = _torch.cos((i+1)*_torch.acos(my_output[:,:,0]))
        
        #This encoding is constant, it doesn't need a backpropagation graph to slow things.
        #TODO: Find out how to tell it it also doesn't need a gradient.
        my_output = my_output.detach()
        
        if is_packed:
            return _rnn_utils.pack_padded_sequence(my_output,lengths,enforce_sorted=False)
        return my_output
