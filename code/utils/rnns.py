import torch as _torch
import torch.nn as _nn
import torch.nn.utils.rnn as _rnn_utils


class RNN_Input_Embedding(_torch.nn.Module):
    """Class for wrapping a py_torch nn.Embedding for use with RNNs further down the pipeline.
    The problem with nn.Embedding is that it addes an extra dimension and often doesn't do it nicely.
    This will ensure that strange input and output data shapes will be fixed in preparation for that.
    """
    def __init__(self,an_embedding):
        super(RNN_Input_Embedding,self).__init__()
        self.ff = an_embedding

    def _forward_packed(self, some_data):
        lengths = None
        x = None
        y = None


        x = some_data.data
        lengths = some_data.batch_sizes

        if x.ndim == 2 and x.shape[-1] == 1:
            x = _torch.flatten(x)

        assert x.ndim == 1, "Packed data should only have 1 dimension before embedding"

        y = self.ff(x)

        return _rnn_utils.PackedSequence(y,batch_sizes=some_data.batch_sizes, sorted_indices=some_data.sorted_indices, unsorted_indices=some_data.unsorted_indices)

    def _forward_padded(self, some_data):
        lengths = None
        x = None
        y = None

        x = some_data

        assert x.ndim >= 2

        if x.ndim == 2:
            pass
            #good
        elif x.ndim == 3 and x.shape[-1] == 1:
            x = _torch.flatten(x,1)
        elif x.ndim >= 4 and x.shape[-1] == 1 and x.shape[-2] == 1:
            x = _torch.flatten(x,2,-1)
        else:
            raise Exception("padded input of this shape not supported yet, : {}".format(x.shape))

        assert x.ndim == 2

        y = self.ff(x)

        assert y.ndim == 3

        return y

    def forward(self, some_data, h_0=None):
        """Runs the contained embedding and spits out either a PackedSequence or padded sequences, depending on the input.
        """
        is_packed = isinstance(some_data,_rnn_utils.PackedSequence)
        if is_packed:
            return self._forward_packed(some_data), _torch.Tensor() #no hidden state
        else:
            return self._forward_padded(some_data), _torch.Tensor() #no hidden state

class RNN_FFLayer(_torch.nn.Module):
    """Class for wrapping a py_torch.nn feed-forward networks for use with RNNs further down the pipeline.

    """
    def __init__(self,ff):
        super(RNN_FFLayer,self).__init__()
        self.ff = ff

    def forward(self,x,h_0=None):
        is_packed = isinstance(x,_rnn_utils.PackedSequence)
        if is_packed:
            y = self.ff(x.data)
            y = _rnn_utils.PackedSequence(y,batch_sizes=x.batch_sizes, sorted_indices=x.sorted_indices, unsorted_indices=x.unsorted_indices)
        else:
            y = self.ff(x)
        return y, _torch.Tensor() #no hidden state.

class RNN_Layers(_torch.nn.Module):
    """Class for stacking multiple feed-forward layers like nn.Sequential.
    Currently does not automatically detect feed-forward layers, so you'll have to wrap those in RNN_FFLayer
    """

    def __init__(self,*args):
        super(RNN_Layers,self).__init__()
        self.layers = _torch.nn.ModuleList(args)

    def forward(self,x,h_0=None):
        """
        I guess I'd like this to be variadic?
        """
        if h_0 == None:
            h_0 = [None for i in range(len(self.layers))]

        output,hidden0 = self.layers[0](x,h_0[0])
        hidden = [hidden0]
        for l,one_h_0 in zip(self.layers[1:],h_0):
            output,h = l(output,one_h_0)
            hidden.append(h)
        return output, tuple(hidden)
