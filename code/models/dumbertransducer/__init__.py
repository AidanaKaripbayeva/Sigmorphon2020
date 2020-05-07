from data.alphabets import Alphabet

import utils.rnns as BRNN
import utils.seq as BS

import torch

import torch.nn.utils.rnn as _rnn_utils

from .attention import *
from .windows import *


class DumberEncoder(torch.nn.Module):
    def __init__(self, alphabet_size, embedding_dim=40, hidden_dim=10, num_layers=3, num_languages=None):
        super().__init__()
        #PARAMETERS
        # The number of different letters in the alphabet.
        self.alphabet_size = alphabet_size
        # The dimension of the output of the embedding layer.
        self.embedding_dim = embedding_dim
        # Then number of different languages.
        self.num_languages = num_languages
        
        self.hidden_dim = [ hidden_dim ]*num_layers if isinstance(hidden_dim, int) else hidden_dim
        assert isinstance(self.hidden_dim, list) or isinstance(self.hidden_dim, tuple), str(type(self.hidden_dim))
        assert len(self.hidden_dim) == num_layers, "Either provide one integer for hidden_dim, or one for each layer"
        
        #LAYERS
        self.position = BS.RNN_Position_Encoding(20,20)
        # Create the embedding layer for the encoder.
        self.embedding = BRNN.RNN_Input_Embedding(torch.nn.Embedding(alphabet_size, embedding_dim))
        
        self.windows = EncodingWindows(1,1)
        
        
        # Forward LSTM for encoding
        #self.f_lstm = torch.nn.LSTM(embedding_dim*3, hidden_dim)
        
        #Layers, totally parallel, sort of.
        tmp_layers = [torch.nn.Linear(embedding_dim*3, self.hidden_dim[0]), torch.nn.Sigmoid()]
        for i in range(1,num_layers):
            tmp_layers.append(torch.nn.Linear(self.hidden_dim[i-1], self.hidden_dim[i]))
            tmp_layers.append(torch.nn.Sigmoid())
        self.f_lstm = BRNN.RNN_FFLayer(torch.nn.Sequential(*tmp_layers))
    
    def forward(self, families, languages, tags, lemma):
        pos = self.position(lemma)#I expect it to blow up here.
        emb,_ = self.embedding(lemma)
        
        window_emb = self.windows(emb)
        
        lstm_output, lstm_hidden = self.f_lstm(window_emb)
        
        return pos, emb, lstm_output, lstm_hidden
        

class DumberDecoder(torch.nn.Module):
    def __init__(self, alphabet_size, tag_vector_dim, embedding_dim=40, hidden_dim=10, num_layers=3, output_length=50, num_languages=None):
        super().__init__()
        self.max_output_length = output_length
        self.alphabet_size = alphabet_size
        #self.linear = torch.nn.Linear(tag_vector_dim + alphabet_size + embedding_dim + hidden_dim, alphabet_size) #self.linear = torch.nn.Linear(embedding_dim, alphabet_size)
        self.linear = torch.nn.Linear(embedding_dim, alphabet_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.final_softmax = torch.nn.Softmax(dim=-1)
        
        self.position = BS.RNN_Position_Encoding(20,20)
        
        self.pos_attention = KeyValueAttention(
                torch.nn.Sequential(torch.nn.Linear(42,42), torch.nn.Sigmoid())
            )
    
    def forward(self, family, language, tags, position, embedding, states, hidden):
        
        output_sequence = torch.zeros(self.max_output_length, dtype=torch.int32)
        output_probs = torch.zeros((self.max_output_length, self.alphabet_size))
        
        output_sequence[0] = Alphabet.start_integer
        output_sequence[1:] = Alphabet.stop_integer #pading
        
        output_probs[0,Alphabet.start_integer] = 1.0
        output_probs[1:,Alphabet.stop_integer] = 1.0
        
        #Need to change the extents of this loop.
        for output_i in range(1, min(self.max_output_length, embedding.shape[0]) ):
            
            #one_attended_emb = self.pos_attention(position[output_i], position, embedding)
            #one_attended_state = self.pos_attention(position[output_i], position, states)
            
            #import pdb; pdb.set_trace()
            #x = self.linear(torch.cat([output_probs[output_i-1], embedding[output_i], states[output_i]])) #x = self.linear(embedding[output_i])
            x = self.linear(embedding[output_i])
            x = self.sigmoid(x)
            y = self.final_softmax(x)
            
            output_symbol = y.argmax(-1)
            
            output_sequence[output_i] = output_symbol
            output_probs[output_i,:] = y
            
            if output_symbol == Alphabet.stop_integer:
                break
        
        
        return output_probs[:output_i+1] #no probs past predicted end.
        
        

class DumberTransducer(torch.nn.Module):
    def __init__(self, alphabet_size, num_languages, tag_vector_dim, embedding_dim=40, hidden_dim=10, num_layers=3, output_length=50):
        super().__init__()
        # The number of different letters in the alphabet.
        self.alphabet_size = alphabet_size
        # Then number of different languages.
        self.num_languages = num_languages
        # The dimension of the tag vector.
        self.tag_vector_dim = tag_vector_dim
        # The dimension of the output of the embedding layer.
        self.embedding_dim = embedding_dim
        # The dimension of the hidden layer in the LSTM unit in the encoder.
        self.hidden_dim = hidden_dim
        # The number of layers in the LSTM units in the encoder and in the decoder.
        self.num_layers = num_layers
        # The length of the output string (including the start, end and padding tokens).
        self.output_length = output_length
        
        
        self.encoder = DumberEncoder(alphabet_size, embedding_dim)
        
        self.decoder = DumberDecoder(alphabet_size, tag_vector_dim, embedding_dim)
        
        self.device = None
        
    def to(self, target_device):
        self.device = target_device
        super().to(self.device)
        return self

    def forward(self, families, languages, tags, lemmata, teacher_forcing=None):
        batch_size = len(tags)
        batch_outputs = []  # Will hold the integral representation of the characters to be output for this batch.
        batch_probabilities = []   # Will hold the probability vectors for this batch.
        
        assert isinstance(lemmata, list) or isinstance(lemmata, tuple), str(type(lemmata)) + " Only lists are implemented"
        lengths = [len(i) for i in lemmata]
        
        #DEVICE MOVEMENT #TODO: move out of here.
        if self.device is not None:
            families = [i.to(self.device) for i in families]
            languages = [i.to(self.device) for i in languages]
            tags = [i.to(self.device) for i in tags]
            lemmata = [i.to(self.device) for i in lemmata]
        
        #
        #The encoder uses layers that can do everything at once.
        #ENCODE EVERYTHING
        #
        packed_fam   = _rnn_utils.pack_sequence(families,  enforce_sorted=False)
        packed_langs = _rnn_utils.pack_sequence(languages, enforce_sorted=False)
        packed_lems  = _rnn_utils.pack_sequence(lemmata,   enforce_sorted=False)
        
        packed_pos, packed_emb, packed_x, hidden = self.encoder(packed_fam, packed_langs, tags, packed_lems)
        
        padded_pos, pos_ls = _rnn_utils.pad_packed_sequence(packed_pos)
        padded_emb, emb_ls = _rnn_utils.pad_packed_sequence(packed_emb)
        padded_x,     x_ls = _rnn_utils.pad_packed_sequence(packed_x)
        
        assert (pos_ls == emb_ls).all() and (emb_ls == x_ls).all(), "everything must be the same length"
        
        # For each input in this batch ...
        for i in range(batch_size):
            
            all_hiddens_for_single_item = tuple([h[:,i,:] for h in hidden])
            
            y = self.decoder(families[i], languages[i], tags[i],
                                padded_pos[:,i,:],
                                padded_emb[:,i,:], padded_x[:,i,:],
                                all_hiddens_for_single_item
                            )
            
            outputs = y.argmax(-1)#y was already stacked.
            
            # Put the results in the list.
            batch_probabilities.append(y)
            batch_outputs.append(outputs)
            
        # Report the results of this batch.
        return batch_probabilities
        
    
