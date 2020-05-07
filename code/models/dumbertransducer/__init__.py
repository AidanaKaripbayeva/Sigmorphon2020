from data.alphabets import Alphabet

import utils.rnns as BRNN
import utils.seq as BS

import torch

import torch.nn.utils.rnn as _rnn_utils

from .attention import *
from .windows import *


class LetterContextEncoder(torch.nn.Module):
    def __init__(self, alphabet_size, embedding_dim=40, hidden_dim=10, num_layers=3, context_before=2,context_after=1,context_dim=20,num_languages=None):
        super().__init__()
        #PARAMETERS
        # The number of different letters in the alphabet.
        self.alphabet_size = alphabet_size
        # The dimension of the output of the embedding layer.
        self.embedding_dim = embedding_dim
        # Then number of different languages.
        self.num_languages = num_languages
        
        self.context_before = context_before
        self.context_after = context_after
        self.context_dim = context_dim
        
        self.hidden_dim = [ hidden_dim ]*num_layers if isinstance(hidden_dim, int) else hidden_dim
        assert isinstance(self.hidden_dim, list) or isinstance(self.hidden_dim, tuple), str(type(self.hidden_dim))
        assert len(self.hidden_dim) == num_layers, "Either provide one integer for hidden_dim, or one for each layer"
        
        #LAYERS
        # Create the embedding layer for the encoder.
        self.embedding = BRNN.RNN_Input_Embedding(torch.nn.Embedding(alphabet_size, embedding_dim))
        
        self.context_embedder = torch.nn.Sequential(torch.nn.Linear(self.embedding_dim, self.context_dim))
        
        self.before_context = BRNN.RNN_FFLayer(torch.nn.Sequential(
                                            self.context_embedder,
                                            OffsetWindows(-context_before,True),
                                            torch.nn.Linear(context_before* self.context_dim, self.context_dim),
                                            torch.nn.Sigmoid()
                                        ))
        self.after_context  = BRNN.RNN_FFLayer(torch.nn.Sequential(
                                            self.context_embedder,
                                            OffsetWindows(context_after,True),
                                            torch.nn.Linear(context_after*self.context_dim, self.context_dim),
                                            torch.nn.Sigmoid()
                                        ))
        self.all_context = BRNN.RNN_FFLayer(torch.nn.Sequential(
                                            torch.nn.Linear(context_dim*2, context_dim),
                                            torch.nn.Sigmoid()
                                        ))
        
        
        #self.f_lstm = BRNN.RNN_FFLayer(torch.nn.Sequential(*tmp_layers))
    
    def forward(self, families, languages, tags, lemma):
        emb,_ = self.embedding(lemma)
        
        b4,_ = self.before_context(emb)
        aft,_ = self.after_context(emb)
        
        ctxt,_ = self.all_context(torch.cat([b4, aft],dim=2))
        
        return emb, ctxt, tuple()
        
        

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
    
    def forward(self, family, language, tags, embedding, states, hidden, tf_forms=None):
        
        output_sequence = torch.zeros(self.max_output_length, dtype=torch.int32)
        output_probs = torch.zeros((self.max_output_length, self.alphabet_size))
        
        output_sequence[0] = Alphabet.start_integer
        output_sequence[1:] = Alphabet.stop_integer #pading
        
        output_probs[0,Alphabet.start_integer] = 1.0
        output_probs[1:,Alphabet.stop_integer] = 1.0
        
        #Need to change the extents of this loop.
        
        #faster? FASTER!
        x = self.linear(embedding)
        x = self.sigmoid(x)
        y = self.final_softmax(x)
        output_symbols=y.argmax(-1)
        output_probs[:y.shape[0]] = y
        
        for output_i in range(1, min(self.max_output_length, embedding.shape[0])):
            if(output_symbols[output_i] == Alphabet.stop_integer):
                return output_probs[:output_i]
        return output_probs[:output_i]
        
        

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
        
        
        self.encoder = LetterContextEncoder(alphabet_size, embedding_dim)
        
        self.decoder = DumberDecoder(alphabet_size, tag_vector_dim, embedding_dim)
        
        self.device = None
        
    def to(self, target_device):
        self.device = target_device
        super().to(self.device)
        return self
        
    def do_decode(self, families, languages, tags, emb, rnn_states,
                        hidden_states, tf_forms=None):
        
        batch_size = len(tags)
        batch_outputs = []  # Will hold the integral representation of the characters to be output for this batch.
        batch_probabilities = []   # Will hold the probability vectors for this batch.
        
        # For each input in this batch ...
        for i in range(batch_size):
            
            all_hiddens_for_single_item = tuple([h[:,i,:] for h in hidden_states])
            
            y = self.decoder(
                                families[:,i],
                                languages[:,i], tags[i],
                                emb[:,i,:], rnn_states[:,i,:],
                                all_hiddens_for_single_item,
                                tf_forms=tf_forms[:,i]
                            )
            
            outputs = y.argmax(-1)#y was already stacked.
            
            # Put the results in the list.
            batch_probabilities.append(y)
            batch_outputs.append(outputs)
        
        return batch_probabilities, batch_outputs
    
    def forward(self, families, languages, tags, lemmata, tf_forms=None):
        batch_size = len(tags)
        batch_outputs = []  # Will hold the integral representation of the characters to be output for this batch.
        batch_probabilities = []   # Will hold the probability vectors for this batch.
        
        assert isinstance(lemmata, list) or isinstance(lemmata, tuple), "Type was: " + str(type(lemmata)) + " but only lists are implemented"
        lengths = [len(i) for i in lemmata]
        
        #DEVICE MOVEMENT #TODO: move out of here.
        if self.device is not None:
            families = [i.to(self.device) for i in families]
            languages = [i.to(self.device) for i in languages]
            tags = [i.to(self.device) for i in tags]
            lemmata = [i.to(self.device) for i in lemmata]
            if tf_forms is not None:
                assert isinstance(tf_forms, list) or isinstance(tf_forms, tuple), "only lists are implemented"
                tf_forms = [t.to(self.device) for t in tf_forms]
        
        #
        #The encoder uses layers that can do everything at once.
        #ENCODE EVERYTHING
        #
        padded_fam = _rnn_utils.pad_sequence(families, batch_first=False, padding_value=Alphabet.stop_integer)
        padded_langs = _rnn_utils.pad_sequence(languages, batch_first=False, padding_value=Alphabet.stop_integer)
        padded_lems = _rnn_utils.pad_sequence(lemmata, batch_first=False, padding_value=Alphabet.stop_integer)
        if tf_forms is not None:
            tf_forms = _rnn_utils.pad_sequence(tf_forms, batch_first=False, padding_value=Alphabet.stop_integer)
        
        padded_emb, padded_x, hidden = self.encoder(padded_fam, padded_langs, tags, padded_lems)
        
        
        batch_probabilities, batch_outputs = self.do_decode(padded_fam, padded_langs, tags, padded_emb, padded_x,
                            hidden, tf_forms=tf_forms)
            
        # Report the results of this batch.
        return batch_probabilities
        
    
