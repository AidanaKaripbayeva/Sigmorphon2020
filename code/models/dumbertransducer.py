from data.alphabets import Alphabet

import utils.rnns as BRNN
import utils.seq as BS

import torch

import torch.nn.utils.rnn as _rnn_utils

class DumberEncoder(torch.nn.Module):
    def __init__(self, alphabet_size, embedding_dim=40, hidden_dim=10, num_layers=1,num_languages=None):
        super().__init__()
        #PARAMETERS
        # The number of different letters in the alphabet.
        self.alphabet_size = alphabet_size
        # The dimension of the output of the embedding layer.
        self.embedding_dim = embedding_dim
        # Then number of different languages.
        self.num_languages = num_languages
        
        #LAYERS
        self.position = BS.RNN_Position_Encoding(20,20)
        # Create the embedding layer for the encoder.
        self.embedding = BRNN.RNN_Input_Embedding(torch.nn.Embedding(alphabet_size, embedding_dim))
        
        # Forward LSTM for encoding
        self.f_lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
    
    def forward(self, families, languages, tags, lemma):
        pos = self.position(lemma)#I expect it to blow up here.
        emb,_ = self.embedding(lemma)
        
        lstm_output, lstm_hidden = self.f_lstm(emb)
        
        return pos, emb, lstm_output, lstm_hidden
        

class DumberDecoder(torch.nn.Module):
    def __init__(self, alphabet_size, tag_vector_dim, embedding_dim=40, hidden_dim=10, num_layers=3, output_length=50, num_languages=None):
        super().__init__()
        self.linear = torch.nn.Linear(embedding_dim, alphabet_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.final_softmax = torch.nn.Softmax(dim=-1)
    
    def forward(self, family, language, tags, position, embedding, states, hidden):
        
        x = self.linear(embedding)
        x = self.sigmoid(x)
        y = self.final_softmax(x)
        
        
        return y
        
        

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
        

    def forward(self, families, languages, tags, lemmata):
        batch_size = len(tags)
        batch_outputs = []  # Will hold the integral representation of the characters to be output for this batch.
        batch_probabilities = []   # Will hold the probability vectors for this batch.
        
        assert isinstance(lemmata, list) or isinstance(lemmata, tuple), str(type(lemmata)) + " Only lists are implemented"
        lengths = [len(i) for i in lemmata]
        
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
            
            y = self.decoder(families[i], languages[i], tags[i],
                                padded_pos[:,i,:], padded_emb[:,i,:], padded_x[:,i,:],
                                hidden)
            
            outputs = y.argmax(-1)#y was already stacked.
            
            # Put the results in the list.
            batch_probabilities.append(y)
            batch_outputs.append(outputs)
            
        # Report the results of this batch.
        return batch_probabilities