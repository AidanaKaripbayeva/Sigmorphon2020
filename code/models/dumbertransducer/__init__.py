from data.alphabets import Alphabet

import utils.rnns as BRNN
import utils.seq as BS

import torch

import torch.nn.utils.rnn as _rnn_utils

from .attention import *
from .windows import *
from .pretrainer import TWGPretrainer

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
        self.embedding = BRNN.RNN_Input_Embedding(torch.nn.Embedding(alphabet_size, self.embedding_dim))
        
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
    def __init__(self, alphabet_size, tag_vector_dim, embedding_dim=40, hidden_dim=10, num_layers=3, output_length=50, context_dim = 20, num_languages=None):
        super().__init__()
        self.tag_vector_dim = tag_vector_dim[0] if isinstance(tag_vector_dim, torch.Size) else tag_vector_dim
        self.max_output_length = output_length
        self.alphabet_size = alphabet_size
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.abs_freqs = 60
        self.rel_freqs = 60
        
        
        self.scratch_space = torch.zeros((self.max_output_length, self.abs_freqs + self.context_dim + self.alphabet_size + self.tag_vector_dim))
        
        self.abs_position = self.scratch_space[:,:self.abs_freqs]
        
        BS.RNN_Absolute_Position(self.abs_freqs).create_matrix(output_length,out=self.abs_position)
        #self.rel_position = BS.RNN_Relative_Position(self.rel_freqs)
        
        self.context_before = 1
        self.attention_dimension = 100
        
        
        
        self.context_reembedder = torch.nn.Sequential(torch.nn.Linear(self.alphabet_size, self.context_dim),
                                                    torch.nn.Sigmoid()
                                                    )
        self.before_context = BRNN.RNN_FFLayer(torch.nn.Sequential(
                                            self.context_reembedder,
                                            OffsetWindows(-self.context_before,True),
                                            torch.nn.Linear(self.context_before* self.context_dim, self.context_dim),
                                            torch.nn.Sigmoid()
                                        ))
        
        #self.linear = torch.nn.Linear(self.tag_vector_dim + alphabet_size + embedding_dim + hidden_dim, alphabet_size) #self.linear = torch.nn.Linear(embedding_dim, alphabet_size)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear((self.context_dim+self.embedding_dim) +
                            (self.context_dim + alphabet_size )   +
                            (self.context_dim+alphabet_size+self.tag_vector_dim), alphabet_size),
            torch.nn.Sigmoid(),
            torch.nn.Softmax(dim=-1)
            )
            
        _ = """self.ff = torch.nn.Sequential(
            torch.nn.Linear(
                            (self.context_dim+alphabet_size+self.tag_vector_dim), alphabet_size),
            torch.nn.Sigmoid(),
            torch.nn.Softmax(dim=-1)
            )
        """
        
        self.input_attention = KeyValueAttention(
                torch.nn.Sequential(
                    torch.nn.Linear(self.abs_freqs+self.context_dim+self.tag_vector_dim+self.alphabet_size, self.attention_dimension),
                    torch.nn.Sigmoid(),
                    ),
                torch.nn.Sequential(
                    torch.nn.Linear(self.abs_freqs+self.context_dim+self.embedding_dim, self.attention_dimension),
                    torch.nn.Sigmoid(),
                    ),
                False
            )
        
        self.output_attention = KeyValueAttention(
                torch.nn.Sequential(
                    torch.nn.Linear(self.abs_freqs+self.context_dim+self.tag_vector_dim+self.alphabet_size, self.attention_dimension),
                    torch.nn.Sigmoid(),
                    ),
                torch.nn.Sequential(
                    torch.nn.Linear(self.abs_freqs+self.context_dim+self.alphabet_size, self.attention_dimension),
                    torch.nn.Sigmoid(),
                    ),
                False
            )
        
    
    def forward(self, family, language, tags, embedding, states, hidden, tf_forms=None):
        
        output_sequence = torch.zeros(self.max_output_length, dtype=torch.int32)
        
        
        #output_probs = self.scratch_space[:,self.abs_freqs+self.context_dim:self.abs_freqs+self.context_dim+self.alphabet_size]
        #output_probs = torch.zeros((self.max_output_length, self.alphabet_size))
        output_probs = torch.zeros((1, self.alphabet_size))
        
        output_sequence[0] = Alphabet.start_integer
        output_sequence[1:] = Alphabet.stop_integer #pading
        
        output_probs[0,Alphabet.start_integer] = 1.0
        output_probs[1:,Alphabet.stop_integer] = 1.0
        
        #output_context = self.scratch_space[:,self.abs_freqs:self.abs_freqs+self.context_dim]
        output_context = torch.zeros(self.max_output_length, self.context_dim)
        output_context[0,:] = self.before_context(output_probs[:1])[0]
        #DEBUG
        output_context = self.before_context(output_probs[:1])[0][:1,:]
        
        #TODO: This must be megaslow
        self.scratch_space[:,self.abs_freqs+self.context_dim+self.alphabet_size:self.abs_freqs+self.context_dim+self.alphabet_size+self.tag_vector_dim] = tags
        
        one_output_probs = torch.zeros((1, self.alphabet_size))
        
        input_stack = torch.cat([states,embedding],dim=-1)
        positional_input_stack = torch.cat([self.abs_position[:input_stack.shape[0],:],input_stack],dim=-1)
        
        tf_output_probs = output_probs
        if tf_forms is not None:
            tf_output_probs = torch.zeros((self.max_output_length, self.alphabet_size))
            tf_output_probs[:tf_forms.shape[0],tf_forms] = 1.0
        
        #sequential decoding
        for output_i in range(1, min(self.max_output_length, embedding.shape[0])):
            #output_context[output_i,:] = self.before_context(output_probs[max(0,output_i-self.context_before):output_i].detach())[0]
            
            use_for_context = output_probs
            assert output_probs.shape[0] == output_i, "Wrong size for output probs"
            if tf_forms is not None:
                use_for_context = tf_output_probs[:output_i]
            #    running_output_probs[output_i,:] = 0.0
            #    running_output_probs[output_i,tf_forms[output_i]] = 1.0;
            
            output_context = self.before_context(use_for_context)[0]
            attention_query_input = torch.cat([self.abs_position[output_i,:].reshape(1,-1),
                                                                        output_context[output_i-1,:].reshape(1,-1),
                                                                        use_for_context[output_i-1,:].reshape(1,-1),
                                                                        tags.float().reshape(1,-1)],
                                                                dim=-1)
            input_attention_results = self.input_attention(attention_query_input,
                                                        positional_input_stack,
                                                        input_stack)
            output_attention_results = torch.zeros(1,(self.context_dim + self.alphabet_size ))
            #output_attention_results = self.output_attention(attention_query_input,
            #                                            torch.cat([])
            #                                            )
            #output_probs[output_i,:] = self.ff(self.scratch_space[output_i-1,self.abs_freqs:])
            #blah = [output_context[-1:,:], output_probs[-1:,:], tags[None,:].float()]
            #for i in blah:
            #    print("shape",i.shape)
            #import sys;sys.exit(1)
            
            #output_probs[output_i,:] = self.ff(torch.cat([one_output_context, output_probs[-1:,:].detach(), tags[None,:].float().detach()],dim=-1))
            one_output_probs = self.ff(torch.cat([
                                    input_attention_results,
                                    output_attention_results,
                                        output_context[-1:,:], use_for_context[-1:,:], tags[None,:].float().detach()
                                        ],dim=-1))
            output_probs = torch.cat([output_probs, one_output_probs],dim=0)
            
            
            output_sequence[output_i] = torch.argmax(output_probs[output_i],dim=-1)
            
            if(output_sequence[output_i] == Alphabet.stop_integer):
                return output_probs[:output_i]
        return output_probs[:output_i]
        
        

class DumberTransducer(torch.nn.Module):
    def __init__(self, alphabet_size, num_languages, tag_vector_dim, embedding_dim=40, hidden_dim=10, num_layers=3, output_length=50):
        super().__init__()
        self.pretrainer = TWGPretrainer
        # The number of different letters in the alphabet.
        self.alphabet_size = alphabet_size
        # Then number of different languages.
        self.num_languages = num_languages
        # The dimension of the tag vector.
        self.tag_vector_dim = tag_vector_dim if isinstance(tag_vector_dim, int) else tag_vector_dim[0]
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
                                tf_forms=None if tf_forms is None else tf_forms[:,i]
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
        
    
