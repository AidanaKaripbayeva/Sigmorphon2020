from data.alphabets import Alphabet
import torch


class DumbCopy(torch.nn.Module):
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

        # Create the embedding layer for the encoder.
        self.embedding = torch.nn.Embedding(alphabet_size, embedding_dim)
        # Create the LSTM unit for the encoder.
        self.linear = torch.nn.Linear(embedding_dim, alphabet_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.final_softmax = torch.nn.Softmax(dim=-1)

    def forward(self, families, languages, tags, lemmata):
        batch_size = len(tags)
        batch_outputs = []  # Will hold the integral representation of the characters to be output for this batch.
        batch_probabilities = []   # Will hold the probability vectors for this batch.
        
        # For each input in this batch ...
        for i in range(batch_size):
            #probabilities = []  # Will hold the probability vectors for this input.
            #outputs = []  # Will hold the integral representation of the characters to be output for this input.

            # Encode
            # lemmata[i].shape == (seq_length,)
            x = self.embedding(lemmata[i]).float()
            
            x = self.linear(x)
            x = self.sigmoid(x)
            y = self.final_softmax(x)
            # code.shape == (1, seq_length, hidden_dim)
            # hidden.shape == (num_layers, seq_length, hidden_dim)
            # cell.shape == (num_layers, seq_length, hidden_dim)
            
            
            #extra_ys = torch.stack([y[-1] for i in range(self.output_length-y.shape[0])])
            #y = torch.cat([y,extra_ys])
            
            outputs = y.argmax(-1)#y was already stacked.
            
            # Put the results in the list.
            batch_probabilities.append(y)
            batch_outputs.append(outputs)

        # Report the results of this batch.
        return batch_probabilities
