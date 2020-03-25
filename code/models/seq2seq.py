from data.alphabets import Alphabet
import torch


class Seq2Seq(torch.nn.Module):
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
        self.lstm1 = torch.nn.LSTM(tag_vector_dim[0] + embedding_dim, hidden_dim, num_layers, dropout=0.1,
                                   bidirectional=False)
        # Create the LSTM unit for the decoder.
        self.lstm2 = torch.nn.LSTM(tag_vector_dim[0] + alphabet_size, alphabet_size, num_layers, dropout=0.1)
        # Create the final linear layer for the decoder.
        self.linear = torch.nn.Linear(alphabet_size, alphabet_size)

    def forward(self, families, languages, tags, lemmata):
        batch_size = len(tags)
        batch_probabilities = []   # Will hold the probability vectors for this batch.

        # For each input in this batch ...
        for i in range(batch_size):
            probabilities = []  # Will hold the probability vectors for this input.

            # Encode
            # lemmata[i].shape == (seq_length,)
            x = self.embedding(lemmata[i]).float()
            # x.shape == (seq_length, embedding_dim)
            # tags[i].shape == (tag_vector_dim)
            # tags[i].expand((1, -1)).shape == (1, tag_vector_dim)
            # tags[i].expand((1, -1)).repeat((len(lemmata[i]), 1)).shape == (seq_length, tag_vector_dim)
            x = torch.cat([x.T, tags[i].float().expand((1, -1)).repeat((len(lemmata[i]), 1)).T]).T
            # x.shape == (seq_length, tag_vector_dim + embedding_dim)
            x = x.expand((1, x.shape[0], x.shape[1]))
            # x.shape == (1, seq_length, tag_vector_dim + embedding_dim)
            code, (hidden, cell) = self.lstm1(x)
            # code.shape == (1, seq_length, hidden_dim)
            # hidden.shape == (num_layers, seq_length, hidden_dim)
            # cell.shape == (num_layers, seq_length, hidden_dim)

            # Decode
            start_vector = torch.zeros((1, self.alphabet_size))
            start_vector[0, Alphabet.start_integer] = 1   # One-hot representation of the start token.
            # Start with a sequence containing only the start token.
            probabilities.append(start_vector)
            # Initialize the LSTM with zero vectors.
            hidden = torch.zeros(self.num_layers, 1, self.alphabet_size)
            cell = torch.zeros(self.num_layers, 1, self.alphabet_size)
            # For each position after the start token, in this output sequence ...
            for t in range(self.output_length - 1):
                # probabilities[-1].shape == (1, alphabet_size)
                # tags[i].shape == (tag_vector_dim)
                # tags[i].expand((1, -1)).shape == (1, tag_vector_dim)
                x = torch.cat([probabilities[-1].T, tags[i].float().expand((1, -1)).T]).T.expand((1, 1, -1))
                # x.shape == (1, 1, alphabet_size + tag_vector_dim)
                # hidden.shape = (num_layers, 1, alphabet_size)
                # cell.shape = (num_layers, 1, alphabet_size)
                y, (hidden, cell) = self.lstm2(x, (hidden, cell))
                # y.shape == (1, 1, alphabet_size)
                prediction = torch.sigmoid(self.linear(y.squeeze(0)))  # The prob. vector for this place in the output.
                # prediction.shape == (1, alphabet_size)
                probabilities.append(prediction)
            # Put the results in the list.
            batch_probabilities.append(torch.cat(probabilities))

        # Report the results of this batch.
        return batch_probabilities
