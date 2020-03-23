from data.alphabets import Alphabet
import torch


class Seq2Seq(torch.nn.Module):
    def __init__(self, alphabet_size, tag_vector_dim, embedding_dim=40, hidden_dim=10, num_layers=3):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.tag_vector_dim = tag_vector_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = torch.nn.Embedding(alphabet_size, embedding_dim)
        print(tag_vector_dim, embedding_dim, hidden_dim, num_layers, alphabet_size)
        self.lstm1 = torch.nn.LSTM(tag_vector_dim + embedding_dim, hidden_dim, num_layers, dropout=0.1,
                                   bidirectional=False)
        self.lstm2 = torch.nn.LSTM(tag_vector_dim + alphabet_size, alphabet_size, num_layers, dropout=0.1)
        self.linear = torch.nn.Linear(alphabet_size, alphabet_size)

    def forward(self, families, languages, tags, lemmata):
        batch_size = len(lemmata)
        outputs = []
        probabilities = []
        tags = tags.float()
        for i in range(batch_size):
            batch_probabilities = []
            batch_outputs = []
            # Encode
            x = self.embedding(lemmata[i]).float()
            x = torch.cat([x.T, tags[i, :].expand((1, -1)).repeat((len(lemmata[i]), 1)).T]).T
            x = x.expand((1, x.shape[0], x.shape[1]))
            code, (hidden, cell) = self.lstm1(x)

            # Decode
            start_vector = torch.zeros((1, self.alphabet_size))
            start_vector[0, Alphabet.start_integer] = 1
            output = [Alphabet.start_integer]
            batch_probabilities.append(start_vector)
            hidden = torch.zeros(self.num_layers, 1, self.alphabet_size)
            cell = torch.zeros(self.num_layers, 1, self.alphabet_size)
            for t in range(30):
                x = torch.cat([batch_probabilities[-1].T, tags[i, :].expand((1, -1)).T]).T.expand((1, 1, -1))
                y, (hidden, cell) = self.lstm2(x, (hidden, cell))
                prediction = torch.sigmoid(self.linear(y.squeeze(0)))
                batch_probabilities.append(prediction)
                batch_outputs.append(prediction.argmax(1))
            probabilities.append(torch.cat(batch_probabilities))
            outputs.append(batch_outputs)
        return probabilities, outputs
