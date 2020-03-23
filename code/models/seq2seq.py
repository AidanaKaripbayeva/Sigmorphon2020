from data.alphabets import Alphabet
import torch


class Seq2Seq(torch.nn.Module):
    def __init__(self, alphabet_vector_dim, tag_vector_dim, hidden_dim=10, num_layers=3):
        super().__init__()
        self.alphabet_vector_dim = alphabet_vector_dim
        self.tag_vector_dim = tag_vector_dim
        self.lstm1 = torch.nn.LSTM(tag_vector_dim + alphabet_vector_dim, hidden_dim, num_layers, dropout=0.1,
                                   bidirectional=False)
        self.lstm2 = torch.nn.LSTM(tag_vector_dim + alphabet_vector_dim, hidden_dim, num_layers, dropout=0.1)
        self.linear = torch.nn.Linear(hidden_dim, 100)

    def forward(self, families, languages, tags, lemmata):
        batch_size = len(families)
        outputs = []
        probabilities = []
        for i in range(batch_size):
            # Encode
            x = torch.cat([lemmata[i].expand((1, -1)), tags.expand((1, -1)).T.repeat((1, len(lemmata[i])))])
            x = x.expand((1, x.shape[0], x.shape[1])).transpose(1, 0).transpose(2, 0)
            code, (hidden, cell) = self.lstm1(x)

            # Decode
            output = [Alphabet.start_integer]
            probabilities.append([])
            for t in range(30):
                x = torch.cat([torch.Tensor([[output[-1]]]), tags[i].expand((1, -1)).T]).T.expand((1, 1, -1))
                y, (hidden, cell) = self.lstm2(x, (hidden, cell))
                prediction = torch.sigmoid(self.linear(y.squeeze(0)))
                probabilities[i].append(prediction)
                output.append(prediction.argmax(1))
            probabilities[i] = torch.cat(probabilities[i])
            outputs.append(torch.cat(output[1:]))
        return probabilities, outputs
