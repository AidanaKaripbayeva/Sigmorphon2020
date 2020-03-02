from data.unimorph_dataloader import decode_tags, START_TOKEN, END_TOKEN, PADDING_TOKEN
import data.unimorph_dataloader
import torch


class Seq2Seq(torch.nn.Module):
    def __init__(self, embedding, reverse_embedding, hidden_dim=10, num_layers=3):
        super().__init__()
        categories = data.unimorph_dataloader.category2int.keys()
        self.embedding = embedding
        self.lstm1 = torch.nn.LSTM(len(categories) + 1, hidden_dim, num_layers, dropout=0.1, bidirectional=False)
        self.lstm2 = torch.nn.LSTM(len(categories) + 1, hidden_dim, num_layers, dropout=0.1)
        self.linear = torch.nn.Linear(hidden_dim, 100)
        self.reverse_embedding = reverse_embedding

    def forward(self, stem, tags, family, language):
        batch_size = len(stem)
        outputs = []
        probabilities = []
        for i in range(batch_size):
            # Encode
            embedded = self.embedding(stem[i], language[i])
            tags_vector = decode_tags(tags[i])
            x = torch.cat([embedded.expand((1, -1)), tags_vector.expand((1, -1)).T.repeat((1, len(embedded)))])
            x = x.expand((1, x.shape[0], x.shape[1])).transpose(1, 0).transpose(2, 0)
            code, (hidden, cell) = self.lstm1(x)

            # Decode
            output = [START_TOKEN]
            probabilities.append([])
            for t in range(30):
                x = torch.cat([torch.Tensor([[output[-1]]]), tags_vector.expand((1, -1)).T]).T.expand((1, 1, -1))
                y, (hidden, cell) = self.lstm2(x, (hidden, cell))
                prediction = torch.sigmoid(self.linear(y.squeeze(0)))
                probabilities[i].append(prediction)
                output.append(prediction.argmax(1))
            probabilities[i] = torch.cat(probabilities[i])
            outputs.append(torch.cat(output[1:]))
        return probabilities, outputs
