from data.alphabets import Alphabet
import math
import random
import torch


EPSILON = 1e-7


class StackedLSTM(torch.nn.Module):
    def __init__(self, input_siz, rnn_siz, nb_layers, dropout):
        super().__init__()
        self.nb_layers = nb_layers
        self.rnn_siz = rnn_siz
        self.layers = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout)

        for _ in range(nb_layers):
            self.layers.append(torch.nn.LSTMCell(input_siz, rnn_siz))
            input_siz = rnn_siz

    def get_init_hx(self, batch_size):
        h_0_s, c_0_s = [], []
        for _ in range(self.nb_layers):
            h_0 = torch.zeros((batch_size, self.rnn_siz))
            c_0 = torch.zeros((batch_size, self.rnn_siz))
            h_0_s.append(h_0)
            c_0_s.append(c_0)
        return h_0_s, c_0_s

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = self.dropout(h_1_i)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class BabyTransducer(torch.nn.Module):
    """
    Sequence to sequence transformer model with attention mechanism.
    """
    def __init__(self, alphabet_size, tag_vector_dim, embed_dim, src_hid_size, src_nb_layers, trg_nb_layers,
                 trg_hid_size, attention_dim, dropout_p, teacher_force=0.0, output_length=50):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.tag_vector_dim = tag_vector_dim
        self.embed_dim = embed_dim
        self.src_hid_size = src_hid_size
        self.src_nb_layers = src_nb_layers
        self.trg_hid_size = trg_hid_size
        self.trg_nb_layers = trg_nb_layers
        self.dropout_p = dropout_p
        self.src_embed = torch.nn.Embedding(alphabet_size, embed_dim, padding_idx=Alphabet.stop_integer)
        self.trg_embed = torch.nn.Embedding(alphabet_size, embed_dim, padding_idx=Alphabet.stop_integer)
        self.enc_rnn = torch.nn.LSTM(embed_dim, src_hid_size, src_nb_layers, bidirectional=True,
                                     dropout=dropout_p)
        self.dec_rnn = StackedLSTM(embed_dim, trg_hid_size, trg_nb_layers, dropout_p)
        self.out_dim = trg_hid_size + trg_hid_size + tag_vector_dim
        self.scale_enc_hs = torch.nn.Linear(src_hid_size * 2, trg_hid_size)
        self.fc_out = torch.nn.Sequential(torch.nn.Linear(self.out_dim, self.out_dim),
                                          torch.nn.Linear(self.out_dim, alphabet_size))
        self.attention_dim = attention_dim
        self.enc_attention = torch.nn.Linear(trg_hid_size, attention_dim)
        self.dec_attention = torch.nn.Linear(trg_hid_size, attention_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.output_length = output_length
        self.teacher_force = teacher_force

    def attention(self, h_t, enc_hs):
        # h_t.shape == (batch_size, trg_hid_size)
        # enc_hs.shape == (len_lemma, batch_size, trg_hid_size)
        len_lemma = enc_hs.shape[0]
        trg_hid_size = enc_hs.shape[2]
        enc_attention_vec = self.enc_attention(enc_hs)
        # enc_attention_vec.shape == (len_lemma, batch_size, attention_dim)
        dec_attention_vec = self.dec_attention(h_t)
        # dec_attention_vec.shape == (batch_size, attention_dim)
        weights = (enc_attention_vec * dec_attention_vec[None, :, :].repeat((len_lemma, 1, 1))).sum(dim=2)
        # weights.shape == (len_lemma, batch_size)
        context = (weights[:, :, None].repeat((1, 1, trg_hid_size)) * enc_hs).sum(dim=0)
        # context.shape == (batch_size, trg_hid_size)
        return context

    def decode(self, enc_hs, family, language, tag, target=None):
        batch_size = enc_hs.shape[1]
        # enc_hs.shape == (len_lemma, batch_size, trg_hid_size)
        # family.shape == (batch_size, num_families)
        # language.shape == (batch_size, num_languages)
        # tag.shape == (batch_size, tag_vector_dim)
        # lemma.shape == (len_lemma, batch_size)
        # target.shape == (len_target, batch_size)
        # Start with a sequence containing only the start token.
        start_vector = get_one_hot_vec(self.alphabet_size, Alphabet.start_integer)
        # start_vector.shape == (alphabet_size, )
        output = [start_vector]
        hidden = self.dec_rnn.get_init_hx(batch_size)
        # len(hidden[0]) == len(hidden[1]) == trg_nb_layers
        # hidden[0][i].shape == hidden[1][i].shape == (batch_size, trg_hid_size)
        for idx in range(self.output_length):
            if target is not None and idx < len(target) and random.uniform(0, 1) < self.teacher_force:
                input_ = get_one_hot_vec(self.alphabet_size, target[idx])
            else:
                input_ = torch.max(output[-1], dim=1)[1]
            # input_ is an vector of shape (batch_size, ) in range(alphabet_size)?
            input_ = self.dropout(self.trg_embed(input_))
            # input_.shape == (batch_size, embed_dim)
            word_logprob, hidden = self.decode_step(input_, enc_hs, hidden, family, language, tag)
            # word_logprob.shape == (batch_size, alphabet_size)
            output.append(word_logprob)
        return output

    def decode_step(self, input_, enc_hs, hidden, family, language, tag):
        # input_.shape == (batch_size, embed_dim)
        # enc_hs.shape == (len_lemma, batch_size, trg_hid_size)
        # len(hidden[0]) == len(hidden[1]) == trg_nb_layers
        # hidden[0][i].shape == hidden[1][i].shape == (batch_size, trg_hid_size)
        # family.shape == (batch_size, num_families)
        # language.shape == (batch_size, num_languages)
        # tag.shape == (batch_size, tag_vector_dim)
        h_t, hidden = self.dec_rnn(input_, hidden)
        # h_t.shape == (batch_size, trg_hid_size)
        # len(hidden[0]) == len(hidden[1]) == trg_nb_layers
        # hidden[0].shape == hidden[1].shape == (trg_nb_layers, batch_size, trg_hid_size)
        context = self.attention(h_t, enc_hs)
        # context.shape == (batch_size, trg_hid_size)
        x = torch.cat((context, h_t, tag.float()), dim=1)
        # context.shape == (batch_size, trg_hid_size + trg_hid_size + tag_vector_dim)
        word_logprob = torch.nn.functional.log_softmax(torch.tanh(self.fc_out(x)), dim=-1)
        # word_logprob.shape == (batch_size, alphabet_size)
        return word_logprob, hidden

    def encode(self, lemma):
        # lemma.shape == (len_lemma, batch_size)
        # src_embed(lemma).shape == (len_lemma, batch_size, embed_dim)
        # dropout(src_embed(lemma)).shape == (len_lemma, batch_size, embed_dim)
        enc_hs, _ = self.enc_rnn(self.dropout(self.src_embed(lemma)))
        # enc_hs.shape = (len_lemma, batch_size, src_hid_size * 2)
        scale_enc_hs = self.scale_enc_hs(enc_hs)
        # scale_enc_hs.shape = (len_lemma, batch_size, trg_hid_size)
        return enc_hs, scale_enc_hs

    def forward(self, families, languages, tags, lemmata, targets=None):
        batch_size = len(families)
        # len(families) == batch_size
        # len(languages) == batch_size
        # len(tags)  == batch_size
        # len(lemmata) == batch_size
        # len(targets) == batch_size
        batch_output = []
        for i in range(batch_size):
            family = families[i][None, :]
            language = languages[i][None, :]
            tag = tags[i][None, :]
            lemma = lemmata[i][:, None]
            target = targets[i][:, None] if targets is not None else None
            # family.shape == (batch_size, num_families)
            # language.shape == (batch_size, num_languages)
            # tag.shape == (batch_size, tag_vector_dim)
            # lemma.shape == (len_lemma, batch_size)
            # target.shape == (len_target, batch_size)
            # Encoder
            enc_hs, scale_enc_hs = self.encode(lemma)
            # enc_hs.shape = (len_lemma, batch_size, src_hid_size * 2)
            # scale_enc_hs.shape = (len_lemma, batch_size, trg_hid_size)
            output = self.decode(scale_enc_hs, family, language, tag, target)
            batch_output.append(torch.cat(output))
        return batch_output


def fancy_gather(value, index):
    assert value.size(1) == index.size(1)
    split = zip(value.split(1, dim=1), index.split(1, dim=1))
    return torch.cat([v[i.view(-1)] for v, i in split], dim=1)


def get_one_hot_vec(length, index):
    vector = torch.zeros((1, length))
    vector[0, index] = 1
    return vector
