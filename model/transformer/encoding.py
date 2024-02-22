import torch
from numpy import sin, cos
from torch import nn

from model.transformer.block import TransformerBlock, DecoderBlock


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_length=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_length, embed_size)
        for pos in range(max_length):
            for i in range(0, embed_size, 2):
                self.encoding[pos, i] = sin(pos / (10000 ** ((2 * i) / embed_size)))
                self.encoding[pos, i + 1] = cos(pos / (10000 ** ((2 * (i + 1)) / embed_size)))
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()


class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(input_vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.dropout(self.positional_encoding(self.word_embedding(x)))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length):
        super(Decoder, self).__init__()
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, max_length)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.dropout(self.positional_encoding(self.word_embedding(x)))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        return self.fc_out(x)
