#model
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_heads, n_layers, dropout, pad_idx, max_len=100):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(emb_dim, max_len)
        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        return (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        B, T = tgt.shape
        pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        subsequent_mask = torch.tril(torch.ones((T, T), device=tgt.device)).bool()
        tgt_mask = subsequent_mask.unsqueeze(0).unsqueeze(1)
        return pad_mask & tgt_mask

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        src_emb = self.dropout(self.pos_encoder(self.embedding(src)))
        tgt_emb = self.dropout(self.pos_encoder(self.embedding(tgt)))

        output = self.transformer(
            src_emb, tgt_emb,
            src_key_padding_mask=(src == self.pad_idx),
            tgt_key_padding_mask=(tgt == self.pad_idx),
            memory_key_padding_mask=(src == self.pad_idx),
            tgt_mask=None
        )
        return self.fc_out(output)

