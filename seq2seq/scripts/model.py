import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout, pad_idx, rnn_type="gru"):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn_type = rnn_type.lower()
        self.dropout = nn.Dropout(dropout)

        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(emb_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))  #[batch, src_len, emb_dim]
        outputs, hidden = self.rnn(embedded)          #hidden = [n_layers, batch, hidden_dim]
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, pad_idx, rnn_type="gru"):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.rnn_type = rnn_type.lower()
        self.dropout = nn.Dropout(dropout)

        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(emb_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden):
        #input: [batch] -> [batch, 1]
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))  #[batch, 1, emb_dim]

        if self.rnn_type == "gru":
            output, hidden = self.rnn(embedded, hidden)  #output: [batch, 1, hidden_dim]
        else:  #LSTM returns (output, (hidden, cell))
            output, (hidden, cell) = self.rnn(embedded, hidden)
            hidden = (hidden, cell)

        prediction = self.fc_out(output.squeeze(1))  #[batch, output_dim]
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, sos_idx, eos_idx, rnn_type="gru"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.rnn_type = rnn_type.lower()

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        #src: [batch, src_len], trg: [batch, trg_len]
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        output_dim = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)

        hidden = self.encoder(src)
        if self.rnn_type == "lstm":
            #wrap hidden in a tuple if not already
            if not isinstance(hidden, tuple):
                hidden = (hidden, torch.zeros_like(hidden))

        input = trg[:, 0]  #start with <sos> token

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs

