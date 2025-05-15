#model
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))  #[batch, src_len, emb_dim]
        outputs, hidden = self.rnn(embedded)           #outputs: [batch, src_len, hidden_dim]
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        #decoder_hidden: [batch, hidden_dim]
        #encoder_outputs: [batch, src_len, hidden_dim]
        src_len = encoder_outputs.size(1)

        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)  #[batch, src_len, hidden_dim]
        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2)))  #[batch, src_len, hidden_dim]
        attention = self.v(energy).squeeze(2)  #[batch, src_len]
        return F.softmax(attention, dim=1)  #normalized weights


class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hidden_dim, hidden_dim)
        self.rnn = nn.LSTM(emb_dim + hidden_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        #input: [batch] -> [batch, 1]
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))  #[batch, 1, emb_dim]

        #hidden: (hidden_state, cell_state)
        dec_hidden = hidden[0][-1]  #last layer hidden state

        attn_weights = self.attention(dec_hidden, encoder_outputs)  #[batch, src_len]
        attn_weights = attn_weights.unsqueeze(1)  #[batch, 1, src_len]
        context = torch.bmm(attn_weights, encoder_outputs)  #[batch, 1, hidden_dim]

        rnn_input = torch.cat((embedded, context), dim=2)  #[batch, 1, emb_dim + hidden_dim]
        output, hidden = self.rnn(rnn_input, hidden)

        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1)), dim=1))  #[batch, output_dim]
        return prediction, hidden


class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device, sos_idx, eos_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        output_dim = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs

