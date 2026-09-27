import torch
import torch.nn as nn

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
NUM_CLASSES = 3
CLASS_NAMES = ["Negative", "Neutral", "Positive"]


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128,
                 num_layers=1, num_classes=NUM_CLASSES, pad_idx=0, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        # x: (batch, seq_len) token ids
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (hidden, _cell) = self.lstm(packed)

        # hidden: (num_layers * 2, batch, hidden_dim) -> take last layer, both directions
        forward_h = hidden[-2, :, :]
        backward_h = hidden[-1, :, :]
        final_hidden = torch.cat([forward_h, backward_h], dim=1)  # (batch, hidden_dim*2)

        out = self.dropout(final_hidden)
        logits = self.fc(out)
        return logits
