from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


CellType = Literal["rnn", "lstm", "gru"]


class TextRNNClassifier(nn.Module):
    """
    Flexible text classifier supporting RNN/LSTM/GRU, uni- or bi-directional, with configurable layers.
    Expects token id inputs of shape [batch, seq_len] (LongTensor).
    """

    def __init__(
        self,
        cell_type: CellType,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.cell_type = cell_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.bidirectional = bool(bidirectional)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)

        rnn_kwargs = dict(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        if cell_type == "rnn":
            self.rnn = nn.RNN(**rnn_kwargs)
        elif cell_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif cell_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}")

        num_directions = 2 if self.bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * num_directions, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len]
        emb = self.embedding(x)  # [batch, seq_len, embed_dim]
        out = self.rnn(emb)
        if self.cell_type == "lstm":
            _, (hn, _) = out  # hn shape: [num_layers * num_dirs, batch, hidden_dim]
        else:
            _, hn = out  # hn shape: [num_layers * num_dirs, batch, hidden_dim]

        if self.bidirectional:
            # Concatenate the last forward and last backward hidden states
            # Forward last layer: hn[-2], Backward last layer: hn[-1]
            hidden_cat = torch.cat((hn[-2], hn[-1]), dim=1)  # [batch, hidden_dim*2]
            logits = self.fc(self.dropout(hidden_cat))
        else:
            last_hidden = hn[-1]  # [batch, hidden_dim]
            logits = self.fc(self.dropout(last_hidden))
        return logits


