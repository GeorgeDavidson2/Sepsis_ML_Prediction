"""
src/models.py
Neural network architectures for the Sepsis ML pipeline.
"""

import torch
import torch.nn as nn


class SepsisLSTM(nn.Module):
    """
    Many-to-many LSTM that outputs a risk score at every timestep.

    Uses packed sequences so padded positions are skipped inside the LSTM.
    Dropout is disabled on a single-layer model to avoid a PyTorch warning.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super(SepsisLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch, max_seq_len) — apply sigmoid for probabilities."""
        # Pack to skip padded positions inside the LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        output, _   = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        output = self.dropout(output)
        logits = self.fc(output).squeeze(-1)  # (batch, seq_len)
        return logits
