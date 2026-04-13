"""
src/models.py
──────────────────────────────────────────────────────────────────────────────
Neural network architectures for the Sepsis ML pipeline.

Usage:
    from src.models import SepsisLSTM
    model = SepsisLSTM(input_size=40)
"""

import torch
import torch.nn as nn


class SepsisLSTM(nn.Module):
    """
    Many-to-many LSTM for hourly sepsis risk prediction.

    Reads a patient's full ICU timeline (variable length, up to MAX_SEQ_LEN
    hours) and outputs a risk score at every timestep. Uses packed sequences
    to skip padded timesteps during LSTM computation.

    Parameters
    ----------
    input_size  : number of features per timestep (40 for Strategy A, 76 for B)
    hidden_size : LSTM hidden state dimension (default 64)
    num_layers  : number of stacked LSTM layers (default 2)
    dropout     : dropout rate applied between LSTM layers and before FC (default 0.3)
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
        """
        Parameters
        ----------
        x       : (batch, max_seq_len, input_size) — padded feature sequences
        lengths : (batch,) — true sequence length for each patient

        Returns
        -------
        logits : (batch, max_seq_len) — raw (pre-sigmoid) risk score per timestep
        """
        # Pack to skip padded positions inside the LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        output, _   = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        output = self.dropout(output)
        logits = self.fc(output).squeeze(-1)  # (batch, seq_len)
        return logits
