"""
src/train.py
──────────────────────────────────────────────────────────────────────────────
Dataset, DataLoader factory, and training loop for SepsisLSTM.

Usage:
    from src.train import SepsisDataset, make_loaders, train_lstm
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score

from src.config import MAX_SEQ_LEN, RANDOM_SEED


# ── Dataset ───────────────────────────────────────────────────────────────────

class SepsisDataset(Dataset):
    """
    PyTorch Dataset that groups rows by patient and pads sequences to MAX_SEQ_LEN.

    Each item is:
        X       : (MAX_SEQ_LEN, n_features) float32 tensor — padded with zeros
        y       : (MAX_SEQ_LEN,) float32 tensor — padded with zeros
        length  : int — true sequence length (number of real timesteps)
    """

    def __init__(
        self,
        df,                   # patient-level DataFrame with feature cols + EarlyLabel
        patient_ids: list,
        feature_cols: list,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.samples      = []
        self.feature_cols = feature_cols
        self.max_seq_len  = max_seq_len

        # Group once — avoids O(n×m) full-DataFrame scan per patient
        pid_set = set(patient_ids)
        grouped = (
            df[df['patient_id'].isin(pid_set)]
            .sort_values(['patient_id', 'ICULOS'])
            .groupby('patient_id', sort=False)
        )

        for pid in patient_ids:
            if pid not in grouped.groups:
                continue
            pat = grouped.get_group(pid)
            seq_len = min(len(pat), max_seq_len)

            X = np.zeros((max_seq_len, len(feature_cols)), dtype=np.float32)
            y = np.zeros(max_seq_len, dtype=np.float32)

            X[:seq_len] = pat[feature_cols].values[:seq_len]
            y[:seq_len] = pat['EarlyLabel'].values[:seq_len]

            self.samples.append((
                torch.tensor(X),
                torch.tensor(y),
                seq_len,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate_fn(batch):
    """Sort batch by descending sequence length (required for pack_padded_sequence)."""
    batch = sorted(batch, key=lambda x: x[2], reverse=True)
    X_batch   = torch.stack([b[0] for b in batch])
    y_batch   = torch.stack([b[1] for b in batch])
    lengths   = torch.tensor([b[2] for b in batch])
    return X_batch, y_batch, lengths


def make_loaders(
    train_df, val_df, test_df,
    patient_train, patient_val, patient_test,
    feature_cols: list,
    batch_size: int = 64,
) -> tuple:
    """
    Build train, val, and test DataLoaders for SepsisLSTM.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_ds = SepsisDataset(train_df, patient_train, feature_cols)
    val_ds   = SepsisDataset(val_df,   patient_val,   feature_cols)
    test_ds  = SepsisDataset(test_df,  patient_test,  feature_cols)

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate_fn, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=_collate_fn)

    return train_loader, val_loader, test_loader


# ── Training loop ─────────────────────────────────────────────────────────────

def train_lstm(
    model,
    train_loader,
    val_loader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    pos_weight: float = None,
    patience: int = 10,
    device: str = 'cpu',
) -> tuple:
    """
    Training loop for SepsisLSTM.

    Features:
    - BCEWithLogitsLoss with pos_weight for class imbalance
    - Gradient clipping (max_norm=1.0) — prevents exploding gradients on long sequences
    - ReduceLROnPlateau scheduler — halves lr when val AUPRC plateaus for 5 epochs
    - Early stopping — halts when val AUPRC doesn't improve for 'patience' epochs

    Parameters
    ----------
    model       : SepsisLSTM instance
    train_loader: DataLoader from make_loaders()
    val_loader  : DataLoader from make_loaders()
    n_epochs    : maximum training epochs
    lr          : initial learning rate
    pos_weight  : scalar weight for positive class (neg/pos count ratio)
    patience    : early stopping patience in epochs
    device      : 'cuda' or 'cpu'

    Returns
    -------
    model           : best model (weights restored from best val epoch)
    best_val_auprc  : best validation AUPRC achieved
    """
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    pw = torch.tensor([pos_weight], dtype=torch.float32).to(device) if pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_val_auprc   = 0.0
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        # ── Train ────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for X_batch, y_batch, lengths in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch, lengths)

            # pad_packed_sequence returns seq_len = longest real sequence in
            # this batch, which may be < MAX_SEQ_LEN. Align mask and labels.
            seq_len = logits.shape[1]
            mask = torch.zeros(y_batch.shape[0], seq_len, dtype=torch.bool, device=device)
            for i, l in enumerate(lengths):
                mask[i, :min(int(l), seq_len)] = True

            loss = criterion(logits[mask], y_batch[:, :seq_len].to(device)[mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──────────────────────────────────────
        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch, lengths in val_loader:
                logits = model(X_batch.to(device), lengths)
                probs  = torch.sigmoid(logits)

                seq_len = logits.shape[1]
                mask = torch.zeros(y_batch.shape[0], seq_len, dtype=torch.bool)
                for i, l in enumerate(lengths):
                    mask[i, :min(int(l), seq_len)] = True

                all_probs.extend(probs[mask].cpu().numpy())
                all_labels.extend(y_batch[:, :seq_len][mask].numpy())

        val_auprc = average_precision_score(all_labels, all_probs)
        scheduler.step(val_auprc)

        avg_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1:>3}/{n_epochs} | loss: {avg_loss:.4f} | val AUPRC: {val_auprc:.4f}')

        if val_auprc > best_val_auprc:
            best_val_auprc   = val_auprc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
                break

    model.load_state_dict(best_model_state)
    return model, best_val_auprc
