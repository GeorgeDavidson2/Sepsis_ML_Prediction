"""
tests/test_condition_D_notebook.py
───────────────────────────────────────────────────────────────────────────────
Smoke test for the Condition D notebook logic before running on Colab.

Tests:
  1. make_scaled_df (Strategy B) includes ICULOS — prevents SepsisDataset crash
  2. SepsisDataset can be instantiated with Strategy B data
  3. SepsisLSTM forward pass works with 76 input features
  4. One training step runs without error

Run with:
    source venv_ml/bin/activate
    python -m pytest tests/test_condition_D_notebook.py -v
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch
import pytest

from src.models import SepsisLSTM
from src.train import SepsisDataset, make_loaders, train_lstm


# ── Synthetic data helpers ────────────────────────────────────────────────────

N_PATIENTS  = 20
N_FEATURES  = 76   # Strategy B: 40 original + 36 missingness indicators
SEQ_LEN_MAX = 10
BATCH_SIZE  = 4


def _make_synthetic_df(patient_ids, n_features=N_FEATURES, max_iculos=SEQ_LEN_MAX):
    """Build a minimal DataFrame mimicking Strategy B preprocessed output."""
    feat_cols = [f'feat_{i}' for i in range(n_features)]
    rows = []
    for pid in patient_ids:
        length = np.random.randint(3, max_iculos + 1)
        for t in range(1, length + 1):
            row = {'patient_id': pid, 'ICULOS': t, 'EarlyLabel': float(t == length)}
            for col in feat_cols:
                row[col] = np.random.randn()
            rows.append(row)
    return pd.DataFrame(rows), feat_cols


def _make_scaled_df_strategy_B(original_df, X_scaled, feat_cols):
    """Exact copy of make_scaled_df from the Condition D notebook."""
    meta = original_df[['patient_id', 'ICULOS', 'EarlyLabel']].reset_index(drop=True)
    feat = pd.DataFrame(X_scaled, columns=feat_cols)
    return pd.concat([meta, feat], axis=1)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_make_scaled_df_includes_iculos():
    """ICULOS must survive make_scaled_df so SepsisDataset can sort by it."""
    pids = list(range(N_PATIENTS))
    df, feat_cols = _make_synthetic_df(pids)
    X_scaled = np.random.randn(len(df), N_FEATURES).astype(np.float32)

    result = _make_scaled_df_strategy_B(df, X_scaled, feat_cols)

    assert 'ICULOS' in result.columns, "ICULOS missing from DataFrame — SepsisDataset will crash"
    assert 'patient_id' in result.columns
    assert 'EarlyLabel' in result.columns
    assert len([c for c in result.columns if c.startswith('feat_')]) == N_FEATURES


def test_sepsis_dataset_strategy_B():
    """SepsisDataset must instantiate without KeyError on ICULOS."""
    pids = list(range(N_PATIENTS))
    df, feat_cols = _make_synthetic_df(pids)
    X_scaled = np.random.randn(len(df), N_FEATURES).astype(np.float32)
    scaled_df = _make_scaled_df_strategy_B(df, X_scaled, feat_cols)

    # Should not raise
    ds = SepsisDataset(scaled_df, pids, feat_cols, max_seq_len=SEQ_LEN_MAX)
    assert len(ds) == N_PATIENTS

    X, y, length = ds[0]
    assert X.shape == (SEQ_LEN_MAX, N_FEATURES)
    assert y.shape == (SEQ_LEN_MAX,)
    assert 1 <= length <= SEQ_LEN_MAX


def test_lstm_forward_pass_76_features():
    """SepsisLSTM must accept 76-feature input without shape errors."""
    model = SepsisLSTM(input_size=N_FEATURES, hidden_size=64, num_layers=2, dropout=0.2)
    model.eval()

    batch = 4
    seq   = SEQ_LEN_MAX
    x       = torch.randn(batch, seq, N_FEATURES)
    lengths = torch.tensor([seq, seq - 2, seq - 4, seq - 5])

    # Sort descending (required by pack_padded_sequence)
    lengths, idx = lengths.sort(descending=True)
    x = x[idx]

    with torch.no_grad():
        logits = model(x, lengths)

    assert logits.shape[0] == batch
    assert logits.shape[1] <= seq   # pad_packed_sequence returns max real length


def test_one_training_step_strategy_B():
    """Full train_lstm call must complete one epoch without error on synthetic data."""
    np.random.seed(42)
    torch.manual_seed(42)

    all_pids   = list(range(N_PATIENTS))
    train_pids = all_pids[:12]
    val_pids   = all_pids[12:16]
    test_pids  = all_pids[16:]

    df, feat_cols = _make_synthetic_df(all_pids)
    X_scaled = np.random.randn(len(df), N_FEATURES).astype(np.float32)
    scaled_df = _make_scaled_df_strategy_B(df, X_scaled, feat_cols)

    train_df = scaled_df[scaled_df['patient_id'].isin(train_pids)]
    val_df   = scaled_df[scaled_df['patient_id'].isin(val_pids)]
    test_df  = scaled_df[scaled_df['patient_id'].isin(test_pids)]

    train_loader, val_loader, _ = make_loaders(
        train_df, val_df, test_df,
        train_pids, val_pids, test_pids,
        feature_cols=feat_cols, batch_size=BATCH_SIZE,
    )

    model = SepsisLSTM(input_size=N_FEATURES, hidden_size=32, num_layers=2, dropout=0.2)

    model, best_auprc = train_lstm(
        model, train_loader, val_loader,
        n_epochs=2, lr=1e-3,
        pos_weight=10.0,
        patience=5, device='cpu',
    )

    assert best_auprc >= 0.0
    assert best_auprc <= 1.0
