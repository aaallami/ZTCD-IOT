#!/usr/bin/env python3
"""
badm_synthetic_selfcheck.py

Runs the BADM pipeline on synthetic data as an environment sanity check
prior to running the real UNSW-NB15/TON_IoT evaluations. Not used to
produce a reported figure; it confirms the pipeline runs end to end and
gives an indication of training time before downloading the full
datasets.

Two implementation details are relevant to reproducing the manuscript's
methodology:

  - Windows are genuinely overlapping (stride < window length),
    consistent with the sliding-window description in Section IV-E1.
  - The anomaly threshold is a percentile of the training partition's
    own score distribution, computed without reference to the test
    labels.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------
# Synthetic data generator -- stands in for UNSW-NB15/TON_IoT only for
# this feasibility test. NOT used for any number that goes in the paper.
# ---------------------------------------------------------------------
N_FEATURES = 10
N_DEVICES = 50
T_STEPS_PER_DEVICE = 2000          # timesteps of telemetry per device
ANOMALY_FRACTION = 0.05            # fraction of timesteps that are anomalous (test set only)


def generate_synthetic_data():
    """Each device has a stable 'normal' feature distribution. A random
    subset of timesteps are perturbed to simulate anomalous behavior.
    Ground-truth labels are kept ONLY for evaluation after the fact --
    never used to pick the threshold."""
    rng = np.random.RandomState(SEED)
    all_features = []
    all_labels = []
    all_device_ids = []

    for dev in range(N_DEVICES):
        mean = rng.uniform(0.3, 0.7, size=N_FEATURES)
        cov_scale = rng.uniform(0.02, 0.06, size=N_FEATURES)
        normal = rng.normal(mean, cov_scale, size=(T_STEPS_PER_DEVICE, N_FEATURES))
        labels = np.zeros(T_STEPS_PER_DEVICE, dtype=int)

        anomaly_mask = rng.random(T_STEPS_PER_DEVICE) < ANOMALY_FRACTION
        # Anomalies: shifted mean + higher variance, simulating a real
        # behavioral deviation rather than pure noise.
        shift = rng.uniform(0.3, 0.6, size=N_FEATURES) * rng.choice([-1, 1], size=N_FEATURES)
        normal[anomaly_mask] += shift
        normal[anomaly_mask] += rng.normal(0, 0.1, size=(anomaly_mask.sum(), N_FEATURES))
        labels[anomaly_mask] = 1

        all_features.append(np.clip(normal, 0, 1))
        all_labels.append(labels)
        all_device_ids.append(np.full(T_STEPS_PER_DEVICE, dev))

    return (np.concatenate(all_features), np.concatenate(all_labels),
            np.concatenate(all_device_ids))


# ---------------------------------------------------------------------
# Overlapping sliding window (Section IV-E1).
# ---------------------------------------------------------------------
WINDOW_LEN = 10
STRIDE = 3  # stride < window length, giving genuine overlap between windows


def make_overlapping_windows(features, labels, device_ids):
    """Builds overlapping windows per device (never spanning a device
    boundary), with stride less than the window length."""
    X_windows = []
    y_windows = []  # a window is labeled anomalous if any timestep in it is
    for dev in np.unique(device_ids):
        mask = device_ids == dev
        feat = features[mask]
        lab = labels[mask]
        n = len(feat)
        for start in range(0, n - WINDOW_LEN + 1, STRIDE):
            X_windows.append(feat[start:start + WINDOW_LEN])
            y_windows.append(int(lab[start:start + WINDOW_LEN].max()))
    return np.array(X_windows), np.array(y_windows)


# ---------------------------------------------------------------------
# LSTM autoencoder -- real PyTorch model, real backprop training.
# ---------------------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=32, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, n_features, num_layers, batch_first=True)

    def forward(self, x):
        enc_out, (h, c) = self.encoder(x)
        # feed the encoder's hidden state forward for each timestep
        dec_in = enc_out
        dec_out, _ = self.decoder(dec_in)
        return dec_out


def train_lstm_autoencoder(X_train, epochs=15, batch_size=64, lr=1e-3):
    model = LSTMAutoencoder(n_features=X_train.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    n = X_tensor.shape[0]

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = X_tensor[idx]
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        avg_loss = total_loss / n
        print(f"    epoch {epoch+1}/{epochs}  reconstruction MSE = {avg_loss:.5f}")
    return model


def lstm_reconstruction_error(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        recon = model(X_tensor)
        err = torch.mean((recon - X_tensor) ** 2, dim=(1, 2)).numpy()
    return err


def minmax_scale(train_vals, test_vals):
    lo, hi = train_vals.min(), train_vals.max()
    rng = max(hi - lo, 1e-9)
    return np.clip((test_vals - lo) / rng, 0, 1)


def main():
    print("=" * 72)
    print("BADM METHODOLOGY FEASIBILITY TEST (synthetic data)")
    print("=" * 72)

    t0 = time.time()
    features, labels, device_ids = generate_synthetic_data()
    print(f"Generated {len(features)} timesteps across {N_DEVICES} devices "
          f"({labels.mean()*100:.2f}% anomalous)")

    X_windows, y_windows = make_overlapping_windows(features, labels, device_ids)
    print(f"Built {len(X_windows)} OVERLAPPING windows "
          f"(window={WINDOW_LEN}, stride={STRIDE}, overlap={WINDOW_LEN-STRIDE} steps)")

    # 80/20 split, train ONLY on windows with no anomalous timestep
    # (per the manuscript: "trained on the normal-traffic partition").
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X_windows))
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    normal_train_idx = train_idx[y_windows[train_idx] == 0]
    X_train = X_windows[normal_train_idx]
    X_test = X_windows[test_idx]
    y_test = y_windows[test_idx]
    print(f"Train (normal only): {len(X_train)} windows   Test: {len(X_test)} windows "
          f"({y_test.mean()*100:.2f}% anomalous)")

    print("\nTraining LSTM autoencoder (real PyTorch backprop)...")
    t1 = time.time()
    model = train_lstm_autoencoder(X_train, epochs=15)
    t2 = time.time()
    print(f"LSTM training time: {t2-t1:.1f}s on 1 CPU core")

    lstm_train_err = lstm_reconstruction_error(model, X_train)
    lstm_test_err = lstm_reconstruction_error(model, X_test)

    print("\nTraining Isolation Forest...")
    t3 = time.time()
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=SEED)
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    iso.fit(X_train_flat)
    t4 = time.time()
    print(f"Isolation Forest training time: {t4-t3:.1f}s")

    iso_train_score = -iso.decision_function(X_train_flat)  # higher = more anomalous
    iso_test_score = -iso.decision_function(X_test_flat)

    # Min-max scale both scores using TRAIN statistics only, per the
    # manuscript's stated normalization procedure.
    lstm_test_scaled = minmax_scale(lstm_train_err, lstm_test_err)
    iso_test_scaled = minmax_scale(iso_train_score, iso_test_score)

    ALPHA = 0.5  # fusion weight, per Eq. (equal weighting of IF and LSTM)
    fused_test = ALPHA * iso_test_scaled + (1 - ALPHA) * lstm_test_scaled

    lstm_train_scaled = minmax_scale(lstm_train_err, lstm_train_err)
    iso_train_scaled = minmax_scale(iso_train_score, iso_train_score)
    fused_train = ALPHA * iso_train_scaled + (1 - ALPHA) * lstm_train_scaled

    # ---------------------------------------------------------------
    # Anomaly threshold: a percentile of the fused score, consistent
    # with the 0.01 contamination prior, computed only from the normal
    # training partition and never from the test labels.
    # ---------------------------------------------------------------
    threshold = np.percentile(fused_train, 99)  # 0.01 contamination -> 99th percentile
    predictions = (fused_test >= threshold).astype(int)

    tp = int(((predictions == 1) & (y_test == 1)).sum())
    fp = int(((predictions == 1) & (y_test == 0)).sum())
    tn = int(((predictions == 0) & (y_test == 0)).sum())
    fn = int(((predictions == 0) & (y_test == 1)).sum())

    dr = tp / max(tp + fn, 1) * 100
    fpr = fp / max(fp + tn, 1) * 100
    precision = tp / max(tp + fp, 1) * 100
    f1 = 2 * precision * dr / max(precision + dr, 1e-9)

    print("\n" + "=" * 72)
    print("RESULTS (synthetic data - not for the paper, just checking the pipeline works)")
    print("=" * 72)
    print(f"  Threshold (99th percentile of fused score, normal training data): {threshold:.4f}")
    print(f"  Detection Rate (Recall): {dr:.1f}%")
    print(f"  False Positive Rate:     {fpr:.1f}%")
    print(f"  Precision:               {precision:.1f}%")
    print(f"  F1-score:                {f1:.1f}%")

    total_time = time.time() - t0
    print(f"\nTotal pipeline time: {total_time:.1f}s "
          f"({len(X_windows)} windows, {N_DEVICES} devices, 1 CPU core)")
    print(f"Extrapolated time for a dataset 100x this size: ~{total_time*100/60:.1f} minutes")


if __name__ == '__main__':
    main()
