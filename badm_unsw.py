#!/usr/bin/env python3
"""
badm_unsw.py

BADM evaluation against the UNSW-NB15 dataset (raw capture file 1,
700,001 flows). Uses the same evaluation approach as the TON_IoT script:
overlapping windows built per source IP and ordered by capture
timestamp, an LSTM autoencoder, an Isolation Forest, and a threshold
computed from the normal-traffic training partition rather than tuned
against the test labels.

Also reports a breakdown by attack category (Generic, Exploits, Fuzzers,
DoS, and so on), since detection performance varies substantially by
attack mechanism, consistent with the per-attack-mechanism variation
observed on TON_IoT.

Optional seed argument: python3 badm_unsw.py [seed]
"""

import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = "/home/claude/unsw_data/UNSW-NB15/CSV Files/UNSW-NB15_1.csv"
COLS = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl',
        'sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb',
        'smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt',
        'Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd',
        'is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm',
        'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','Label']

WINDOW_LEN = 10
STRIDE = 3
MIN_ROWS_PER_DEVICE = WINDOW_LEN * 3


def load_and_preprocess():
    print("loading UNSW-NB15_1.csv...")
    df = pd.read_csv(DATA_PATH, names=COLS, low_memory=False)
    df['attack_cat'] = df['attack_cat'].fillna('Normal').astype(str).str.strip()

    df = df.sort_values(['srcip', 'Stime']).reset_index(drop=True)

    # frequency encoding rather than one-hot - one-hot on proto/state/
    # service blows up into dozens of mostly-empty columns and dilutes
    # the reconstruction loss
    for col in ['proto', 'state', 'service']:
        freq = df[col].value_counts(normalize=True)
        df[col + '_freq'] = df[col].map(freq).fillna(0.0)

    HEAVY_TAILED = ['dur', 'sbytes', 'dbytes', 'Spkts', 'Dpkts']
    for col in HEAVY_TAILED:
        df[col] = np.log1p(pd.to_numeric(df[col], errors='coerce').fillna(0.0).clip(lower=0))

    df_encoded = df.copy()
    drop_cols = ['srcip', 'dstip', 'sport', 'dsport', 'proto', 'state', 'service',
                 'attack_cat', 'Stime', 'Ltime']
    feature_cols = [c for c in df_encoded.columns if c not in drop_cols + ['Label']]

    print(f"feature count: {len(feature_cols)}")
    return df, df_encoded, feature_cols


def make_overlapping_windows_real(df_original, df_encoded, feature_cols):
    """Windows built per source IP, ordered by real timestamp. Same
    destination-diversity / connection-rate features as the TON_IoT
    script get appended, plus we track the dominant attack category per
    window for the breakdown at the end."""
    features_arr = df_encoded[feature_cols].values.astype(np.float32)
    labels_arr = df_original['Label'].values
    srcip_arr = df_original['srcip'].values
    ts_arr = pd.to_numeric(df_original['Stime'], errors='coerce').fillna(0.0).values.astype(np.float64)
    dst_ip_arr = df_original['dstip'].values
    dst_port_arr = df_original['dsport'].values
    cat_arr = df_original['attack_cat'].values

    X_windows, y_windows, cat_windows, device_ids_out = [], [], [], []

    unique_ips = pd.unique(srcip_arr)
    print(f"building windows across {len(unique_ips)} source IPs...")

    for ip in unique_ips:
        mask = srcip_arr == ip
        idx = np.where(mask)[0]
        n = len(idx)
        if n < MIN_ROWS_PER_DEVICE:
            continue
        feat = features_arr[idx]
        lab = labels_arr[idx]
        ts = ts_arr[idx]
        dip = dst_ip_arr[idx]
        dpt = dst_port_arr[idx]
        cat = cat_arr[idx]
        for start in range(0, n - WINDOW_LEN + 1, STRIDE):
            end = start + WINDOW_LEN
            window_feat = feat[start:end]
            window_ts = ts[start:end]
            unique_dst_ips = len(set(dip[start:end]))
            unique_dst_ports = len(set(dpt[start:end]))
            duration = max(window_ts.max() - window_ts.min(), 1e-6)
            flow_rate = WINDOW_LEN / duration
            engineered = np.array([unique_dst_ips, unique_dst_ports,
                                     np.log1p(flow_rate)], dtype=np.float32)
            window_with_eng = np.concatenate(
                [window_feat, np.tile(engineered, (WINDOW_LEN, 1))], axis=1)

            X_windows.append(window_with_eng)
            y_windows.append(int(lab[start:end].max()))
            # if any flow in the window is an attack, label the window
            # with that category; ties just take the first one found
            cats_in_window = cat[start:end]
            attack_cats = [c for c in cats_in_window if c != 'Normal']
            window_cat = attack_cats[0] if attack_cats else 'Normal'
            cat_windows.append(window_cat)
            device_ids_out.append(ip)

    return (np.array(X_windows), np.array(y_windows), np.array(cat_windows),
            np.array(device_ids_out))


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=32, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, n_features, num_layers, batch_first=True)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        dec_out, _ = self.decoder(enc_out)
        return dec_out


def train_lstm_autoencoder(X_train, epochs=10, batch_size=128, lr=1e-3):
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
        print(f"    epoch {epoch+1}/{epochs}  reconstruction MSE = {total_loss/n:.5f}")
    return model


def lstm_reconstruction_error(model, X, batch_size=2048):
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
            recon = model(batch)
            err = torch.mean((recon - batch) ** 2, dim=(1, 2)).numpy()
            errs.append(err)
    return np.concatenate(errs)


def robust_zscore(train_vals, test_vals):
    """Median/MAD z-score - see badm_toniot_network.py for why this
    replaced a plain percentile-of-normal threshold."""
    median = np.median(train_vals)
    mad = max(np.median(np.abs(train_vals - median)), 1e-9)
    return 0.6745 * (test_vals - median) / mad


ROBUST_Z_CUTOFF = 3.5


def main():
    t_start = time.time()
    df_original, df_encoded, feature_cols = load_and_preprocess()

    X_windows, y_windows, cat_windows, device_ids = make_overlapping_windows_real(
        df_original, df_encoded, feature_cols)
    print(f"built {len(X_windows)} overlapping windows "
          f"(window={WINDOW_LEN}, stride={STRIDE}), "
          f"{y_windows.mean()*100:.2f}% anomalous")

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X_windows))
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    normal_train_idx = train_idx[y_windows[train_idx] == 0]
    X_train_raw = X_windows[normal_train_idx]
    X_test_raw = X_windows[test_idx]
    y_test = y_windows[test_idx]
    cat_test = cat_windows[test_idx]

    n_feat = X_train_raw.shape[2]
    flat_train = X_train_raw.reshape(-1, n_feat)
    mean = flat_train.mean(axis=0).astype(np.float32)
    std = flat_train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    X_train = ((X_train_raw - mean) / std).astype(np.float32)
    X_test = ((X_test_raw - mean) / std).astype(np.float32)
    del X_train_raw, X_test_raw, flat_train
    gc.collect()

    print(f"train (normal only): {len(X_train)} windows   "
          f"test: {len(X_test)} windows ({y_test.mean()*100:.2f}% anomalous)")

    print("\ntraining LSTM autoencoder...")
    t1 = time.time()
    model = train_lstm_autoencoder(X_train, epochs=10)
    print(f"training time: {time.time()-t1:.1f}s")

    lstm_train_err = lstm_reconstruction_error(model, X_train)
    lstm_test_err = lstm_reconstruction_error(model, X_test)

    print("\ntraining Isolation Forest...")
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=SEED, n_jobs=1)
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    iso.fit(X_train_flat)
    iso_train_score = -iso.decision_function(X_train_flat)
    iso_test_score = -iso.decision_function(X_test_flat)

    lstm_test_z = robust_zscore(lstm_train_err, lstm_test_err)
    iso_test_z = robust_zscore(iso_train_score, iso_test_score)

    ALPHA = 0.5
    fused_test = ALPHA * iso_test_z + (1 - ALPHA) * lstm_test_z
    predictions = (fused_test >= ROBUST_Z_CUTOFF).astype(int)

    def compute_metrics(mask):
        yt = y_test[mask]
        pr = predictions[mask]
        tp = int(((pr == 1) & (yt == 1)).sum())
        fp = int(((pr == 1) & (yt == 0)).sum())
        tn = int(((pr == 0) & (yt == 0)).sum())
        fn = int(((pr == 0) & (yt == 1)).sum())
        dr = tp / max(tp + fn, 1) * 100
        fpr = fp / max(fp + tn, 1) * 100
        precision = tp / max(tp + fp, 1) * 100
        f1 = 2 * precision * dr / max(precision + dr, 1e-9)
        return dict(dr=dr, fpr=fpr, precision=precision, f1=f1, tp=tp, fp=fp, tn=tn, fn=fn, n=int(mask.sum()))

    print("\n" + "=" * 72)
    print("results")
    print("=" * 72)
    overall = compute_metrics(np.ones(len(y_test), dtype=bool))
    print(f"  OVERALL      DR={overall['dr']:6.1f}%  FPR={overall['fpr']:6.1f}%  "
          f"Precision={overall['precision']:6.1f}%  F1={overall['f1']:6.1f}%  (n={overall['n']})")

    print("\n" + "-" * 72)
    print("by attack category (excluding normal)")
    print("-" * 72)
    results_by_cat = {}
    for cat in sorted(pd.unique(cat_test)):
        if cat == 'Normal':
            continue
        mask = cat_test == cat
        if mask.sum() < 5:
            continue
        r = compute_metrics(mask)
        results_by_cat[cat] = r
        print(f"  {cat:16s} DR={r['dr']:6.1f}%  (n={r['n']}, {r['tp']+r['fn']} of which are this type)")

    total_time = time.time() - t_start
    print(f"\ntotal time: {total_time:.1f}s")

    with open(f'badm_unsw_results_seed{SEED}.txt', 'w') as f:
        f.write(f"OVERALL: DR={overall['dr']:.2f}% FPR={overall['fpr']:.2f}% "
                f"Precision={overall['precision']:.2f}% F1={overall['f1']:.2f}%\n")
        for cat, r in results_by_cat.items():
            f.write(f"{cat}: DR={r['dr']:.2f}% (n_attack={r['tp']+r['fn']})\n")
        f.write(f"seed={SEED}\ntotal_time_s={total_time:.1f}\n")
    print(f"results written to badm_unsw_results_seed{SEED}.txt")


if __name__ == '__main__':
    main()
