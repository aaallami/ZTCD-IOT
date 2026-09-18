#!/usr/bin/env python3
"""
badm_hybrid.py

Combines UNSW-NB15 and TON_IoT network-flow data into a single
evaluation, matching the manuscript's stated hybrid-dataset methodology
(Section V-F), using all four capture files available for each dataset.

The two datasets do not share a schema, so both are mapped to a common
feature set before combining: duration, source/destination bytes,
source/destination packets, protocol frequency, service frequency, plus
three engineered window-level features (destination-IP diversity,
destination-port diversity, connection rate).

The robust (median/MAD) anomaly threshold is calibrated separately per
source dataset before the same fixed cutoff is applied to both, since
UNSW-NB15 and TON_IoT normal traffic sit on different natural scales; a
single global calibration understates the false-positive rate on the
dataset with the tighter baseline. This also reports an ablation
comparing the fused detector against each component (Isolation Forest
only, LSTM autoencoder only) in isolation, computed from the same
trained models used for the fused result.
"""

import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

UNSW_PATHS = [
    "/home/claude/unsw_data/UNSW-NB15/CSV Files/UNSW-NB15_1.csv",
    "/home/claude/unsw_data/UNSW-NB15/CSV Files/UNSW-NB15_2.csv",
    "/home/claude/unsw_data/UNSW-NB15/CSV Files/UNSW-NB15_3.csv",
    "/home/claude/unsw_data/UNSW-NB15/CSV Files/UNSW-NB15_4.csv",
]
TONIOT_PATHS = [
    "/home/claude/ton_iot_network/TON_IoT/TON_IoT datasets/Processed_datasets/"
    "Processed_Network_dataset/Network_dataset_1.csv",
    "/home/claude/ton_iot_network/TON_IoT/TON_IoT datasets/Processed_datasets/"
    "Processed_Network_dataset/Network_dataset_10.csv",
    "/home/claude/ton_iot_network/TON_IoT/TON_IoT datasets/Processed_datasets/"
    "Processed_Network_dataset/Network_dataset_11.csv",
    "/home/claude/ton_iot_network/TON_IoT/TON_IoT datasets/Processed_datasets/"
    "Processed_Network_dataset/Network_dataset_12.csv",
]

UNSW_COLS = ['srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl',
             'sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb',
             'smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt',
             'Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd',
             'is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm',
             'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','Label']

WINDOW_LEN = 10
STRIDE = 3
MIN_ROWS_PER_DEVICE = WINDOW_LEN * 3
MAX_WINDOWS_PER_SOURCE = 60000  # per-file cap; see README for the memory-budget rationale

HEAVY_TAILED = ['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts']
HARMONIZED_NUMERIC = HEAVY_TAILED
HARMONIZED_FREQ = ['proto', 'service']


def load_unsw(path, file_idx):
    print(f"loading {path.split('/')[-1]}...")
    df = pd.read_csv(path, names=UNSW_COLS, low_memory=False)
    df = df.rename(columns={
        'dur': 'duration', 'sbytes': 'src_bytes', 'dbytes': 'dst_bytes',
        'Spkts': 'src_pkts', 'Dpkts': 'dst_pkts', 'srcip': 'src_ip',
        'dstip': 'dst_ip', 'dsport': 'dst_port', 'Label': 'label', 'Stime': 'ts',
    })
    df['dataset_source'] = 'UNSW'
    # source-IP values repeat across UNSW-NB15's separate capture files
    # but represent unrelated capture sessions; file_idx disambiguates
    # them so two different captures are never merged into one device.
    df['src_ip'] = f"f{file_idx}_" + df['src_ip'].astype(str)
    keep = ['src_ip', 'dst_ip', 'dst_port', 'ts', 'proto', 'service',
            'duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts',
            'label', 'dataset_source']
    return df[keep]


def load_toniot(path, file_idx):
    print(f"loading {path.split('/')[-1]}...")
    df = pd.read_csv(path, low_memory=False)
    df['dataset_source'] = 'TON_IoT'
    df['src_ip'] = f"f{file_idx}_" + df['src_ip'].astype(str)
    keep = ['src_ip', 'dst_ip', 'dst_port', 'ts', 'proto', 'service',
            'duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts',
            'label', 'dataset_source']
    return df[keep]


def build_windows_for_source(df, freq_maps, source_name):
    """Builds overlapping windows for one dataset source, following the
    same procedure as the single-dataset scripts. device_key includes
    the source name so a UNSW-NB15 device sequence is never mixed with
    a TON_IoT device sequence."""
    df = df.copy()
    df['device_key'] = source_name + '_' + df['src_ip'].astype(str)
    df = df.sort_values(['device_key', 'ts']).reset_index(drop=True)

    for col in HARMONIZED_FREQ:
        df[col + '_freq'] = df[col].astype(str).map(freq_maps[col]).fillna(0.0)
    for col in HARMONIZED_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    for col in HEAVY_TAILED:
        df[col] = np.log1p(df[col].clip(lower=0))

    feature_cols = HARMONIZED_NUMERIC + [c + '_freq' for c in HARMONIZED_FREQ]
    features_arr = df[feature_cols].values.astype(np.float32)
    labels_arr = df['label'].values.astype(np.int64)
    ts_arr = pd.to_numeric(df['ts'], errors='coerce').fillna(0.0).values.astype(np.float64)
    dst_ip_arr = df['dst_ip'].values
    dst_port_arr = df['dst_port'].values
    device_arr = df['device_key'].values
    del df
    gc.collect()

    X_list, y_list = [], []
    for dev in pd.unique(device_arr):
        mask = device_arr == dev
        idx = np.where(mask)[0]
        n = len(idx)
        if n < MIN_ROWS_PER_DEVICE:
            continue
        feat = features_arr[idx]
        lab = labels_arr[idx]
        ts = ts_arr[idx]
        dip = dst_ip_arr[idx]
        dpt = dst_port_arr[idx]
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
            X_list.append(window_with_eng)
            y_list.append(int(lab[start:end].max()))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    if len(X) > MAX_WINDOWS_PER_SOURCE:
        keep_idx = np.random.RandomState(SEED).choice(len(X), MAX_WINDOWS_PER_SOURCE, replace=False)
        X, y = X[keep_idx], y[keep_idx]
    return X, y


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=32, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, n_features, num_layers, batch_first=True)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        dec_out, _ = self.decoder(enc_out)
        return dec_out


def train_lstm_autoencoder(X_train, epochs=10, batch_size=256, lr=1e-3):
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


def lstm_reconstruction_error(model, X, batch_size=4096):
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
    median = np.median(train_vals)
    mad = max(np.median(np.abs(train_vals - median)), 1e-9)
    return 0.6745 * (test_vals - median) / mad


ROBUST_Z_CUTOFF = 3.5


def main():
    t_start = time.time()

    print("computing joint proto/service frequency maps across all files...")
    freq_parts = []
    for p in UNSW_PATHS:
        freq_parts.append(pd.read_csv(p, names=UNSW_COLS, usecols=['proto', 'service'], low_memory=False))
    for p in TONIOT_PATHS:
        freq_parts.append(pd.read_csv(p, usecols=['proto', 'service'], low_memory=False))
    freq_maps = {}
    for col in HARMONIZED_FREQ:
        combined = pd.concat([d[col].astype(str) for d in freq_parts])
        freq_maps[col] = combined.value_counts(normalize=True).to_dict()
    del freq_parts
    gc.collect()

    X_unsw_parts, y_unsw_parts = [], []
    for i, p in enumerate(UNSW_PATHS, 1):
        df_unsw = load_unsw(p, i)
        X, y = build_windows_for_source(df_unsw, freq_maps, 'UNSW')
        X_unsw_parts.append(X)
        y_unsw_parts.append(y)
        del df_unsw, X, y
        gc.collect()
    X_unsw = np.concatenate(X_unsw_parts, axis=0)
    y_unsw = np.concatenate(y_unsw_parts, axis=0)
    del X_unsw_parts, y_unsw_parts
    gc.collect()
    print(f"UNSW-NB15 ({len(UNSW_PATHS)} files): {len(X_unsw)} windows "
          f"({y_unsw.mean()*100:.2f}% anomalous)")

    X_ton_parts, y_ton_parts = [], []
    for i, p in enumerate(TONIOT_PATHS, 1):
        df_ton = load_toniot(p, i)
        X, y = build_windows_for_source(df_ton, freq_maps, 'TON_IoT')
        X_ton_parts.append(X)
        y_ton_parts.append(y)
        del df_ton, X, y
        gc.collect()
    X_ton = np.concatenate(X_ton_parts, axis=0)
    y_ton = np.concatenate(y_ton_parts, axis=0)
    del X_ton_parts, y_ton_parts
    gc.collect()
    print(f"TON_IoT ({len(TONIOT_PATHS)} files):   {len(X_ton)} windows "
          f"({y_ton.mean()*100:.2f}% anomalous)")

    X_windows = np.concatenate([X_unsw, X_ton], axis=0)
    y_windows = np.concatenate([y_unsw, y_ton], axis=0)
    source_labels = np.array(['UNSW'] * len(X_unsw) + ['TON_IoT'] * len(X_ton))
    del X_unsw, X_ton, y_unsw, y_ton
    gc.collect()
    print(f"\ncombined hybrid dataset: {len(X_windows)} windows "
          f"({y_windows.mean()*100:.2f}% anomalous)")

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X_windows))
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]
    normal_train_idx = train_idx[y_windows[train_idx] == 0]

    X_train_raw = X_windows[normal_train_idx]
    X_test_raw = X_windows[test_idx]
    y_test = y_windows[test_idx]
    source_test = source_labels[test_idx]
    source_train = source_labels[normal_train_idx]
    del X_windows
    gc.collect()

    n_feat = X_train_raw.shape[2]
    flat_train = X_train_raw.reshape(-1, n_feat)
    mean = flat_train.mean(axis=0).astype(np.float32)
    std = flat_train.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    X_train = ((X_train_raw - mean) / std).astype(np.float32)
    X_test = ((X_test_raw - mean) / std).astype(np.float32)
    del X_train_raw, X_test_raw, flat_train
    gc.collect()

    print(f"train (normal only): {len(X_train)}   test: {len(X_test)} "
          f"({y_test.mean()*100:.2f}% anomalous)")

    print("\ntraining LSTM autoencoder on the combined dataset...")
    model = train_lstm_autoencoder(X_train, epochs=10)

    lstm_train_err = lstm_reconstruction_error(model, X_train)
    lstm_test_err = lstm_reconstruction_error(model, X_test)

    print("\ntraining Isolation Forest...")
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=SEED, n_jobs=1)
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    iso.fit(X_train_flat)
    iso_train_score = -iso.decision_function(X_train_flat)
    iso_test_score = -iso.decision_function(X_test_flat)

    # Calibrate median/MAD separately per source dataset, then apply the
    # same fixed cutoff to both (see module docstring). The per-component
    # z-scores are retained (iso_only_test, lstm_only_test) so the
    # ablation below is computed from the same trained models used for
    # the fused result, rather than requiring separate training runs.
    ALPHA = 0.5
    fused_test = np.zeros(len(X_test), dtype=np.float64)
    iso_only_test = np.zeros(len(X_test), dtype=np.float64)
    lstm_only_test = np.zeros(len(X_test), dtype=np.float64)
    for src in np.unique(source_train):
        train_mask = source_train == src
        test_mask = source_test == src
        if train_mask.sum() < 10 or test_mask.sum() == 0:
            continue
        lstm_z = robust_zscore(lstm_train_err[train_mask], lstm_test_err[test_mask])
        iso_z = robust_zscore(iso_train_score[train_mask], iso_test_score[test_mask])
        fused_test[test_mask] = ALPHA * iso_z + (1 - ALPHA) * lstm_z
        iso_only_test[test_mask] = iso_z
        lstm_only_test[test_mask] = lstm_z

    predictions = (fused_test >= ROBUST_Z_CUTOFF).astype(int)
    iso_only_predictions = (iso_only_test >= ROBUST_Z_CUTOFF).astype(int)
    lstm_only_predictions = (lstm_only_test >= ROBUST_Z_CUTOFF).astype(int)

    def compute_metrics_for(preds, mask, label_name):
        yt = y_test[mask]
        pr = preds[mask]
        tp = int(((pr == 1) & (yt == 1)).sum())
        fp = int(((pr == 1) & (yt == 0)).sum())
        tn = int(((pr == 0) & (yt == 0)).sum())
        fn = int(((pr == 0) & (yt == 1)).sum())
        dr = tp / max(tp + fn, 1) * 100
        fpr = fp / max(fp + tn, 1) * 100
        precision = tp / max(tp + fp, 1) * 100
        f1 = 2 * precision * dr / max(precision + dr, 1e-9)
        print(f"  {label_name:12s} DR={dr:6.1f}%  FPR={fpr:6.1f}%  "
              f"Precision={precision:6.1f}%  F1={f1:6.1f}%  (n={mask.sum()})")
        return dict(dr=dr, fpr=fpr, precision=precision, f1=f1, tp=tp, fp=fp, tn=tn, fn=fn)

    def compute_metrics(mask, label_name):
        return compute_metrics_for(predictions, mask, label_name)

    print("\n" + "=" * 72)
    print("results - hybrid UNSW-NB15 + TON_IoT")
    print("=" * 72)
    all_mask = np.ones(len(y_test), dtype=bool)
    overall = compute_metrics(all_mask, "OVERALL")
    unsw_only = compute_metrics(source_test == 'UNSW', "UNSW-only")
    ton_only = compute_metrics(source_test == 'TON_IoT', "TON_IoT-only")

    print("\n" + "=" * 72)
    print("ablation - fused vs. each component in isolation (overall)")
    print("=" * 72)
    fused_ablation = compute_metrics_for(predictions, all_mask, "Fused")
    iso_ablation = compute_metrics_for(iso_only_predictions, all_mask, "IF-only")
    lstm_ablation = compute_metrics_for(lstm_only_predictions, all_mask, "LSTM-only")

    print(f"\ntotal time: {time.time()-t_start:.1f}s")

    with open('badm_hybrid_results.txt', 'w') as f:
        for name, r in [('OVERALL', overall), ('UNSW_only', unsw_only), ('TON_IoT_only', ton_only)]:
            f.write(f"{name}: DR={r['dr']:.2f}% FPR={r['fpr']:.2f}% "
                    f"Precision={r['precision']:.2f}% F1={r['f1']:.2f}%\n")
        f.write("\nABLATION (overall, same trained models as fused result):\n")
        for name, r in [('Fused', fused_ablation), ('IF_only', iso_ablation), ('LSTM_only', lstm_ablation)]:
            f.write(f"{name}: DR={r['dr']:.2f}% FPR={r['fpr']:.2f}% "
                    f"Precision={r['precision']:.2f}% F1={r['f1']:.2f}%\n")
        f.write(f"seed={SEED}\ntotal_time_s={time.time()-t_start:.1f}\n")
    print("results written to badm_hybrid_results.txt")


if __name__ == '__main__':
    main()
