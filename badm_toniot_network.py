#!/usr/bin/env python3
"""
badm_toniot_network.py

BADM evaluation against TON_IoT's network-flow data. Takes a CSV path
and optionally a seed as command-line args:

    python3 badm_toniot_network.py Network_dataset_1.csv [seed]

TON_IoT's per-device physical sensor files (Processed_IoT_dataset) are
not used here because none of that dataset's attack types (backdoor,
ddos, injection, password, ransomware, xss) alter the physical reading
itself; a DDoS against a fridge's network interface does not change the
temperature it reports, so a physical-sensor feature set carries no
signal for these attack types. The network-flow captures are used
instead, since these attacks leave a footprint at the connection level.

Uses the same evaluation approach as the UNSW-NB15 script: overlapping
windows built per source IP, an LSTM autoencoder, an Isolation Forest,
and a robust (median/MAD) anomaly threshold computed from the
normal-traffic training partition.
"""

import gc
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

import sys
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else ("/home/claude/ton_iot_network/TON_IoT/TON_IoT datasets/"
             "Processed_datasets/Processed_Network_dataset/Network_dataset_1.csv")
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
np.random.seed(SEED)
torch.manual_seed(SEED)

WINDOW_LEN = 10
STRIDE = 3
MIN_ROWS_PER_DEVICE = WINDOW_LEN * 3
MAX_WINDOWS = 300000  # memory cap for this box, see README

NUMERIC_COLS = ['duration', 'src_bytes', 'dst_bytes', 'missed_bytes', 'src_pkts',
                 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes', 'dns_qclass', 'dns_qtype',
                 'dns_rcode', 'http_trans_depth', 'http_request_body_len',
                 'http_response_body_len', 'http_status_code']
FREQ_COLS = ['proto', 'service', 'conn_state', 'dns_query', 'dns_AA', 'dns_RD', 'dns_RA',
              'dns_rejected', 'ssl_version', 'ssl_cipher', 'ssl_resumed', 'ssl_established',
              'http_method', 'http_version', 'weird_name', 'weird_notice']


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
    """Median/MAD Z-score, used in place of mean/standard-deviation or
    percentile-based scaling. For some attack types (port scanning, for
    example), the normal training data's own 99th percentile can exceed
    the maximum reconstruction error across the entire attack class,
    since a small number of legitimately atypical normal events give the
    normal distribution a heavier tail than the attack cluster; this
    defeats any threshold built from a percentile of normal data
    regardless of prior scaling. The median absolute deviation has a 50%
    breakdown point, so a small number of outliers in the normal data
    cannot shift it the way they would a percentile or standard
    deviation. The 0.6745 constant and the 3.5 cutoff are the standard
    values for this transform (Iglewicz and Hoaglin, 1993).
    """
    median = np.median(train_vals)
    mad = np.median(np.abs(train_vals - median))
    mad = max(mad, 1e-9)
    return 0.6745 * (test_vals - median) / mad


ROBUST_Z_CUTOFF = 3.5


def main():
    t_start = time.time()
    print(f"loading {DATA_PATH.split('/')[-1]}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    df = df.sort_values(['src_ip', 'ts']).reset_index(drop=True)

    for col in FREQ_COLS:
        freq = df[col].astype(str).value_counts(normalize=True)
        df[col + '_freq'] = df[col].astype(str).map(freq).fillna(0.0)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # byte/packet counts here are wildly heavy-tailed (src_bytes goes up
    # to 1.3e8), so log1p them before anything gets standardized or the
    # handful of huge values will swamp the mean/std for everyone else
    HEAVY_TAILED_COLS = ['duration', 'src_bytes', 'dst_bytes', 'src_pkts',
                          'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes']
    for col in HEAVY_TAILED_COLS:
        df[col] = np.log1p(df[col].clip(lower=0))

    feature_cols = NUMERIC_COLS + [c + '_freq' for c in FREQ_COLS]
    print(f"feature count: {len(feature_cols)}")

    features_arr = df[feature_cols].values.astype(np.float32)
    labels_arr = df['label'].values.astype(np.int64)
    ip_arr = df['src_ip'].values
    type_arr = df['type'].astype(str).values
    ts_arr = pd.to_numeric(df['ts'], errors='coerce').fillna(0.0).values.astype(np.float64)
    dst_ip_arr = df['dst_ip'].values
    dns_query_arr = df['dns_query'].astype(str).values
    dst_port_arr = df['dst_port'].values
    del df
    gc.collect()

    # A per-flow feature vector alone does not capture connection-pattern
    # signals such as destination diversity or connection rate, which are
    # informative for distinguishing scanning-type traffic from normal
    # traffic. These three features are computed once per window and
    # copied across every timestep in it.
    X_list, y_list, type_list = [], [], []
    for ip in pd.unique(ip_arr):
        mask = ip_arr == ip
        idx = np.where(mask)[0]
        n = len(idx)
        if n < MIN_ROWS_PER_DEVICE:
            continue
        feat = features_arr[idx]
        lab = labels_arr[idx]
        ts = ts_arr[idx]
        dip = dst_ip_arr[idx]
        dpt = dst_port_arr[idx]
        typ = type_arr[idx]
        dnsq = dns_query_arr[idx]
        for start in range(0, n - WINDOW_LEN + 1, STRIDE):
            end = start + WINDOW_LEN
            window_feat = feat[start:end]
            window_ts = ts[start:end]
            window_dip = dip[start:end]
            window_dpt = dpt[start:end]

            unique_dst_ips = len(set(window_dip))
            unique_dst_ports = len(set(window_dpt))
            duration = max(window_ts.max() - window_ts.min(), 1e-6)
            flow_rate = WINDOW_LEN / duration

            # also grab the max byte count seen anywhere in the window,
            # not just the per-timestep values - a single huge DDoS
            # burst hiding among 9 normal-looking flows gets averaged
            # away otherwise, since reconstruction error works on the
            # whole window
            src_bytes_col_idx = NUMERIC_COLS.index('src_bytes')
            dst_bytes_col_idx = NUMERIC_COLS.index('dst_bytes')
            max_src_bytes = window_feat[:, src_bytes_col_idx].max()
            max_dst_bytes = window_feat[:, dst_bytes_col_idx].max()

            engineered = np.array([unique_dst_ips, unique_dst_ports,
                                     np.log1p(flow_rate), max_src_bytes,
                                     max_dst_bytes], dtype=np.float32)
            engineered_broadcast = np.tile(engineered, (WINDOW_LEN, 1))
            window_with_engineered = np.concatenate([window_feat, engineered_broadcast], axis=1)

            X_list.append(window_with_engineered)
            y_list.append(int(lab[start:end].max()))
            types_in_window = [t for t in typ[start:end] if t != 'normal']
            type_list.append(types_in_window[0] if types_in_window else 'normal')

    X_windows = np.array(X_list, dtype=np.float32)
    y_windows = np.array(y_list, dtype=np.int64)
    type_windows = np.array(type_list)
    del X_list, y_list, type_list, features_arr, labels_arr, ip_arr, type_arr
    gc.collect()
    print(f"built {len(X_windows)} overlapping windows, {y_windows.mean()*100:.2f}% anomalous")

    if len(X_windows) > MAX_WINDOWS:
        keep = np.random.RandomState(SEED).choice(len(X_windows), MAX_WINDOWS, replace=False)
        X_windows, y_windows, type_windows = X_windows[keep], y_windows[keep], type_windows[keep]
        print(f"subsampled down to {MAX_WINDOWS} windows to fit in memory "
              f"({y_windows.mean()*100:.2f}% anomalous after subsampling)")

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(X_windows))
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]
    normal_train_idx = train_idx[y_windows[train_idx] == 0]
    type_test = type_windows[test_idx]

    X_train_raw = X_windows[normal_train_idx]
    X_test_raw = X_windows[test_idx]
    y_test = y_windows[test_idx]
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

    # Sanity check prior to training: confirms whether the attack class
    # sits further from normal in feature-vector norm, since a class that
    # instead sits closer to zero than normal traffic (as observed for
    # scanning) inverts the assumption underlying reconstruction-error
    # detection.
    test_norms = np.linalg.norm(X_test.reshape(len(X_test), -1), axis=1)
    print(f"  feature-vector norm by label:")
    for lab in [0, 1]:
        m = y_test == lab
        if m.sum() > 0:
            print(f"    label={lab} (n={m.sum()}): mean_norm={test_norms[m].mean():.3f}  "
                  f"std={test_norms[m].std():.3f}")

    print("\ntraining LSTM autoencoder...")
    t1 = time.time()
    model = train_lstm_autoencoder(X_train, epochs=10)
    print(f"training time: {time.time()-t1:.1f}s")

    lstm_train_err = lstm_reconstruction_error(model, X_train)
    lstm_test_err = lstm_reconstruction_error(model, X_test)

    print(f"  LSTM reconstruction error by label:")
    for lab in [0, 1]:
        m = y_test == lab
        if m.sum() > 0:
            vals = lstm_test_err[m]
            print(f"    label={lab} (n={m.sum()}): mean={vals.mean():.5f}  "
                  f"median={np.median(vals):.5f}  p90={np.percentile(vals,90):.5f}  "
                  f"p99={np.percentile(vals,99):.5f}  max={vals.max():.5f}")

    print("\ntraining Isolation Forest...")
    t3 = time.time()
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=SEED, n_jobs=1)
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    iso.fit(X_train_flat)
    print(f"training time: {time.time()-t3:.1f}s")

    iso_train_score = -iso.decision_function(X_train_flat)
    iso_test_score = -iso.decision_function(X_test_flat)

    print(f"  Isolation Forest score by label:")
    for lab in [0, 1]:
        m = y_test == lab
        if m.sum() > 0:
            print(f"    label={lab} (n={m.sum()}): mean={iso_test_score[m].mean():.5f}  "
                  f"std={iso_test_score[m].std():.5f}")

    lstm_test_z = robust_zscore(lstm_train_err, lstm_test_err)
    iso_test_z = robust_zscore(iso_train_score, iso_test_score)
    lstm_train_z = robust_zscore(lstm_train_err, lstm_train_err)
    iso_train_z = robust_zscore(iso_train_score, iso_train_score)

    ALPHA = 0.5
    fused_test = ALPHA * iso_test_z + (1 - ALPHA) * lstm_test_z
    fused_train = ALPHA * iso_train_z + (1 - ALPHA) * lstm_train_z

    threshold = ROBUST_Z_CUTOFF
    predictions = (fused_test >= threshold).astype(int)

    print(f"  threshold = {threshold:.5f}, fused score by label:")
    for lab in [0, 1]:
        m = y_test == lab
        if m.sum() > 0:
            vals = fused_test[m]
            print(f"    label={lab}: median={np.median(vals):.5f}  p10={np.percentile(vals,10):.5f}  "
                  f"p90={np.percentile(vals,90):.5f}  frac_above={(vals>=threshold).mean()*100:.1f}%")

    tp = int(((predictions == 1) & (y_test == 1)).sum())
    fp = int(((predictions == 1) & (y_test == 0)).sum())
    tn = int(((predictions == 0) & (y_test == 0)).sum())
    fn = int(((predictions == 0) & (y_test == 1)).sum())

    dr = tp / max(tp + fn, 1) * 100
    fpr = fp / max(fp + tn, 1) * 100
    precision = tp / max(tp + fp, 1) * 100
    f1 = 2 * precision * dr / max(precision + dr, 1e-9)

    print("\n" + "=" * 72)
    print("results")
    print("=" * 72)
    print(f"  detection rate: {dr:.1f}%   fpr: {fpr:.1f}%   "
          f"precision: {precision:.1f}%   f1: {f1:.1f}%")
    print(f"  tp={tp}  fp={fp}  tn={tn}  fn={fn}")

    print("\n" + "-" * 72)
    print("by attack type (excluding normal)")
    print("-" * 72)
    for atype in sorted(pd.unique(type_test)):
        if atype == 'normal':
            continue
        m = type_test == atype
        if m.sum() < 5:
            continue
        yt = y_test[m]
        pr = predictions[m]
        tp_t = int(((pr == 1) & (yt == 1)).sum())
        fn_t = int(((pr == 0) & (yt == 1)).sum())
        dr_t = tp_t / max(tp_t + fn_t, 1) * 100
        print(f"  {atype:15s} DR={dr_t:6.1f}%  (n={m.sum()})")

    print(f"\ntotal time: {time.time()-t_start:.1f}s")

    out_name = ('badm_toniot_network_results_' + DATA_PATH.split('_')[-1].replace('.csv','')
                + f'_seed{SEED}.txt')
    with open(out_name, 'w') as f:
        f.write(f"detection_rate={dr:.2f}\nfpr={fpr:.2f}\nprecision={precision:.2f}\nf1={f1:.2f}\n")
        f.write(f"tp={tp}\nfp={fp}\ntn={tn}\nfn={fn}\nseed={SEED}\n")
    print(f"results written to {out_name}")


if __name__ == '__main__':
    main()
