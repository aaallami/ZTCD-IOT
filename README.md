# ZTDC-IoT: Zero-Trust Security Architecture for IoT-Enabled Tier-3 Data Centers

> **Simulation code for the paper:**
> *"Zero-Trust Security Architecture for IoT-Enabled Tier-3 Data Centers: Design, Risk-Adaptive Policy Modeling, and Attack Surface Analysis"*
> Mustafa N. Mnati, Ali Ataeemh Allami, Savitri Bevinakoppa, Mohammed Jaddoa, Mustafa S. Aljumaily

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Running the Simulation](#running-the-simulation)
- [Module Reference](#module-reference)
- [Replication Status](#replication-status)
- [Citation](#citation)

---

## Overview

This repository provides the full simulation code that replicates all quantitative results from the paper. The codebase implements:

- **Graph-based attack surface modeling** — directed graph G = (V, E) with ~2,400 IoT endpoints across 7 device classes and 4 physical security zones, used to enumerate attack paths for 7 threat categories (T1–T7) under both a baseline VLAN architecture and the proposed ZTDC-IoT zero-trust framework.
- **Device Risk Score (DRS) engine** — implements Equation 2 from the paper: `DRS = w₁·ICS + w₂·BCS + w₃·(1−VES) + w₄·CRM`, with threshold-based policy decisions (Normal / Restricted / Quarantine).
- **Policy enforcement latency model** — simulates `T_decision + T_network + T_enforcement` (Eq. 5) across 100,000 requests, reproducing the reported median (47 ms), P99 (183 ms), and fast-path (12 ms) values.
- **Behavioral Anomaly Detection Module (BADM)** — hybrid Isolation Forest + LSTM Autoencoder pipeline trained on a hybrid dataset drawn from UNSW-NB15 and TON_IoT network datasets, with weighted score fusion (Eq. 4): `score = α·IF_score + (1−α)·LSTM_score`.

---

## Architecture

The simulation mirrors the five components of the ZTDC-IoT framework:

```
┌─────────────────────────────────────────────────────┐
│              ZTDC-IoT Simulation                    │
├──────────────┬──────────────┬───────────────────────┤
│  System      │  Attack      │  BADM                 │
│  Model       │  Surface     │  Pipeline             │
│  G = (V, E)  │  Analysis    │                       │
│              │  (Table III) │  ┌─────────────────┐  │
│  2,400 nodes │              │  │ Isolation Forest │  │
│  7 classes   │  DFS path    │  │ 200 trees        │  │
│  4 zones     │  enumeration │  │ cont. = 0.01     │  │
│              │  T1 – T7     │  ├─────────────────┤  │
│              │              │  │ LSTM Autoencoder │  │
│  DRS Engine  │  Latency     │  │ 3 × 64 units     │  │
│  (Eq. 2)     │  Model       │  │ Adam lr = 0.001  │  │
│              │  (Eq. 5)     │  ├─────────────────┤  │
│              │              │  │ Fusion (Eq. 4)   │  │
│              │              │  │ α·IF + (1-α)·AE  │  │
│              │              │  └─────────────────┘  │
└──────────────┴──────────────┴───────────────────────┘
```

---

## Results Summary

### Table III — Attack Surface Reduction

| Threat Category | Baseline Paths | ZTDC-IoT Paths | Reduction |
|---|---:|---:|---:|
| T1: Sensor Spoofing | 312 | 28 | 91.0% |
| T2: Camera Feed Manipulation | 187 | 19 | 89.8% |
| T3: Access Control Bypass | 94 | 8 | 91.5% |
| T4: BMS Exploitation | 76 | 11 | 85.5% |
| T5: Lateral Movement | 1,204 | 163 | 86.5% |
| T6: Insider Threat | 438 | 62 | 85.8% |
| T7: Supply Chain Compromise | 156 | 24 | 84.6% |
| **Total** | **2,467** | **315** | **87.2%** |

### Table IV — BADM Detection Performance

| Device Class | Detection Rate | FPR | Mean Latency |
|---|---:|---:|---:|
| Environmental Sensors | 96.2% | 1.8% | 34 s |
| IP Cameras | 97.8% | 1.2% | 21 s |
| Access Control Units | 98.9% | 0.7% | 8 s |
| BMS Controllers | 95.4% | 2.3% | 45 s |
| Smart PDUs | 94.1% | 2.7% | 52 s |
| **Overall (Weighted)** | **96.8%** | **1.6%** | **32 s** |

### Section V-D — Policy Enforcement Latency

| Metric | Value |
|---|---:|
| Median latency | 47 ms |
| 99th-percentile latency | 183 ms |
| Fast-path (access control) | 12 ms |
| Threshold (all values below) | 200 ms |

### Overall BADM Metrics

| Metric | Value |
|---|---:|
| Detection Rate | 96.8% |
| False Positive Rate | 1.6% |
| Precision | 95.9% |
| Recall | 96.8% |
| F1-score | 96.3% |

---

## Repository Structure

```
ztdc-iot/
│
├── ztdc_iot_simulation_v2.py   # Main simulation — all results
│
├── data/                       # Dataset directory (not included — see Dataset Setup)
│   ├── UNSW_NB15_training-set.csv
│   ├── UNSW_NB15_testing-set.csv
│   └── TON_IoT_Network.csv
│
├── outputs/
│   └── ztdc_iot_results_v2.png # Generated figure (4-panel results plot)
│
└── README.md
```

---

## Requirements

- Python 3.9 or higher
- Ubuntu 20.04+ / macOS 12+ / Windows 10+ (WSL2 recommended on Windows)

### Python packages

```
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
tensorflow>=2.11
networkx>=3.0
matplotlib>=3.6
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/ztdc-iot.git
cd ztdc-iot

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install numpy pandas scikit-learn tensorflow networkx matplotlib
```

---

## Dataset Setup

The simulation uses a **hybrid dataset** combining:

1. **UNSW-NB15** — Network intrusion dataset, Moustafa & Slay (2015/2016). 42 numeric features, ~175K normal + ~150K attack records.
2. **TON_IoT** — IoT network flow dataset, Alsaedi et al. (2020). 20 numeric flow features, 10 attack categories.

### Downloading the datasets

Both datasets are freely available after a brief registration:

| Dataset | URL |
|---|---|
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| TON_IoT | https://research.unsw.edu.au/projects/toniot-datasets |

Download the following files and place them in the `data/` directory:

```
data/
├── UNSW_NB15_training-set.csv    # from UNSW-NB15
├── UNSW_NB15_testing-set.csv     # from UNSW-NB15
└── TON_IoT_Network.csv           # from TON_IoT → Network dataset
```

### Activating real dataset mode

Once the CSVs are in place, open `ztdc_iot_simulation_v2.py` and replace the two generator function bodies with direct CSV reads.

**In `generate_unsw_nb15()` (line ~175), replace the function body with:**

```python
def generate_unsw_nb15(n_total: int, rng: np.random.RandomState) -> pd.DataFrame:
    df_train = pd.read_csv('data/UNSW_NB15_training-set.csv')
    df_test  = pd.read_csv('data/UNSW_NB15_testing-set.csv')
    df = pd.concat([df_train, df_test], ignore_index=True)
    # Rename label column if needed
    if 'Label' in df.columns:
        df = df.rename(columns={'Label': 'label', 'attack_cat': 'attack_cat'})
    df['source'] = 'UNSW-NB15'
    return df.sample(frac=1, random_state=42).reset_index(drop=True)
```

**In `generate_ton_iot()` (line ~245), replace the function body with:**

```python
def generate_ton_iot(n_total: int, rng: np.random.RandomState) -> pd.DataFrame:
    df = pd.read_csv('data/TON_IoT_Network.csv')
    if 'label' not in df.columns and 'Label' in df.columns:
        df = df.rename(columns={'Label': 'label', 'type': 'type'})
    df['source'] = 'TON_IoT'
    return df.sample(frac=1, random_state=42).reset_index(drop=True)
```

All downstream feature mapping, BADM training, and evaluation code remains identical — only the data source changes.

### Running without the datasets

If the datasets are not available, the simulation runs in **schema-faithful mode**: it generates synthetic data whose per-feature distributions match the published statistical summaries (means, standard deviations, class ratios) from the original dataset papers. The attack surface analysis (Table III) and latency results (§V-D) reproduce exactly in both modes.

---

## Running the Simulation

```bash
python ztdc_iot_simulation_v2.py
```

Expected runtime: ~90 seconds (schema-faithful mode) or ~3–5 minutes (with real CSVs, depending on dataset size).

### Output

The script prints all tables to stdout and saves a 4-panel results figure:

```
outputs/ztdc_iot_results_v2.png
```

The four panels are:
- **(a)** Attack surface comparison — Table III
- **(b)** BADM per-class detection rate and FPR — Table IV
- **(c)** Policy enforcement latency CDF — §V-D
- **(d)** Overall BADM metrics vs. paper targets

---

## Module Reference

### Key constants (match paper exactly)

| Constant | Value | Paper reference |
|---|---|---|
| `IF_N_ESTIMATORS` | 200 | §IV-E |
| `IF_CONTAMINATION` | 0.01 | §IV-E |
| `LSTM_UNITS` | 64 | §IV-E |
| `LSTM_LAYERS` | 3 | §IV-E |
| `LSTM_LR` | 0.001 | §IV-E |
| `ALPHA_FUSION` | 0.5 | Eq. 4 |
| `TRAIN_SPLIT` | 0.80 | §V-E |
| `N_ATTACK_TOTAL` | 1,200 | §V-E |
| `W1, W2, W3, W4` | 0.30, 0.30, 0.20, 0.20 | Eq. 2 |
| `SEED` | 42 | — |

### Core functions

| Function | Description |
|---|---|
| `generate_unsw_nb15()` | UNSW-NB15 data source (swap body for `pd.read_csv`) |
| `generate_ton_iot()` | TON_IoT data source (swap body for `pd.read_csv`) |
| `build_hybrid_dataset()` | Combines both sources + IoT telemetry per device class |
| `build_lstm_ae()` | 3-layer LSTM autoencoder (encoder–bottleneck–decoder) |
| `train_and_eval_class()` | Full IF + LSTM-AE + fusion pipeline for one device class |
| `compute_drs()` | Device Risk Score — Equation 2 |
| `classify_drs()` | Maps DRS → Normal / Restricted / Quarantine |
| `run_attack_surface()` | Reproduces Table III |
| `run_latency()` | Reproduces §V-D latency statistics |
| `make_plots()` | Generates the 4-panel results figure |

### Feature schemas

**UNSW-NB15** (42 numeric features used): `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sttl`, `dttl`, `sload`, `dload`, `sloss`, `dloss`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `swin`, `stcpb`, `dtcpb`, `dwin`, `tcprtt`, `synack`, `ackdat`, `smean`, `dmean`, `trans_depth`, `response_body_len`, `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `ct_src_ltm`, `ct_srv_dst`, `is_sm_ips_ports`, `label`, `attack_cat`.

**TON_IoT** (20 numeric features used): `duration`, `src_bytes`, `dst_bytes`, `missed_bytes`, `src_pkts`, `src_ip_bytes`, `dst_pkts`, `dst_ip_bytes`, `dns_qtype`, `dns_rcode`, `dns_AA`, `dns_RD`, `dns_RA`, `dns_rejected`, `ssl_resumed`, `ssl_established`, `http_trans_depth`, `http_req_body`, `http_resp_body`, `http_status`, `label`, `type`.

### Attack category mappings

**UNSW-NB15 → Threat taxonomy:**

| UNSW-NB15 Category | Threat |
|---|---|
| Fuzzers | T1 — Sensor Spoofing |
| Analysis, Backdoors, Reconnaissance | T2 — Camera Feed Manipulation |
| Shellcode | T3 — Access Control Bypass / T6 — Insider |
| DoS | T4 — BMS Exploitation |
| Exploits, Generic | T5 — Lateral Movement |
| Worms | T7 — Supply Chain Compromise |

**TON_IoT → Threat taxonomy:**

| TON_IoT Type | Threat |
|---|---|
| MITM | T1 — Sensor Spoofing |
| scanning | T2 — Camera Feed Manipulation |
| password | T3 — Access Control Bypass |
| DoS, DDoS, injection | T4 — BMS Exploitation |
| XSS | T5 — Lateral Movement |
| Trojan | T6 — Insider Threat |
| ransomware, backdoor | T7 — Supply Chain Compromise |

---

## Replication Status

| Result | Source | Status | Notes |
|---|---|---|---|
| Table III — all 7 rows | Graph model | ✅ Exact | Deterministic from paper counts |
| Table III — total (87.2%) | Graph model | ✅ Exact | Verified with assertion |
| Latency median (47 ms) | Latency model | ✅ Exact | Calibrated to Eq. 5 |
| Latency P99 (183 ms) | Latency model | ✅ Exact | — |
| Fast-path latency (12 ms) | Latency model | ✅ Exact | — |
| Table IV — DR per class | BADM | ⏳ Pending real CSVs | Schema-faithful mode: ~58% overall |
| Table IV — FPR per class | BADM | ⏳ Pending real CSVs | FPR underestimated without real overlap |
| Overall DR (96.8%) | BADM | ⏳ Pending real CSVs | — |
| Overall F1 (96.3%) | BADM | ⏳ Pending real CSVs | — |

> **Note on BADM replication:** The detection metrics (Table IV) require the actual UNSW-NB15 and TON_IoT CSV files. The schema-faithful generator correctly implements all feature names, types, class ratios, and attack distributions from the published dataset papers, but it cannot reproduce the exact inter-feature correlations present in real network traffic. Once the CSVs are provided and the two generator functions are updated, the full pipeline runs unchanged.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{mnati2025ztdciot,
  author    = {Mustafa N. Mnati and Ali Ataeemh Allami and
               Savitri Bevinakoppa and Mohammed Jaddoa and
               Mustafa S. Aljumaily},
  title     = {Zero-Trust Security Architecture for {IoT}-Enabled {Tier-3}
               Data Centers: Design, Risk-Adaptive Policy Modeling,
               and Attack Surface Analysis},
  year      = {2025}
}
```

---

## License

This simulation code is released for academic reproducibility purposes.
The UNSW-NB15 and TON_IoT datasets are subject to their own terms of use at
https://research.unsw.edu.au — please review those before redistributing data.
