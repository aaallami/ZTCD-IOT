# ZTDC-IoT reproducibility code

Code used to produce the evaluation numbers reported in the paper.

## Contents

| File | What it does |
|---|---|
| `style.py` | Shared Matplotlib styling (palette, axis cleanup, annotations) used by the figure-generating functions below |
| `attack_surface.py` | Builds the baseline and ZTDC-IoT graphs, counts attack paths under the four conditions in Section III, and writes `fig_attack_paths_total.png` / `fig_attack_paths_by_threat.png` |
| `sensitivity_sweep.py` | Sweeps the attack-surface model's parameters across their plausible ranges |
| `latency_model.py` | Six-term policy-enforcement latency simulation (Eq. 5, Section V-E); writes `F5d_latency.png` |
| `t8_t9_resilience.py` | Tests the corroboration/hysteresis/unknown-state logic from Section IV-C under adversarial input; writes `fig_t8_t9_resilience.png` |
| `badm_synthetic_selfcheck.py` | Runs the BADM pipeline on synthetic data, for checking your environment before using real datasets |
| `badm_unsw.py` | BADM evaluation on UNSW-NB15 |
| `badm_toniot_network.py` | BADM evaluation on TON_IoT network-flow data |
| `badm_hybrid.py` | Combined UNSW-NB15 + TON_IoT evaluation, plus the IF-only / LSTM-only ablation (Table 9) |
| `network_emulation/` | Mininet + Open vSwitch validation with real MQTT, Modbus, and token-authentication implementations (Section V-G); see its own README |

## Setup

```
pip install -r requirements.txt
```

`attack_surface.py`, `sensitivity_sweep.py`, `latency_model.py`, and
`t8_t9_resilience.py` need no external data:

```
python3 attack_surface.py
python3 sensitivity_sweep.py
python3 latency_model.py
python3 t8_t9_resilience.py
```

The BADM scripts additionally need `torch` and `scikit-learn`
(included in requirements.txt) and the datasets described below.

## Datasets

- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
  (uses the raw capture files `UNSW-NB15_1.csv` through `_4.csv`)
- TON_IoT: https://research.unsw.edu.au/projects/toniot-datasets
  (uses the `Processed_Network_dataset` files)

Update the file paths at the top of `badm_unsw.py`, `badm_toniot_network.py`,
and `badm_hybrid.py` to point at your local copies.

```
python3 badm_unsw.py [seed]
python3 badm_toniot_network.py <path_to_csv> [seed]
python3 badm_hybrid.py
```

Seed defaults to 42.

## Methodology notes

**Attack surface** (`attack_surface.py`): builds a real `networkx` graph
from the Table 1 device inventory and enumerates paths under the four
conjunctive conditions (reachability, auth bypass, privilege escalation,
command execution). Device zone assignment, per-class identity and
vulnerability scores, and the max hop bound are documented assumptions
in the script, not measured values - `sensitivity_sweep.py` checks the
result's sensitivity to each of them.

**Latency** (`latency_model.py`): each of the six terms in Eq. 5 is
drawn from an independent distribution. T_rule's distribution is
grounded in the flow-table update rates and control/data-plane
divergence figures reported in Kuzniar et al. (PAM 2015). That paper
covers switch-side performance only, not SDN controller processing, so
T_C remains a generic estimate.

**T8/T9** (`t8_t9_resilience.py`): simulates the DTI-based corroboration
rule (quarantine requires 2 of 3 independent degraded trust components)
and the unknown-state telemetry-loss handling under adversarial input.
Trust-score distributions for legitimate vs. compromised devices are
documented assumptions.

**BADM** (`badm_unsw.py`, `badm_toniot_network.py`, `badm_hybrid.py`):
LSTM autoencoder plus Isolation Forest, fused via a robust median/MAD
Z-score with a fixed cutoff of 3.5 (standard convention, Iglewicz and
Hoaglin 1993). Categorical features (protocol, service, connection state)
are frequency-encoded. Byte and packet counts are log-transformed
before standardization. Each window includes three engineered features
- destination-IP diversity, destination-port diversity, and connection
rate - alongside the per-flow features. The training partition contains
only windows with no anomalous flow; the threshold is computed from
that partition's score distribution and never touches the test labels.
`badm_hybrid.py` calibrates the Z-score baseline separately for each
source dataset before applying the same fixed cutoff to both, since
UNSW-NB15 and TON_IoT traffic sit on different natural scales. It also
reports an ablation (Table 9) comparing the fused detector against
Isolation Forest alone and the LSTM autoencoder alone, computed from
the per-component scores of the same trained models used for the fused
result, rather than requiring three separate training runs.

Only 4 of the TON_IoT `Processed_Network_dataset` files are referenced
here (1, 10, 11, 12); confirm your download contains real data for
these rather than placeholder files before running.

**Network emulation** (`network_emulation/`): validates the segmentation
mechanism against real, unmodified protocol implementations (MQTT via
Mosquitto, Modbus TCP via pymodbus, and a token-authenticated command
service standing in for RFID/BACnet access control) running under
Mininet with a real Open vSwitch switch enforcing OpenFlow rules. See
`network_emulation/README.md` for setup, which requires Mininet, Open
vSwitch, and root privileges.

## Reproducing published numbers

All scripts are seeded (default 42). Runs against the same dataset
files with the same seed reproduce the reported figures exactly for
the deterministic scripts (`attack_surface.py`, `sensitivity_sweep.py`,
`latency_model.py`, `t8_t9_resilience.py`) and within a percentage
point or two for the BADM scripts, which involve stochastic training.
