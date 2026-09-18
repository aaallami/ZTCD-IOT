#!/usr/bin/env python3
"""
attack_surface.py

Builds the baseline (VLAN) and ZTDC-IoT graphs from the device inventory
in Table 1 and runs the attack-path enumeration described in Section III.

All reported figures are computed directly from the graph model; changing
a constant below changes the resulting numbers accordingly.

Model summary:
  - graph G = (V, E) for the baseline and for ZTDC-IoT
  - a path counts only if it satisfies all four conditions from Section III:
    reachability, authentication bypass, privilege escalation, and
    command execution
  - paths are enumerated with BFS/DFS and deduplicated, consistent with
    the no-double-counting rule stated in Section III

Several parameters below are not stated numerically in the manuscript,
which describes the four conditions conceptually rather than specifying
exact values. Each such parameter is documented with a comment at its
definition; these are the values to verify first before treating a
derived figure as final.
"""

import random
import itertools
import networkx as nx

SEED = 42
random.seed(SEED)

# Device counts taken at the top of each Table 1 range (per the "counts
# drawn at the upper bound" note in V-A).
DEVICE_COUNTS = {
    'sensor': 1200,   # environmental sensors
    'camera': 300,    # IP cameras
    'rfid':   150,    # RFID/biometric access units
    'pdu':    400,    # smart PDUs
    'bms':    100,    # BMS controllers
    'ups':    60,     # networked UPS
    'zigbee': 200,    # vibration/leak sensors
}

# Zone assignment per device class. BMS only ever sits in Critical, per
# the explicit wording in III ("Critical Zone (power systems, BMS
# controllers, fire suppression systems)"). Everything else is spread
# where it plausibly lives.
ZONE_ASSIGNMENT = {
    'sensor': ['Public', 'Operations', 'Restricted'],
    'camera': ['Public', 'Operations', 'Restricted'],
    'rfid':   ['Restricted', 'Critical'],
    'pdu':    ['Restricted', 'Critical'],
    'bms':    ['Critical'],
    'ups':    ['Restricted', 'Critical'],
    'zigbee': ['Operations', 'Restricted'],
}
ZONE_TIER = {'Public': 0, 'Operations': 1, 'Restricted': 2, 'Critical': 3}

# Only these classes can execute a command; sensors and cameras produce
# data only and cannot be the endpoint of a T3/T4-style attack that
# requires control-plane access.
CONTROL_CAPABLE_CLASSES = {'bms', 'pdu', 'ups', 'rfid', 'facilities', 'acs'}

# Identity Confidence and Vulnerability Exposure weights from the DTI
# formula in Section IV (w1=0.30, w2=0.30, w3=0.20, w4=0.20). Only w1/w3
# apply here, since behavioral conformance and contextual trust require
# an observation window not available at the moment of first contact.
DTI_WEIGHTS = {'w1': 0.30, 'w3': 0.20}

# Per-class Identity Confidence Score and Vulnerability Exposure Score,
# each in [0,1]. These are documented assumptions rather than measured
# values, assigned according to the strength of each class's native
# protocol authentication: Modbus and BACnet have effectively none,
# Wiegand has none at the protocol level but benefits from physical
# co-location with the access panel, and so on. The relative ordering
# across classes is preserved from the manuscript's device-class risk
# tiers, expressed here as continuous scores through the DTI formula
# rather than a separate discrete rule.
ICS_BY_CLASS = {
    'sensor': 0.55,
    'camera': 0.65,
    'rfid':   0.60,
    'pdu':    0.60,
    'bms':    0.50,
    'ups':    0.60,
    'zigbee': 0.40,
}
VES_BY_CLASS = {
    'sensor': 0.45,
    'camera': 0.35,
    'rfid':   0.40,
    'pdu':    0.35,
    'bms':    0.45,
    'ups':    0.35,
    'zigbee': 0.55,
}
BYPASS_RNG = random.Random(SEED)

# Infrastructure nodes, using the same roles as the Mininet emulation in
# Section V-G so the two evidence sources remain internally consistent.
INFRA_NODES = {
    'mgmt':       'Operations',
    'vms':        'Operations',
    'acs':        'Restricted',
    'facilities': 'Critical',
}

# Allow-list for ZTDC-IoT, matching the ALLOWLIST used in the network
# emulation of Section V-G so both evaluations test the same policy.
ALLOWLIST = {
    ('sensor', 'mgmt'), ('mgmt', 'sensor'),
    ('camera', 'vms'),  ('vms', 'camera'),
    ('rfid', 'acs'),    ('acs', 'rfid'),
    ('pdu', 'facilities'), ('facilities', 'pdu'),
    ('bms', 'facilities'), ('facilities', 'bms'),
    ('ups', 'facilities'), ('facilities', 'ups'),
    ('zigbee', 'mgmt'), ('mgmt', 'zigbee'),
}

# Threat to target-classes/entry-point mapping, taken from Table 2.
THREATS = {
    'T1: Sensor Spoofing':          {'target_classes': {'sensor'},                    'entry': 'external'},
    'T2: Camera Feed Manipulation': {'target_classes': {'camera'},                    'entry': 'external'},
    'T3: Access Control Bypass':    {'target_classes': {'rfid', 'acs'},               'entry': 'external'},
    'T4: BMS Exploitation':         {'target_classes': {'bms', 'facilities'},         'entry': 'external'},
    'T5: Lateral Movement':         {'target_classes': set(DEVICE_COUNTS) | set(INFRA_NODES), 'entry': 'external'},
    'T6: Insider Threat':           {'target_classes': set(DEVICE_COUNTS) | set(INFRA_NODES), 'entry': 'insider'},
    'T7: Supply Chain Compromise':  {'target_classes': set(DEVICE_COUNTS) | set(INFRA_NODES), 'entry': 'external'},
}


def build_nodes(trust_shift=0.0):
    """Build the node set. trust_shift lets the sensitivity sweep push
    ICS up / VES down (or the reverse) uniformly across the fleet
    without editing the tables above by hand."""
    nodes = {}
    idx = 0
    for cls, count in DEVICE_COUNTS.items():
        zones = ZONE_ASSIGNMENT[cls]
        ics = min(1.0, max(0.0, ICS_BY_CLASS[cls] + trust_shift))
        ves = min(1.0, max(0.0, VES_BY_CLASS[cls] - trust_shift))
        w1, w3 = DTI_WEIGHTS['w1'], DTI_WEIGHTS['w3']
        # Weighted combination of the two DTI terms available at first
        # contact, renormalized over those two weights only.
        first_contact_trust = (w1 * ics + w3 * (1 - ves)) / (w1 + w3)
        bypass_prob = 1 - first_contact_trust
        for i in range(count):
            zone = zones[i % len(zones)]
            nid = f"{cls}_{idx}"
            nodes[nid] = {'class': cls, 'zone': zone, 'bypass_prob': bypass_prob}
            idx += 1
    for cls, zone in INFRA_NODES.items():
        nodes[cls] = {'class': cls, 'zone': zone, 'bypass_prob': 0.0}
    nodes['external'] = {'class': 'external', 'zone': 'Public', 'bypass_prob': 0.0}
    for zone in ZONE_TIER:
        nodes[f'insider_{zone}'] = {'class': 'insider', 'zone': zone, 'bypass_prob': 0.0}
    return nodes


def build_baseline_edges(nodes):
    """Builds a full mesh within each zone, plus NOC-style cross-zone
    monitoring flows, matching the flat-VLAN description in Section V-A."""
    edges = set()
    by_zone = {}
    for nid, attr in nodes.items():
        by_zone.setdefault(attr['zone'], []).append(nid)

    for zone, members in by_zone.items():
        for a, b in itertools.combinations(members, 2):
            edges.add((a, b)); edges.add((b, a))

    # Only the NOC infrastructure boxes (mgmt, vms) get cross-zone reach
    # for monitoring purposes; a generic Operations-zone device does not
    # inherit cross-zone reachability, since that would over-connect the
    # baseline graph relative to a realistic VLAN topology.
    noc_infra = [n for n in ('mgmt', 'vms') if n in nodes]
    for zone in ['Public', 'Restricted', 'Critical']:
        for a in noc_infra:
            for b in by_zone.get(zone, []):
                edges.add((a, b)); edges.add((b, a))

    # external attacker can hit anything sitting in Public
    for b in by_zone.get('Public', []):
        edges.add(('external', b)); edges.add((b, 'external'))

    # insiders are already inside their own zone's mesh, plus they can
    # reach the NOC boxes like any staff member would
    for zone in ZONE_TIER:
        iid = f'insider_{zone}'
        for b in by_zone.get(zone, []) + noc_infra:
            edges.add((iid, b)); edges.add((b, iid))

    return edges


def build_ztdc_edges(nodes):
    """Builds edges under the deny-by-default policy: only the
    allow-listed (source class, destination class) pairs receive an
    edge."""
    edges = set()
    class_index = {}
    for nid, attr in nodes.items():
        class_index.setdefault(attr['class'], []).append(nid)

    for (src_cls, dst_cls) in ALLOWLIST:
        for a in class_index.get(src_cls, []):
            for b in class_index.get(dst_cls, []):
                edges.add((a, b))

    # An external attacker can still reach the Public-zone perimeter;
    # HICMS/DMSE is the mechanism that prevents a direct jump from there
    # into infrastructure nodes.
    public_devices = [n for n, a in nodes.items() if a['zone'] == 'Public']
    for b in public_devices:
        edges.add(('external', b))

    # Insiders remain constrained to the same allow-list from their zone.
    for zone in ZONE_TIER:
        iid = f'insider_{zone}'
        same_zone_devices = [n for n, a in nodes.items() if a['zone'] == zone]
        for b in same_zone_devices:
            edges.add((iid, b))

    return edges


def auth_bypass_ok(nodes, node_id):
    """Tests whether authentication bypass succeeds against this node.
    Not called for the baseline graph, since the baseline performs no
    identity checks and bypass is automatic there; applies to the
    ZTDC-IoT graph only."""
    return BYPASS_RNG.random() < nodes[node_id]['bypass_prob']


def enumerate_paths(nodes, edge_set, threat_name, mode, max_hops=6, detection_rate=0.778):
    """Counts distinct (attacker, target) compromise scenarios for one
    threat on one graph.

    Two stages:
      1) Initial foothold: an external attacker must clear authentication
         bypass against the first device reached; an insider attacker
         already holds valid credentials, so this step does not apply.
      2) Lateral movement from the foothold, governed by the edges
         present in the graph. Privilege escalation is represented
         structurally through the topology itself (the allow-list vs.
         full-mesh difference between the two graphs) rather than as a
         separate probabilistic step.
    """
    spec = THREATS[threat_name]
    entry_kind = spec['entry']
    target_classes = spec['target_classes']

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edge_set)

    if entry_kind == 'external':
        starts = ['external']
    else:
        starts = [f'insider_{z}' for z in ZONE_TIER]

    valid_path_count = 0
    seen_start_target_pairs = set()

    for start in starts:
        if start not in G:
            continue

        footholds = set()
        for nbr in sorted(G.successors(start)):
            if nodes[start]['class'] == 'insider':
                bypass_ok = True
            elif mode == 'baseline':
                bypass_ok = True
            else:
                bypass_ok = auth_bypass_ok(nodes, nbr)
            if bypass_ok:
                footholds.add(nbr)
                if nodes[nbr]['class'] in target_classes:
                    if threat_name in ('T3: Access Control Bypass', 'T4: BMS Exploitation'):
                        if nodes[nbr]['class'] not in CONTROL_CAPABLE_CLASSES:
                            continue
                    key = (start, nbr)
                    if key not in seen_start_target_pairs:
                        seen_start_target_pairs.add(key)
                        valid_path_count += 1

        if not footholds:
            continue

        # sorted() here matters more than it looks - Python randomizes
        # string hashing by default, so without this the iteration
        # order of the footholds set changes between runs, which then
        # changes which node gets which draw from the detection RNG and
        # quietly breaks reproducibility even with a fixed seed. Lost an
        # afternoon to this one.
        for target in _bfs_targets(G, sorted(footholds), target_classes, nodes, max_hops - 1,
                                     detection_gate=(mode == 'ztdc'), detection_rate=detection_rate):
            if threat_name in ('T3: Access Control Bypass', 'T4: BMS Exploitation'):
                if nodes[target]['class'] not in CONTROL_CAPABLE_CLASSES:
                    continue
            key = (start, target)
            if key in seen_start_target_pairs:
                continue
            seen_start_target_pairs.add(key)
            valid_path_count += 1

    return valid_path_count


# BADM's detection rate as reported in Section IV, used here to gate
# whether a lateral pivot under ZTDC-IoT is caught by RACE before it
# completes. Not applied to the initial foothold, since BADM requires an
# observation window before it has a baseline to compare against; every
# subsequent hop is subject to detection.
BADM_DETECTION_RATE = 0.778
DETECTION_RNG = random.Random(SEED)


def _bfs_targets(G, sources, target_classes, nodes, max_hops, detection_gate=False, detection_rate=0.778):
    """Multi-source BFS from a set of already-compromised nodes. When
    detection_gate is set, each hop beyond the foothold must first pass
    a detection check (representing RACE catching the pivot) before it
    is counted as successful."""
    from collections import deque
    visited = {s: 0 for s in sources}
    found_targets = set()
    q = deque(sources)
    while q:
        node = q.popleft()
        hop = visited[node]
        if hop >= max_hops:
            continue
        for nxt in sorted(G.successors(node)):
            if nxt in visited:
                continue
            if detection_gate:
                if DETECTION_RNG.random() < detection_rate:
                    continue
            visited[nxt] = hop + 1
            q.append(nxt)
            if nodes[nxt]['class'] in target_classes and nxt not in found_targets:
                found_targets.add(nxt)
                yield nxt


def make_figures(rows, total_baseline, total_ztdc):
    from style import apply_style, clean_axes, value_label, footnote, BASELINE, GOOD
    import matplotlib.pyplot as plt

    apply_style()
    note = ("T8 (RF jamming) and T9 (trust-input manipulation) are excluded from this "
            "path-count analysis, since they are availability and input-integrity "
            "threats rather than reachability threats; see Section IV-C and the "
            "T8/T9 figure instead.")

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['Baseline\n(VLAN)', 'ZTDC-IoT'], [total_baseline, total_ztdc],
                   color=[BASELINE, GOOD], width=0.55)
    ax.set_ylabel('Total Attack Paths (all threat categories)')
    ax.set_title('Attack Surface Comparison')
    clean_axes(ax)
    for bar, v in zip(bars, [total_baseline, total_ztdc]):
        value_label(ax, bar.get_x() + bar.get_width() / 2, v, f'{v:,}', fontweight='bold')
    footnote(fig, note)
    fig.tight_layout()
    fig.savefig('fig_attack_paths_total.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    threats = [r[0] for r in rows]
    baselines = [r[1] for r in rows]
    ztdcs = [r[2] for r in rows]
    short_labels = [t.split(':')[0] for t in threats]

    x = list(range(len(threats)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([i - width/2 for i in x], baselines, width, label='Baseline (VLAN)', color=BASELINE)
    ax.bar([i + width/2 for i in x], ztdcs, width, label='ZTDC-IoT', color=GOOD)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylabel('Attack Paths')
    ax.set_title('Attack Surface Reduction by Threat Category')
    ax.legend()
    clean_axes(ax)
    footnote(fig, note)
    fig.tight_layout()
    fig.savefig('fig_attack_paths_by_threat.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("\nfigures written: fig_attack_paths_total.png, fig_attack_paths_by_threat.png")


def main():
    print("building nodes...")
    nodes = build_nodes()
    print(f"  total nodes: {len(nodes)}")
    device_nodes = [n for n, a in nodes.items() if a['class'] in DEVICE_COUNTS]
    infra_nodes = [n for n, a in nodes.items() if a['class'] in INFRA_NODES]
    print(f"  endpoint devices: {len(device_nodes)}")
    print(f"  infra nodes: {len(infra_nodes)}")

    print("\nbuilding baseline (VLAN) graph...")
    baseline_edges = build_baseline_edges(nodes)
    print(f"  |E_VLAN| = {len(baseline_edges)}")

    print("\nbuilding ZTDC-IoT graph...")
    ztdc_edges = build_ztdc_edges(nodes)
    print(f"  |E_ZT| = {len(ztdc_edges)}")

    n_possible_directed = len(nodes) * (len(nodes) - 1)
    density_baseline = len(baseline_edges) / n_possible_directed
    density_ztdc = len(ztdc_edges) / n_possible_directed
    print(f"\n  edge density (baseline): {density_baseline:.6f}")
    print(f"  edge density (ZTDC-IoT): {density_ztdc:.6f}")
    print(f"  ZTDC-IoT keeps {100*len(ztdc_edges)/len(baseline_edges):.1f}% of baseline edges")

    print("\n" + "=" * 72)
    print("attack path enumeration")
    print("=" * 72)
    rows = []
    total_baseline = 0
    total_ztdc = 0
    for threat in THREATS:
        b = enumerate_paths(nodes, baseline_edges, threat, 'baseline')
        z = enumerate_paths(nodes, ztdc_edges, threat, 'ztdc')
        red = (1 - z / b) * 100 if b > 0 else 0.0
        rows.append((threat, b, z, red))
        total_baseline += b
        total_ztdc += z
        print(f"  {threat:32s}  baseline={b:6d}  ztdc-iot={z:6d}  reduction={red:5.1f}%")

    total_red = (1 - total_ztdc / total_baseline) * 100 if total_baseline > 0 else 0.0
    print("-" * 72)
    print(f"  {'TOTAL':32s}  baseline={total_baseline:6d}  ztdc-iot={total_ztdc:6d}  reduction={total_red:5.1f}%")

    with open('attack_surface_results.txt', 'w') as f:
        f.write(f"nodes_total={len(nodes)}\n")
        f.write(f"nodes_device={len(device_nodes)}\n")
        f.write(f"nodes_infra={len(infra_nodes)}\n")
        f.write(f"edges_baseline={len(baseline_edges)}\n")
        f.write(f"edges_ztdc={len(ztdc_edges)}\n")
        f.write(f"density_baseline={density_baseline:.6f}\n")
        f.write(f"density_ztdc={density_ztdc:.6f}\n")
        f.write(f"edge_retention_pct={100*len(ztdc_edges)/len(baseline_edges):.2f}\n")
        for threat, b, z, red in rows:
            f.write(f"{threat}: baseline={b} ztdc={z} reduction={red:.1f}%\n")
        f.write(f"TOTAL: baseline={total_baseline} ztdc={total_ztdc} reduction={total_red:.1f}%\n")
        f.write(f"seed={SEED}\n")
        f.write(f"max_hops=6\n")

    print("\nresults written to attack_surface_results.txt")

    import csv
    with open('attack_surface_comparison.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Threat Category', 'Baseline Paths', 'ZTDC-IoT Paths', 'Reduction (%)'])
        for threat, b, z, red in rows:
            writer.writerow([threat, b, z, f"{red:.1f}"])
        writer.writerow(['TOTAL', total_baseline, total_ztdc, f"{total_red:.1f}"])
    print("table written to attack_surface_comparison.csv")

    make_figures(rows, total_baseline, total_ztdc)


if __name__ == '__main__':
    main()
