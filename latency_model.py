#!/usr/bin/env python3
"""
latency_model.py

Simulates the six-term policy-enforcement latency breakdown from Eq. 5
in Section V-E. Each term is drawn independently from its own
distribution, and the reported median/P99 figures follow directly from
those draws; no post-hoc rescaling is applied to the output.

The Kuzniar et al. citation used for T_rule and T_C covers switch-side
flow-table performance only: its controller-emulation engine issues
updates faster than any tested switch can apply them, so the controller
itself was never the bottleneck in that study. Accordingly, this script
attributes only T_rule to that citation; T_C is modeled as an
independent, uncited estimate.

Terms (from Eq. 5):
  T_PDP      - DTI recompute and policy evaluation at RACE
  T_PDP->C   - PDP to controller, northbound
  T_C        - controller-side rule compilation and conflict check
  T_C->PEP   - controller to PEP, southbound
  T_PEP      - enforcement point processing
  T_rule     - hardware flow-table rule installation

The fast path (proactive, cached decision, approximately 30% of requests
per the manuscript's figure caption) skips T_C and T_rule, since the
rule is already present in the flow table.
"""

import random
import math

SEED = 42
N_SAMPLES = 20000
FAST_PATH_FRACTION = 0.30


def lognormal_params_for_median(median_ms, sigma):
    return math.log(median_ms), sigma

# Fast, low-overhead software steps.
DIST_T_PDP = lognormal_params_for_median(3.0, 0.4)
DIST_T_PEP = lognormal_params_for_median(2.0, 0.4)

# Network hop, based on Kuzniar et al.'s measured 0.1-0.5 ms host-to-switch
# RTT for their testbed. The upper end of that range is used rather than
# the midpoint, since a production network is expected to run somewhat
# slower than a tightly-coupled lab testbed; the value remains within
# their reported range rather than an independent estimate.
DIST_T_NET_HOP = lognormal_params_for_median(0.5, 0.5)

# Controller processing time. Kuzniar et al. does not cover this
# quantity (see module docstring), so this is an independent
# software-processing estimate, not backed by that citation the way
# T_rule is below.
DIST_T_C = lognormal_params_for_median(8.0, 0.5)

# T_rule is modeled as a mixture of two regimes rather than a single
# distribution:
#   typical  - moderate table occupancy, several hundred rules/sec
#     across the three switches tested in Kuzniar et al. (Sections 4.1-4.2)
#   degraded - the paper's finding that a single low-priority catch-all
#     rule reduces throughput to 12-30 rules/sec at high occupancy
#     (Section 4.3), together with the separately reported
#     control/data-plane divergence and stall events of up to 400 ms
#     (Section 3.1), folded in as a heavier tail on this component
# The frequency with which a deployment sits in the degraded regime is
# not measured in the source paper for a general case; the mixture
# weight below is a documented modeling assumption, not a citation.
DIST_T_RULE_TYPICAL = lognormal_params_for_median(3.5, 0.4)
DIST_T_RULE_DEGRADED = lognormal_params_for_median(50.0, 0.9)
DEGRADED_MIXTURE_WEIGHT = 0.10


def sample_t_rule(rng):
    if rng.random() < DEGRADED_MIXTURE_WEIGHT:
        return math.exp(rng.gauss(*DIST_T_RULE_DEGRADED))
    return math.exp(rng.gauss(*DIST_T_RULE_TYPICAL))


def sample_lognormal(params, rng, n=1):
    mu, sigma = params
    return [math.exp(rng.gauss(mu, sigma)) for _ in range(n)]


def simulate(n_samples, seed=SEED):
    rng = random.Random(seed)
    n_fast = int(n_samples * FAST_PATH_FRACTION)
    n_full = n_samples - n_fast

    fast_path_latencies = []
    for _ in range(n_fast):
        t_pdp = sample_lognormal(DIST_T_PDP, rng)[0]
        t_net1 = sample_lognormal(DIST_T_NET_HOP, rng)[0]
        t_net2 = sample_lognormal(DIST_T_NET_HOP, rng)[0]
        t_pep = sample_lognormal(DIST_T_PEP, rng)[0]
        total = t_pdp + t_net1 + t_net2 + t_pep
        fast_path_latencies.append(total)

    full_path_latencies = []
    for _ in range(n_full):
        t_pdp = sample_lognormal(DIST_T_PDP, rng)[0]
        t_net1 = sample_lognormal(DIST_T_NET_HOP, rng)[0]
        t_c = sample_lognormal(DIST_T_C, rng)[0]
        t_net2 = sample_lognormal(DIST_T_NET_HOP, rng)[0]
        t_pep = sample_lognormal(DIST_T_PEP, rng)[0]
        t_rule = sample_t_rule(rng)
        total = t_pdp + t_net1 + t_c + t_net2 + t_pep + t_rule
        full_path_latencies.append(total)

    all_latencies = fast_path_latencies + full_path_latencies
    return fast_path_latencies, full_path_latencies, all_latencies


def percentile(data, p):
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def median(data):
    return percentile(data, 50)


def make_figure(fast, full, all_lat):
    from style import apply_style, clean_axes, footnote, INK, ACCENT, GOOD, WARN, BAD
    import matplotlib.pyplot as plt
    import numpy as np

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for data, label, color in [(all_lat, 'All requests', INK),
                                 (fast, 'Fast path (proactive)', GOOD),
                                 (full, 'Full path (reactive)', ACCENT)]:
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, label=label, color=color, linewidth=1.6)

    med = median(all_lat)
    p99 = percentile(all_lat, 99)
    ax.axvline(med, color=WARN, linestyle='--', linewidth=0.9)
    ax.axvline(p99, color=WARN, linestyle=':', linewidth=0.9)
    ax.axvline(200, color=BAD, linestyle='-', linewidth=1.0, alpha=0.7)
    ax.text(202, 0.06, '200 ms budget', color=BAD, fontsize=8)

    ax.set_xlabel('Policy Enforcement Latency (ms)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Policy Enforcement Latency CDF')
    ax.legend(fontsize=8, loc='lower right')
    clean_axes(ax)
    note = ("T_rule is grounded in Kuzniar et al.'s switch flow-table figures. T_C "
            "is not covered by that paper (switch performance, not controller) and "
            "remains a generic estimate.")
    footnote(fig, note, y=-0.08)
    fig.tight_layout()
    fig.savefig('F5d_latency.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("figure written to F5d_latency.png")


def main():
    fast, full, all_lat = simulate(N_SAMPLES)

    print("=" * 72)
    print("policy enforcement latency simulation")
    print("=" * 72)
    print(f"  fast path (n={len(fast)}, {FAST_PATH_FRACTION*100:.0f}% of requests):")
    print(f"    median = {median(fast):.2f} ms   p99 = {percentile(fast, 99):.2f} ms")
    print(f"  full path (n={len(full)}, {(1-FAST_PATH_FRACTION)*100:.0f}% of requests):")
    print(f"    median = {median(full):.2f} ms   p99 = {percentile(full, 99):.2f} ms")
    print(f"  all requests (n={len(all_lat)}):")
    print(f"    median = {median(all_lat):.2f} ms   p99 = {percentile(all_lat, 99):.2f} ms")

    within_budget = sum(1 for x in all_lat if x <= 200) / len(all_lat) * 100
    print(f"\n  within the 200ms budget: {within_budget:.1f}%")

    with open('latency_results.txt', 'w') as f:
        f.write(f"fast_path_median_ms={median(fast):.2f}\n")
        f.write(f"fast_path_p99_ms={percentile(fast, 99):.2f}\n")
        f.write(f"full_path_median_ms={median(full):.2f}\n")
        f.write(f"full_path_p99_ms={percentile(full, 99):.2f}\n")
        f.write(f"overall_median_ms={median(all_lat):.2f}\n")
        f.write(f"overall_p99_ms={percentile(all_lat, 99):.2f}\n")
        f.write(f"pct_within_200ms_budget={within_budget:.2f}\n")
        f.write(f"seed={SEED}\nn_samples={N_SAMPLES}\nfast_path_fraction={FAST_PATH_FRACTION}\n")
        f.write("note: T_rule is grounded in Kuzniar et al.'s switch flow-table "
                "measurements. T_C is not covered by that paper and is modeled as an "
                "independent estimate.\n")

    print("\nresults written to latency_results.txt")
    make_figure(fast, full, all_lat)


if __name__ == '__main__':
    main()
