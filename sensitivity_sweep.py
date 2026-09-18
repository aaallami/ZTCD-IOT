#!/usr/bin/env python3
"""
sensitivity_sweep.py

Sweeps every parameter in attack_surface.py that is not directly stated
in the manuscript, individually and at two combined extremes, to test
whether the reported reduction figure depends on a single arbitrarily
chosen value rather than reporting only a point estimate.

Each swept range is fixed in advance based on a defensible bound for
that parameter, independent of the resulting output; every outcome is
reported regardless of direction.
"""

import attack_surface as m


def run_once(trust_shift=0.0, max_hops=6, detection_rate=0.778):
    """One full run with the RNGs reset to a known seed, so each
    scenario is independent of whatever ran before it in this process."""
    m.BYPASS_RNG.seed(m.SEED)
    m.DETECTION_RNG.seed(m.SEED)

    nodes = m.build_nodes(trust_shift=trust_shift)
    baseline_edges = m.build_baseline_edges(nodes)
    ztdc_edges = m.build_ztdc_edges(nodes)

    total_baseline = 0
    total_ztdc = 0
    for threat in m.THREATS:
        b = m.enumerate_paths(nodes, baseline_edges, threat, 'baseline', max_hops=max_hops)
        z = m.enumerate_paths(nodes, ztdc_edges, threat, 'ztdc', max_hops=max_hops,
                               detection_rate=detection_rate)
        total_baseline += b
        total_ztdc += z
    reduction = (1 - total_ztdc / total_baseline) * 100 if total_baseline > 0 else 0.0
    return total_baseline, total_ztdc, reduction


def main():
    nominal = dict(trust_shift=0.0, max_hops=6, detection_rate=0.778)
    results = []

    print("=" * 78)
    print("nominal (reported figure)")
    print("=" * 78)
    b, z, r = run_once(**nominal)
    print(f"  trust_shift=0.0  max_hops=6  detection_rate=0.778  ->  {r:.1f}%  (baseline={b}, ztdc={z})")
    results.append(("NOMINAL", nominal.copy(), r))

    print("\n" + "=" * 78)
    print("trust_shift - uncertainty in device security posture")
    print("range: -0.20 (pessimistic) to +0.20 (optimistic)")
    print("=" * 78)
    for shift in [-0.20, -0.10, 0.0, 0.10, 0.20]:
        params = dict(nominal); params['trust_shift'] = shift
        b, z, r = run_once(**params)
        print(f"  trust_shift={shift:+.2f}  ->  {r:.1f}%  (baseline={b}, ztdc={z})")
        results.append((f"trust_shift={shift:+.2f}", params.copy(), r))

    print("\n" + "=" * 78)
    print("max_hops - maximum attacker path length")
    print("range: 3 to 10")
    print("=" * 78)
    for hops in [3, 4, 5, 6, 8, 10]:
        params = dict(nominal); params['max_hops'] = hops
        b, z, r = run_once(**params)
        print(f"  max_hops={hops:2d}  ->  {r:.1f}%  (baseline={b}, ztdc={z})")
        results.append((f"max_hops={hops}", params.copy(), r))

    print("\n" + "=" * 78)
    print("detection_rate - BADM's detection rate, used here as an input parameter")
    print("range: 0.50 to 0.95")
    print("(this sweep tests the result's sensitivity to this input; it is not")
    print("an evaluation of BADM's actual detection rate, which is measured")
    print("independently in the BADM evaluation scripts)")
    print("=" * 78)
    for rate in [0.50, 0.60, 0.70, 0.778, 0.85, 0.90, 0.95]:
        params = dict(nominal); params['detection_rate'] = rate
        b, z, r = run_once(**params)
        print(f"  detection_rate={rate:.3f}  ->  {r:.1f}%  (baseline={b}, ztdc={z})")
        results.append((f"detection_rate={rate:.3f}", params.copy(), r))

    print("\n" + "=" * 78)
    print("combined extremes")
    print("=" * 78)
    best_case = dict(trust_shift=-0.20, max_hops=10, detection_rate=0.95)
    b, z, r = run_once(**best_case)
    print(f"  best case for ZTDC-IoT (weak devices, long attacker budget, great detector): {best_case}")
    print(f"  ->  {r:.1f}%  (baseline={b}, ztdc={z})")
    results.append(("BEST CASE (max plausible reduction)", best_case, r))

    worst_case = dict(trust_shift=0.20, max_hops=3, detection_rate=0.50)
    b, z, r = run_once(**worst_case)
    print(f"\n  worst case (strong devices, short attacker budget, weak detector): {worst_case}")
    print(f"  ->  {r:.1f}%  (baseline={b}, ztdc={z})")
    results.append(("WORST CASE (min plausible reduction)", worst_case, r))

    print("\n" + "=" * 78)
    print("summary")
    print("=" * 78)
    all_reductions = [r for _, _, r in results]
    NOMINAL_REDUCTION = 69.7  # reported headline figure, Section V-C
    print(f"  range across all {len(results)} scenarios: "
          f"{min(all_reductions):.1f}% to {max(all_reductions):.1f}%")
    print(f"  does the nominal figure ({NOMINAL_REDUCTION}%) fall in that range? "
          f"{'yes' if min(all_reductions) <= NOMINAL_REDUCTION <= max(all_reductions) else 'no'}")

    with open('sensitivity_results.txt', 'w') as f:
        for name, params, r in results:
            f.write(f"{name}\t{params}\t{r:.2f}%\n")
        f.write(f"\nRange: {min(all_reductions):.1f}% to {max(all_reductions):.1f}%\n")
        f.write(f"Includes nominal figure ({NOMINAL_REDUCTION}%): "
                f"{'YES' if min(all_reductions) <= NOMINAL_REDUCTION <= max(all_reductions) else 'NO'}\n")

    print("\nfull results in sensitivity_results.txt")


if __name__ == '__main__':
    main()
