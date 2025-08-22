import argparse
from typing import List, Tuple
import numpy as np

from sdr_analysis_tools import (
    read_2ch_iq,
    detect_signal_bursts_v2,
    pair_bursts,
)


def compute_phase_difference(
    s1: np.ndarray,
    s2: np.ndarray,
    max_lag_samples: int = 0,
) -> Tuple[float, int, complex]:
    """
    Compute phase difference between s1 and s2 as the angle of cross-correlation.
    If max_lag_samples > 0, search within [-max_lag_samples, +max_lag_samples] and
    return the phase at the lag with maximum correlation magnitude.
    Returns (phase_deg, best_lag, corr_value_at_best_lag).
    """
    assert len(s1) == len(s2)
    if len(s1) == 0:
        return float("nan"), 0, 0j

    def corr_at_lag(lag: int) -> complex:
        if lag == 0:
            return np.sum(s1 * np.conj(s2))
        elif lag > 0:
            return np.sum(s1[lag:] * np.conj(s2[:-lag]))
        else:  # lag < 0
            L = -lag
            return np.sum(s1[:-L] * np.conj(s2[L:]))

    if max_lag_samples <= 0:
        c0 = corr_at_lag(0)
        return float(np.rad2deg(np.angle(c0))), 0, c0

    best_lag = 0
    best_val = 0j
    best_mag = -1.0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        c = corr_at_lag(lag)
        m = abs(c)
        if m > best_mag:
            best_mag = m
            best_val = c
            best_lag = lag
    return float(np.rad2deg(np.angle(best_val))), best_lag, best_val


def main():
    ap = argparse.ArgumentParser(description="Inspect raw time-domain and inter-channel phase")
    ap.add_argument("--filename", default="2ch_iq_data.bin")
    ap.add_argument("--sample_rate", type=float, default=4e6)

    # Detection (per-channel) parameters for very short bursts (~0.2 ms)
    ap.add_argument("--threshold_db", type=float, default=8.0)
    ap.add_argument("--hysteresis_db", type=float, default=2.0)
    ap.add_argument("--smoothing_ms", type=float, default=0.02)
    ap.add_argument("--min_burst_us", type=float, default=20.0)
    ap.add_argument("--merge_gap_us", type=float, default=10.0)
    ap.add_argument("--pre_guard_us", type=float, default=0.0)
    ap.add_argument("--post_guard_us", type=float, default=0.0)
    ap.add_argument("--noise_percentile", type=float, default=20.0)
    ap.add_argument("--edge_refine_fraction", type=float, default=0.25)
    ap.add_argument("--edge_search_ms", type=float, default=0.05)

    # Strong-region selection inside overlap (no extra smoothing)
    ap.add_argument("--strong_db_over_noise", type=float, default=10.0, help="Threshold above overlap noise (dB)")
    ap.add_argument("--overlap_noise_percentile", type=float, default=20.0)
    ap.add_argument("--min_strong_samples", type=int, default=64)

    # Correlation
    ap.add_argument("--max_lag_samples", type=int, default=1, help="Search range for correlation lag")

    # Output control
    ap.add_argument("--max_pairs", type=int, default=5)
    ap.add_argument("--print_samples", type=int, default=128, help="How many raw samples to print per channel")
    args = ap.parse_args()

    ch0, ch1 = read_2ch_iq(args.filename)
    if ch0 is None:
        print(f"Error: cannot read IQ file '{args.filename}'")
        return

    # Detect per-channel bursts (short smoothing suitable for ~0.2 ms bursts)
    bursts0, _ = detect_signal_bursts_v2(
        ch0,
        sample_rate=args.sample_rate,
        threshold_db=args.threshold_db,
        hysteresis_db=args.hysteresis_db,
        smoothing_ms=args.smoothing_ms,
        min_burst_us=args.min_burst_us,
        merge_gap_us=args.merge_gap_us,
        pre_guard_us=args.pre_guard_us,
        post_guard_us=args.post_guard_us,
        noise_percentile=args.noise_percentile,
        edge_refine_fraction=args.edge_refine_fraction,
        edge_search_ms=args.edge_search_ms,
    )
    bursts1, _ = detect_signal_bursts_v2(
        ch1,
        sample_rate=args.sample_rate,
        threshold_db=args.threshold_db,
        hysteresis_db=args.hysteresis_db,
        smoothing_ms=args.smoothing_ms,
        min_burst_us=args.min_burst_us,
        merge_gap_us=args.merge_gap_us,
        pre_guard_us=args.pre_guard_us,
        post_guard_us=args.post_guard_us,
        noise_percentile=args.noise_percentile,
        edge_refine_fraction=args.edge_refine_fraction,
        edge_search_ms=args.edge_search_ms,
    )

    pairs = pair_bursts(bursts0, bursts1, max_center_diff=int(0.3e-3 * args.sample_rate))
    print(f"Detected pairs: {len(pairs)} (showing up to {args.max_pairs})")
    if not pairs:
        return

    for pi, (b0, b1) in enumerate(pairs[: args.max_pairs]):
        s0, e0 = b0
        s1, e1 = b1
        s = max(s0, s1)
        e = min(e0, e1)
        if e <= s:
            print(f"\nPair #{pi+1}: no overlap after per-channel detection (b0={b0}, b1={b1})")
            continue

        seg0 = ch0[s : e + 1]
        seg1 = ch1[s : e + 1]
        dur_us = (e - s + 1) / args.sample_rate * 1e6

        # Strong-region: instant power thresholding inside overlap, independently on both channels
        p0 = np.abs(seg0) ** 2
        p1 = np.abs(seg1) ** 2
        n0 = np.percentile(p0, args.overlap_noise_percentile)
        n1 = np.percentile(p1, args.overlap_noise_percentile)
        thr0 = n0 * (10.0 ** (args.strong_db_over_noise / 10.0))
        thr1 = n1 * (10.0 ** (args.strong_db_over_noise / 10.0))
        mask = (p0 >= thr0) & (p1 >= thr1)
        strong_idx = np.flatnonzero(mask)

        print("\n" + "=" * 60)
        print(f"Pair #{pi+1}: b0={b0}, b1={b1}, overlap=[{s},{e}]  len={e-s+1}  dur={dur_us:.2f} us")
        print(f"Strong threshold (overlap): ch0_thr={10*np.log10(thr0+1e-18):.2f} dBFS, ch1_thr={10*np.log10(thr1+1e-18):.2f} dBFS")
        print(f"Strong samples in overlap: {strong_idx.size}")

        if strong_idx.size >= args.min_strong_samples:
            s_str = strong_idx[0]
            e_str = strong_idx[-1]
            s1_str = seg0[s_str : e_str + 1]
            s2_str = seg1[s_str : e_str + 1]
            phi_deg, best_lag, cval = compute_phase_difference(s1_str, s2_str, max_lag_samples=args.max_lag_samples)
            print(f"Phase (deg) based on correlation over strong-overlap: {phi_deg:.2f}  (best_lag={best_lag} samples)")
            print(f"Corr magnitude: {abs(cval):.3e}")

            # Print raw samples around the center of strong region
            center = s_str + (e_str - s_str) // 2
            half = min(args.print_samples // 2, (e - s + 1) // 2)
            left = max(0, center - half)
            right = min(len(seg0), center + half)
            print(f"Raw snippet (centered in strong-overlap), samples [{s+left}:{s+right}):")
            for i in range(left, right):
                v0 = seg0[i]
                v1 = seg1[i]
                print(
                    f"  idx={s+i:8d}  ch0=({v0.real:+.5f},{v0.imag:+.5f})  ch1=({v1.real:+.5f},{v1.imag:+.5f})"
                )
        else:
            print("Strong region too short; printing center of overlap snippet (raw):")
            center = (e - s) // 2
            half = min(args.print_samples // 2, (e - s + 1) // 2)
            left = max(0, center - half)
            right = min(e - s + 1, center + half)
            for i in range(left, right):
                v0 = seg0[i]
                v1 = seg1[i]
                print(
                    f"  idx={s+i:8d}  ch0=({v0.real:+.5f},{v0.imag:+.5f})  ch1=({v1.real:+.5f},{v1.imag:+.5f})"
                )


if __name__ == "__main__":
    main()


