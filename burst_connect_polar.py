import argparse
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

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
    Return (phase_deg, best_lag, corr_complex) using complex correlation over s1,s2.
    If max_lag_samples > 0, search lag in [-max_lag_samples, +max_lag_samples] for max |corr|.
    """
    assert len(s1) == len(s2)
    if len(s1) == 0:
        return float("nan"), 0, 0j

    def corr_at_lag(lag: int) -> complex:
        if lag == 0:
            return np.sum(s1 * np.conj(s2))
        elif lag > 0:
            return np.sum(s1[lag:] * np.conj(s2[:-lag]))
        else:
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
    ap = argparse.ArgumentParser(description="Per-burst amplitude plots and global correlation summary")
    ap.add_argument("--filename", default="2ch_iq_data.bin")
    ap.add_argument("--sample_rate", type=float, default=4e6)

    # Detection tuned for ~0.2 ms short bursts
    ap.add_argument("--threshold_db", type=float, default=14.0)
    ap.add_argument("--hysteresis_db", type=float, default=5.0)
    ap.add_argument("--smoothing_ms", type=float, default=0.02)
    ap.add_argument("--min_burst_us", type=float, default=5.0)
    ap.add_argument("--merge_gap_us", type=float, default=0.02)
    ap.add_argument("--pre_guard_us", type=float, default=0.0)
    ap.add_argument("--post_guard_us", type=float, default=0.0)
    ap.add_argument("--noise_percentile", type=float, default=20.0)
    ap.add_argument("--edge_refine_fraction", type=float, default=0.25)
    ap.add_argument("--edge_search_ms", type=float, default=0.05)

    # Strong region selection inside overlap
    ap.add_argument("--strong_db_over_noise", type=float, default=10.0)
    ap.add_argument("--overlap_noise_percentile", type=float, default=20.0)
    ap.add_argument("--min_strong_samples", type=int, default=64)
    ap.add_argument("--topk_fallback", type=int, default=256, help="If strong region too short, use top-K power samples")

    # Correlation
    ap.add_argument("--max_lag_samples", type=int, default=1)

    # Output
    ap.add_argument("--amplitude_dir", default="amplitude_plots")
    ap.add_argument("--summary_path", default="correlation_summary.png")
    ap.add_argument("--polar_path", default="correlation_polar.png")
    ap.add_argument("--limit_pairs", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    os.makedirs(args.amplitude_dir, exist_ok=True)

    ch0, ch1 = read_2ch_iq(args.filename)
    if ch0 is None:
        print(f"Error: cannot read IQ file '{args.filename}'")
        return

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
    print(f"Paired bursts: {len(pairs)}")
    if not pairs:
        return

    if args.limit_pairs > 0:
        pairs = pairs[: args.limit_pairs]

    corr_mags: List[float] = []
    corr_phases: List[float] = []
    corr_lags: List[int] = []

    for idx, (b0, b1) in enumerate(pairs):
        s0, e0 = b0
        s1, e1 = b1
        s = max(s0, s1)
        e = min(e0, e1)
        if e <= s:
            continue

        seg0 = ch0[s : e + 1]
        seg1 = ch1[s : e + 1]

        # Strong selection in overlap (instant power based)
        p0 = np.abs(seg0) ** 2
        p1 = np.abs(seg1) ** 2
        n0 = np.percentile(p0, args.overlap_noise_percentile)
        n1 = np.percentile(p1, args.overlap_noise_percentile)
        thr0 = n0 * (10.0 ** (args.strong_db_over_noise / 10.0))
        thr1 = n1 * (10.0 ** (args.strong_db_over_noise / 10.0))
        mask = (p0 >= thr0) & (p1 >= thr1)
        strong_idx = np.flatnonzero(mask)

        if strong_idx.size >= args.min_strong_samples:
            s_str = strong_idx[0]
            e_str = strong_idx[-1]
            s1_str = seg0[s_str : e_str + 1]
            s2_str = seg1[s_str : e_str + 1]
        else:
            # Fallback: take top-K power indices common to both channels
            k = min(args.topk_fallback, len(seg0))
            idx0 = np.argsort(p0)[-k:]
            idx1 = np.argsort(p1)[-k:]
            common = np.intersect1d(idx0, idx1)
            if common.size == 0:
                continue
            common.sort()
            s1_str = seg0[common]
            s2_str = seg1[common]

        phi_deg, best_lag, cval = compute_phase_difference(s1_str, s2_str, max_lag_samples=args.max_lag_samples)
        corr_mags.append(abs(cval))
        corr_phases.append(phi_deg)
        corr_lags.append(best_lag)

        # Per-frame amplitude plot (raw |seg0| and |seg1|)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.abs(seg0), label="|Ch0|")
        ax.plot(np.abs(seg1), label="|Ch1|", alpha=0.8)
        ax.set_title(f"Frame #{idx+1}: overlap [{s},{e}]  len={e-s+1}; lag={best_lag}; |corr|={abs(cval):.3e}; phase={phi_deg:.2f}°")
        ax.set_xlabel("Sample index (within overlap)")
        ax.set_ylabel("Amplitude")
        ax.grid(True)
        ax.legend()
        out_path = os.path.join(args.amplitude_dir, f"frame_{idx+1:03d}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    # Global correlation summary figure
    if corr_mags:
        # 1) Legacy summary (keep)
        x = np.arange(1, len(corr_mags) + 1)
        fig1, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axs[0].plot(x, corr_mags, "o-")
        axs[0].set_ylabel("|Correlation|")
        axs[0].grid(True)
        axs[0].set_title("Global Correlation Magnitude across frames")

        axs[1].plot(x, np.mod(np.array(corr_phases) + 180.0, 360.0) - 180.0, "o-")
        axs[1].set_ylabel("Phase (deg)")
        axs[1].set_xlabel("Frame index")
        axs[1].grid(True)

        txt = (
            f"N={len(corr_mags)}\n"
            f"|corr| median={np.median(corr_mags):.3e}\n"
            f"phase mean={np.mean(corr_phases):.2f}°  std={np.std(corr_phases):.2f}°\n"
            f"lags unique={sorted(set(corr_lags))[:8]}"
        )
        axs[0].text(0.01, 0.95, txt, transform=axs[0].transAxes, va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

        fig1.tight_layout()
        fig1.savefig(args.summary_path, dpi=150)
        plt.close(fig1)

        # 2) Polar clustering view: radius = 20*log10(|corr|), theta = phase in [0, 360)
        mags = np.asarray(corr_mags)
        mag_db = 20.0 * np.log10(mags + 1e-18)
        theta_rad = np.deg2rad(np.mod(np.asarray(corr_phases), 360.0))

        fig2 = plt.figure(figsize=(8, 8))
        axp = fig2.add_subplot(111, projection='polar')
        axp.set_theta_zero_location('E')  # 0 deg at +x axis
        axp.set_theta_direction(1)        # CCW
        sc = axp.scatter(theta_rad, mag_db, c=mags, cmap='viridis', s=24, alpha=0.9, edgecolors='none')
        cbar = fig2.colorbar(sc, pad=0.1)
        cbar.set_label('|corr| (linear)')
        axp.set_rlabel_position(135)
        axp.set_title('Correlation clustering (radius=dB, angle=phase)')
        fig2.tight_layout()
        fig2.savefig(args.polar_path, dpi=150)
        plt.close(fig2)


if __name__ == "__main__":
    main()


