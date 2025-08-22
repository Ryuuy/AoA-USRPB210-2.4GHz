import argparse
import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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


def intersect_subbursts(
    bursts0: List[List[int]],
    bursts1: List[List[int]],
    min_len: int,
) -> List[Tuple[int, int]]:
    """Intersect sub-bursts from both channels; return local index ranges [s,e] within overlap.
    Keeps only intersections with length >= min_len.
    """
    result: List[Tuple[int, int]] = []
    i, j = 0, 0
    bursts0_sorted = sorted(bursts0)
    bursts1_sorted = sorted(bursts1)
    while i < len(bursts0_sorted) and j < len(bursts1_sorted):
        s0, e0 = bursts0_sorted[i]
        s1, e1 = bursts1_sorted[j]
        s = max(s0, s1)
        e = min(e0, e1)
        if e >= s:
            if (e - s + 1) >= min_len:
                result.append((s, e))
        if e0 < e1:
            i += 1
        else:
            j += 1
    return result


def split_overlap_into_subbursts(
    seg0: np.ndarray,
    seg1: np.ndarray,
    sample_rate: float,
    inner_threshold_db: float,
    inner_hysteresis_db: float,
    inner_smoothing_ms: float,
    inner_min_burst_us: float,
    inner_merge_gap_us: float,
    inner_pre_guard_us: float,
    inner_post_guard_us: float,
    inner_noise_percentile: float,
    inner_edge_refine_fraction: float,
    inner_edge_search_ms: float,
    require_intersection: bool,
    min_subburst_samples: int,
) -> List[Tuple[int, int]]:
    """Re-segment the overlapped region with stricter params; return sub-bursts in local indices.
    If require_intersection=True, keeps only intersections present in both channels.
    """
    b0, _ = detect_signal_bursts_v2(
        seg0,
        sample_rate=sample_rate,
        threshold_db=inner_threshold_db,
        hysteresis_db=inner_hysteresis_db,
        smoothing_ms=inner_smoothing_ms,
        min_burst_us=inner_min_burst_us,
        merge_gap_us=inner_merge_gap_us,
        pre_guard_us=inner_pre_guard_us,
        post_guard_us=inner_post_guard_us,
        noise_percentile=inner_noise_percentile,
        edge_refine_fraction=inner_edge_refine_fraction,
        edge_search_ms=inner_edge_search_ms,
    )
    b1, _ = detect_signal_bursts_v2(
        seg1,
        sample_rate=sample_rate,
        threshold_db=inner_threshold_db,
        hysteresis_db=inner_hysteresis_db,
        smoothing_ms=inner_smoothing_ms,
        min_burst_us=inner_min_burst_us,
        merge_gap_us=inner_merge_gap_us,
        pre_guard_us=inner_pre_guard_us,
        post_guard_us=inner_post_guard_us,
        noise_percentile=inner_noise_percentile,
        edge_refine_fraction=inner_edge_refine_fraction,
        edge_search_ms=inner_edge_search_ms,
    )

    if require_intersection:
        subs = intersect_subbursts(b0, b1, min_len=min_subburst_samples)
    else:
        # union and min length filter
        all_b = sorted(b0 + b1)
        subs = [(s, e) for s, e in all_b if (e - s + 1) >= min_subburst_samples]

    return subs


# ---------------- Valley-based forced split -----------------

def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.astype(float)
    win = int(win)
    k = np.ones(win, dtype=float) / float(win)
    return np.convolve(x.astype(float), k, mode="same")


def split_by_valleys(
    seg0: np.ndarray,
    seg1: np.ndarray,
    sample_rate: float,
    valley_smoothing_ms: float,
    valley_db_above_noise: float,
    min_valley_us: float,
    min_subburst_samples: int,
) -> List[Tuple[int, int]]:
    """Force split by valleys using a joint envelope = min(|seg0|^2, |seg1|^2).
    We detect contiguous runs where joint envelope dips near noise for >= min_valley_len
    and cut at those gaps.
    """
    p0 = np.abs(seg0) ** 2
    p1 = np.abs(seg1) ** 2
    p_joint = np.minimum(p0, p1)

    win = max(1, int(valley_smoothing_ms * 1e-3 * sample_rate))
    p_smooth = _moving_average(p_joint, win)

    noise_lin = np.percentile(p_smooth, 20.0)
    noise_lin = max(noise_lin, 1e-18)
    valley_thr = noise_lin * (10.0 ** (valley_db_above_noise / 10.0))

    valley = p_smooth <= valley_thr
    min_valley_len = max(1, int(min_valley_us * 1e-6 * sample_rate))

    # find gaps where valley True for >= min_valley_len
    gaps: List[Tuple[int, int]] = []
    in_gap = False
    gs = 0
    for i, v in enumerate(valley):
        if v and not in_gap:
            in_gap = True
            gs = i
        elif not v and in_gap:
            ge = i - 1
            if ge - gs + 1 >= min_valley_len:
                gaps.append((gs, ge))
            in_gap = False
    if in_gap:
        ge = len(valley) - 1
        if ge - gs + 1 >= min_valley_len:
            gaps.append((gs, ge))

    # Convert gaps into segments between them
    segments: List[Tuple[int, int]] = []
    start = 0
    for gs, ge in gaps:
        if gs - start >= min_subburst_samples:
            segments.append((start, gs - 1))
        start = ge + 1
    if len(p_smooth) - start >= min_subburst_samples:
        segments.append((start, len(p_smooth) - 1))

    return segments


def split_by_valleys_and_minima(
    seg0: np.ndarray,
    seg1: np.ndarray,
    sample_rate: float,
    valley_smoothing_ms: float,
    noise_percentile: float,
    active_db_over_noise: float,
    valley_db_over_noise: float,
    min_valley_us: float,
    minima_prom_db: float,
    minima_distance_us: float,
    edge_db_over_noise: float,
    min_subburst_samples: int,
) -> List[Tuple[int, int]]:
    """General splitter using joint envelope with mandatory deep valleys and
    additional splits at prominent local minima.

    Steps
    - joint envelope: geometric mean of powers sqrt(p0*p1)
    - smooth with short moving-average
    - deep valleys (<= noise+valley_db) of duration >= min_valley_us define hard gaps
    - inside each chunk, find prominent minima (prominence >= minima_prom_db) to split
    - trim edges where envelope <= noise + edge_db
    """
    if len(seg0) == 0:
        return []

    p0 = np.abs(seg0) ** 2
    p1 = np.abs(seg1) ** 2
    # geometric mean is robust and only high when both channels are high
    p_joint = np.sqrt(p0 * p1)

    win = max(1, int(valley_smoothing_ms * 1e-3 * sample_rate))
    env = _moving_average(p_joint, win)

    noise_lin = np.percentile(env, noise_percentile)
    noise_lin = max(noise_lin, 1e-18)
    active_thr = noise_lin * (10.0 ** (active_db_over_noise / 10.0))
    valley_thr = noise_lin * (10.0 ** (valley_db_over_noise / 10.0))
    edge_thr = noise_lin * (10.0 ** (edge_db_over_noise / 10.0))

    # Active mask useful to ignore very low tails
    active = env >= active_thr

    # Deep valleys (hard gaps)
    valley = env <= valley_thr
    min_gap_len = max(1, int(min_valley_us * 1e-6 * sample_rate))
    gaps: List[Tuple[int, int]] = []
    in_gap = False
    gs = 0
    for i, v in enumerate(valley):
        if v and not in_gap:
            in_gap = True
            gs = i
        elif not v and in_gap:
            ge = i - 1
            if ge - gs + 1 >= min_gap_len:
                gaps.append((gs, ge))
            in_gap = False
    if in_gap:
        ge = len(valley) - 1
        if ge - gs + 1 >= min_gap_len:
            gaps.append((gs, ge))

    # Initial chunks between gaps
    chunks: List[Tuple[int, int]] = []
    start = 0
    for gs, ge in gaps:
        if gs - start >= min_subburst_samples:
            chunks.append((start, gs - 1))
        start = ge + 1
    if len(env) - start >= min_subburst_samples:
        chunks.append((start, len(env) - 1))

    # Refine each chunk by trimming edges using edge_thr and splitting at prominent minima
    result: List[Tuple[int, int]] = []
    min_dist = max(1, int(minima_distance_us * 1e-6 * sample_rate))
    env_db = 10.0 * np.log10(env + 1e-18)
    for cs, ce in chunks:
        # Edge trim
        s = cs
        while s <= ce and env[s] <= edge_thr:
            s += 1
        e = ce
        while e >= s and env[e] <= edge_thr:
            e -= 1
        if e - s + 1 < min_subburst_samples:
            continue

        # Find prominent minima inside [s,e]
        local = env_db[s:e+1]
        # peaks on negative envelope correspond to minima on envelope
        peaks, props = find_peaks(-local, prominence=minima_prom_db, distance=min_dist)
        if peaks.size == 0:
            result.append((s, e))
            continue
        # Convert to absolute indices and build segments
        cut_points = s + peaks
        # ensure cut points are strictly increasing and not at edges
        cut_points = cut_points[(cut_points > s + min_subburst_samples//2) & (cut_points < e - min_subburst_samples//2)]
        if cut_points.size == 0:
            result.append((s, e))
            continue
        prev = s
        for cp in cut_points:
            if cp - prev >= min_subburst_samples:
                result.append((prev, cp - 1))
                prev = cp + 1
        if e - prev + 1 >= min_subburst_samples:
            result.append((prev, e))

    return result


# ---------------- Level-based split (1D K-means over envelope) ---------------

def _binary_open_close(mask: np.ndarray, open_len: int, close_len: int) -> np.ndarray:
    # open = erode then dilate; close = dilate then erode; implemented via moving counts
    m = mask.astype(np.int32)
    if open_len > 1:
        k = np.ones(open_len, dtype=np.int32)
        eroded = (np.convolve(m, k, 'same') >= open_len).astype(np.int32)
        m = ((np.convolve(eroded, k, 'same') > 0)).astype(np.int32)
    if close_len > 1:
        k = np.ones(close_len, dtype=np.int32)
        dil = (np.convolve(m, k, 'same') > 0).astype(np.int32)
        m = (np.convolve(dil, k, 'same') >= close_len).astype(np.int32)
    return m.astype(bool)


def split_by_levels(
    seg0: np.ndarray,
    seg1: np.ndarray,
    sample_rate: float,
    smoothing_ms: float,
    noise_percentile: float,
    k_levels: int,
    min_hold_us: float,
    open_us: float,
    close_us: float,
    keep_top_levels: int,
    min_subburst_samples: int,
) -> List[Tuple[int, int]]:
    if len(seg0) == 0:
        return []
    p0 = np.abs(seg0) ** 2
    p1 = np.abs(seg1) ** 2
    env = _moving_average(np.sqrt(p0 * p1), max(1, int(smoothing_ms * 1e-3 * sample_rate)))

    # log domain for more stable clustering
    env_db = 10.0 * np.log10(env + 1e-18)
    active_thr = np.percentile(env, noise_percentile) * (10 ** (3.0 / 10.0))
    active = env >= active_thr
    x = env_db[active]
    if x.size == 0:
        return []

    # simple 1D k-means
    k = max(2, int(k_levels))
    c = np.percentile(x, np.linspace(10, 90, k))
    for _ in range(20):
        d = np.abs(x[:, None] - c[None, :])
        lab = np.argmin(d, axis=1)
        for i in range(k):
            xi = x[lab == i]
            if xi.size > 0:
                c[i] = np.mean(xi)
    order = np.argsort(c)  # low -> high
    high_labels = set(order[-keep_top_levels:])

    # map labels back to full timeline
    full_lab = np.zeros_like(env_db, dtype=int)
    full_lab[:] = order[0]
    full_lab[np.where(active)[0]] = order[lab]
    keep_mask = np.isin(full_lab, list(high_labels))

    # morphology
    min_hold = max(1, int(min_hold_us * 1e-6 * sample_rate))
    open_len = max(1, int(open_us * 1e-6 * sample_rate))
    close_len = max(1, int(close_us * 1e-6 * sample_rate))
    keep_mask = _binary_open_close(keep_mask, open_len=open_len, close_len=close_len)

    # extract segments
    segments: List[Tuple[int, int]] = []
    in_seg = False
    s = 0
    for i, v in enumerate(keep_mask):
        if v and not in_seg:
            in_seg = True
            s = i
        elif not v and in_seg:
            e = i - 1
            if e - s + 1 >= max(min_subburst_samples, min_hold):
                segments.append((s, e))
            in_seg = False
    if in_seg:
        e = len(keep_mask) - 1
        if e - s + 1 >= max(min_subburst_samples, min_hold):
            segments.append((s, e))

    return segments


def main():
    ap = argparse.ArgumentParser(description="Disconnect polar: split overlapped frames into sub-bursts to avoid mixing sources")
    ap.add_argument("--filename", default="2ch_iq_data.bin")
    ap.add_argument("--sample_rate", type=float, default=4e6)

    # Outer detection (frame-level)
    ap.add_argument("--threshold_db", type=float, default=14.0)
    ap.add_argument("--hysteresis_db", type=float, default=5.0)
    ap.add_argument("--smoothing_ms", type=float, default=0.02)
    ap.add_argument("--min_burst_us", type=float, default=5.0)
    ap.add_argument("--merge_gap_us", type=float, default=20.0)
    ap.add_argument("--pre_guard_us", type=float, default=0.0)
    ap.add_argument("--post_guard_us", type=float, default=0.0)
    ap.add_argument("--noise_percentile", type=float, default=20.0)
    ap.add_argument("--edge_refine_fraction", type=float, default=0.25)
    ap.add_argument("--edge_search_ms", type=float, default=0.05)

    # Inner re-segmentation (within overlap) using v2 (optional)
    ap.add_argument("--inner_threshold_db", type=float, default=18.0)
    ap.add_argument("--inner_hysteresis_db", type=float, default=6.0)
    ap.add_argument("--inner_smoothing_ms", type=float, default=0.01)
    ap.add_argument("--inner_min_burst_us", type=float, default=4.0)
    ap.add_argument("--inner_merge_gap_us", type=float, default=1.0)
    ap.add_argument("--inner_pre_guard_us", type=float, default=0.0)
    ap.add_argument("--inner_post_guard_us", type=float, default=0.0)
    ap.add_argument("--inner_noise_percentile", type=float, default=20.0)
    ap.add_argument("--inner_edge_refine_fraction", type=float, default=0.35)
    ap.add_argument("--inner_edge_search_ms", type=float, default=0.02)

    ap.add_argument("--require_intersection", action="store_true", help="Keep only sub-bursts present in both channels")
    ap.add_argument("--min_subburst_samples", type=int, default=64)

    # Valley/minima based forced split params
    ap.add_argument("--valley_smoothing_ms", type=float, default=0.003)
    ap.add_argument("--noise_percentile_valley", type=float, default=20.0)
    ap.add_argument("--active_db_over_noise", type=float, default=6.0)
    ap.add_argument("--valley_db_over_noise", type=float, default=1.5)
    ap.add_argument("--min_valley_us", type=float, default=20.0)
    ap.add_argument("--minima_prom_db", type=float, default=3.0)
    ap.add_argument("--minima_distance_us", type=float, default=60.0)
    ap.add_argument("--edge_db_over_noise", type=float, default=2.0)
    ap.add_argument("--use_valley_split", action="store_true", help="Use valley+minima forced split instead of inner v2")

    # Strong region selection inside sub-burst
    ap.add_argument("--strong_db_over_noise", type=float, default=10.0)
    ap.add_argument("--overlap_noise_percentile", type=float, default=20.0)
    ap.add_argument("--min_strong_samples", type=int, default=64)
    ap.add_argument("--topk_fallback", type=int, default=256)

    # Correlation
    ap.add_argument("--max_lag_samples", type=int, default=1)

    # Output
    ap.add_argument("--amplitude_dir", default="amplitude_plots_disconnect")
    ap.add_argument("--polar_path", default="correlation_polar_disconnect.png")
    ap.add_argument("--limit_pairs", type=int, default=0)

    # Level-based split params
    ap.add_argument("--k_levels", type=int, default=5)
    ap.add_argument("--min_hold_us", type=float, default=10.0)
    ap.add_argument("--open_us", type=float, default=5.0)
    ap.add_argument("--close_us", type=float, default=5.0)
    ap.add_argument("--keep_top_levels", type=int, default=3)
    ap.add_argument("--use_level_split", action="store_true", help="Use level-based split instead of inner v2")

    # Debug and dedup options
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_max_pairs", type=int, default=0, help="Print debug for first N pairs (0=all when --debug)")
    ap.add_argument("--debug_pair", type=int, default=0, help="Print debug for this 1-based pair index only (overrides max_pairs check)")
    ap.add_argument("--save_debug_envelope", action="store_true", help="Save envelope debug plots with cut points")
    ap.add_argument("--dedup_iou", type=float, default=0.8, help="IoU threshold to treat two sub-bursts as duplicates")
    ap.add_argument("--dedup_min_gap_us", type=float, default=10.0, help="Minimum separation to keep two adjacent sub-bursts")

    # Amplitude-based gating and trimming
    ap.add_argument("--min_mean_amp", type=float, default=0.05, help="Drop sub-bursts whose mean(|ch0|,|ch1|) is below this")
    ap.add_argument("--amp_edge_thr", type=float, default=0.05, help="Trim sub-burst edges where joint amplitude < thr")
    ap.add_argument("--amp_trim_min_us", type=float, default=10.0, help="Minimum retained length after amplitude-trim (us)")

    args = ap.parse_args()

    os.makedirs(args.amplitude_dir, exist_ok=True)

    ch0, ch1 = read_2ch_iq(args.filename)
    if ch0 is None:
        print(f"Error: cannot read IQ file '{args.filename}'")
        return

    # Outer detection
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
    print(f"Paired bursts (outer): {len(pairs)}")
    if not pairs:
        return
    if args.limit_pairs > 0:
        pairs = pairs[: args.limit_pairs]

    corr_mags: List[float] = []
    corr_phases: List[float] = []

    sub_count = 0
    for pidx, (b0, b1) in enumerate(pairs):
        s0, e0 = b0
        s1, e1 = b1
        s = max(s0, s1)
        e = min(e0, e1)
        if e <= s:
            continue
        seg0 = ch0[s : e + 1]
        seg1 = ch1[s : e + 1]

        debug_this = (args.debug and (args.debug_max_pairs == 0 or pidx < args.debug_max_pairs)) or (args.debug_pair and pidx+1 == args.debug_pair)
        if debug_this:
            print(f"[PAIR {pidx+1}] overlap: [{s},{e}] len={e-s+1}")

        # Choose splitting strategy
        if args.use_valley_split:
            subbursts = split_by_valleys_and_minima(
                seg0,
                seg1,
                sample_rate=args.sample_rate,
                valley_smoothing_ms=args.valley_smoothing_ms,
                noise_percentile=args.noise_percentile_valley,
                active_db_over_noise=args.active_db_over_noise,
                valley_db_over_noise=args.valley_db_over_noise,
                min_valley_us=args.min_valley_us,
                minima_prom_db=args.minima_prom_db,
                minima_distance_us=args.minima_distance_us,
                edge_db_over_noise=args.edge_db_over_noise,
                min_subburst_samples=args.min_subburst_samples,
            )
            if (args.debug and (args.debug_max_pairs == 0 or pidx < args.debug_max_pairs)) or (args.debug_pair and pidx+1 == args.debug_pair):
                # 保存联合包络与切分点
                p0 = np.abs(seg0) ** 2
                p1 = np.abs(seg1) ** 2
                env = _moving_average(np.sqrt(p0*p1), max(1, int(args.valley_smoothing_ms * 1e-3 * args.sample_rate)))
                noise_lin = np.percentile(env, args.noise_percentile_valley)
                thr_active = noise_lin * (10**(args.active_db_over_noise/10.0))
                thr_valley = noise_lin * (10**(args.valley_db_over_noise/10.0))
                thr_edge = noise_lin * (10**(args.edge_db_over_noise/10.0))
                if args.save_debug_envelope:
                    import matplotlib.pyplot as _plt
                    fig,_ax = _plt.subplots(figsize=(10,3))
                    _ax.plot(10*np.log10(env+1e-18), label='env_db')
                    _ax.axhline(10*np.log10(thr_active+1e-18), color='g', ls='--', label='active')
                    _ax.axhline(10*np.log10(thr_valley+1e-18), color='r', ls='--', label='valley')
                    _ax.axhline(10*np.log10(thr_edge+1e-18), color='k', ls=':', label='edge')
                    for ss,ee in subbursts:
                        _ax.axvspan(ss, ee, color='y', alpha=0.2)
                    _ax.legend(); _ax.set_title(f'PAIR {pidx+1} env & cuts')
                    fig.tight_layout()
                    fig.savefig(os.path.join(args.amplitude_dir, f"pair_{pidx+1:03d}_env_debug.png"), dpi=150)
                    _plt.close(fig)
        elif args.use_level_split:
            subbursts = split_by_levels(
                seg0,
                seg1,
                sample_rate=args.sample_rate,
                smoothing_ms=args.inner_smoothing_ms,
                noise_percentile=args.inner_noise_percentile,
                k_levels=args.k_levels,
                min_hold_us=args.min_hold_us,
                open_us=args.open_us,
                close_us=args.close_us,
                keep_top_levels=args.keep_top_levels,
                min_subburst_samples=args.min_subburst_samples,
            )
        else:
            subbursts = split_overlap_into_subbursts(
                seg0,
                seg1,
                sample_rate=args.sample_rate,
                inner_threshold_db=args.inner_threshold_db,
                inner_hysteresis_db=args.inner_hysteresis_db,
                inner_smoothing_ms=args.inner_smoothing_ms,
                inner_min_burst_us=args.inner_min_burst_us,
                inner_merge_gap_us=args.inner_merge_gap_us,
                inner_pre_guard_us=args.inner_pre_guard_us,
                inner_post_guard_us=args.inner_post_guard_us,
                inner_noise_percentile=args.inner_noise_percentile,
                inner_edge_refine_fraction=args.inner_edge_refine_fraction,
                inner_edge_search_ms=args.inner_edge_search_ms,
                require_intersection=args.require_intersection,
                min_subburst_samples=args.min_subburst_samples,
            )
        if debug_this:
            print(f"  initial subbursts (local idx): {subbursts}")

        # Dedup/merge: sort and remove large-overlap duplicates; enforce min-gap
        def iou(a, b):
            ss = max(a[0], b[0]); ee = min(a[1], b[1])
            inter = max(0, ee - ss + 1)
            union = (a[1]-a[0]+1) + (b[1]-b[0]+1) - inter
            return inter/union if union > 0 else 0.0

        if subbursts:
            subbursts = sorted(subbursts)
            pruned = []
            min_gap = int(max(0, args.dedup_min_gap_us) * 1e-6 * args.sample_rate)
            for sb in subbursts:
                keep = True
                for pb in pruned:
                    if iou(sb, pb) >= args.dedup_iou:
                        keep = False
                        if debug_this:
                            print(f"    drop {sb} due to high IoU with {pb}")
                        break
                    # enforce min gap
                    if sb[0] <= pb[1] + min_gap and pb[0] <= sb[1] + min_gap:
                        # overlapping or too close: keep the longer one
                        if (sb[1]-sb[0]) <= (pb[1]-pb[0]):
                            keep = False
                            if debug_this:
                                print(f"    drop {sb} due to too-close to {pb}")
                            break
                        else:
                            if debug_this:
                                print(f"    replace {pb} by {sb} (longer)")
                            pruned.remove(pb)
                            break
                if keep:
                    pruned.append(sb)
            subbursts = pruned

        if debug_this:
            print(f"  pruned subbursts: {subbursts}")

        if not subbursts:
            continue

        for sbidx, (ls, le) in enumerate(subbursts):
            sub0 = seg0[ls : le + 1]
            sub1 = seg1[ls : le + 1]

            # Amplitude-based strict trim and gating
            joint_amp = 0.5 * (np.abs(sub0) + np.abs(sub1))
            # Trim edges where amplitude < amp_edge_thr
            left = 0
            while left < len(joint_amp) and joint_amp[left] < args.amp_edge_thr:
                left += 1
            right = len(joint_amp) - 1
            while right >= left and joint_amp[right] < args.amp_edge_thr:
                right -= 1
            if right - left + 1 <= 0:
                if debug_this:
                    print(f"    drop sub {ls,le}: trimmed to empty by amp_edge_thr={args.amp_edge_thr}")
                continue
            sub0 = sub0[left:right+1]
            sub1 = sub1[left:right+1]

            # Mean amplitude gating
            mean_amp = float(np.mean(0.5 * (np.abs(sub0) + np.abs(sub1))))
            if mean_amp < args.min_mean_amp:
                if debug_this:
                    print(f"    drop sub {ls,le}: mean_amp={mean_amp:.3f} < min_mean_amp={args.min_mean_amp}")
                continue

            # Ensure length after trimming
            if len(sub0) < max(args.min_strong_samples, int(args.amp_trim_min_us * 1e-6 * args.sample_rate)):
                if debug_this:
                    print(f"    drop sub {ls,le}: too short after amp trim len={len(sub0)}")
                continue

            # Strong selection
            p0 = np.abs(sub0) ** 2
            p1 = np.abs(sub1) ** 2
            n0 = np.percentile(p0, args.overlap_noise_percentile)
            n1 = np.percentile(p1, args.overlap_noise_percentile)
            thr0 = n0 * (10.0 ** (args.strong_db_over_noise / 10.0))
            thr1 = n1 * (10.0 ** (args.strong_db_over_noise / 10.0))
            mask = (p0 >= thr0) & (p1 >= thr1)
            strong_idx = np.flatnonzero(mask)

            if strong_idx.size >= args.min_strong_samples:
                ss = strong_idx[0]
                ee = strong_idx[-1]
                s1_str = sub0[ss : ee + 1]
                s2_str = sub1[ss : ee + 1]
            else:
                k = min(args.topk_fallback, len(sub0))
                idx0 = np.argsort(p0)[-k:]
                idx1 = np.argsort(p1)[-k:]
                common = np.intersect1d(idx0, idx1)
                if common.size == 0:
                    continue
                common.sort()
                s1_str = sub0[common]
                s2_str = sub1[common]

            phi_deg, best_lag, cval = compute_phase_difference(s1_str, s2_str, max_lag_samples=args.max_lag_samples)
            corr_mags.append(abs(cval))
            corr_phases.append(phi_deg)

            sub_count += 1
            # Amplitude plot per sub-burst
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(np.abs(sub0), label="|Ch0|")
            ax.plot(np.abs(sub1), label="|Ch1|", alpha=0.85)
            ax.set_title(f"Pair {pidx+1}, sub {sbidx+1}: len={le-ls+1}; lag={best_lag}; |corr|={abs(cval):.3e}; phase={phi_deg:.2f}°")
            ax.set_xlabel("Sample index (sub-burst)")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
            ax.legend()
            out_path = os.path.join(args.amplitude_dir, f"pair_{pidx+1:03d}_sub_{sbidx+1:02d}.png")
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

    print(f"Total sub-bursts analyzed: {sub_count}")

    # Polar clustering plot
    if corr_mags:
        mags = np.asarray(corr_mags)
        mag_db = 20.0 * np.log10(mags + 1e-18)
        theta_rad = np.deg2rad(np.mod(np.asarray(corr_phases), 360.0))

        fig2 = plt.figure(figsize=(8, 8))
        axp = fig2.add_subplot(111, projection='polar')
        axp.set_theta_zero_location('E')
        axp.set_theta_direction(1)
        sc = axp.scatter(theta_rad, mag_db, c=mags, cmap='viridis', s=24, alpha=0.9, edgecolors='none')
        cbar = fig2.colorbar(sc, pad=0.1)
        cbar.set_label('|corr| (linear)')
        axp.set_rlabel_position(135)
        axp.set_title('Disconnect polar (radius=dB, angle=phase)')
        fig2.tight_layout()
        fig2.savefig(args.polar_path, dpi=150)
        plt.close(fig2)


if __name__ == "__main__":
    main()
