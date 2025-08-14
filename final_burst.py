import os
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =============================================================================
#  HELPER & UTILITY FUNCTIONS
# =============================================================================

def _ensure_parent_dir(path: str) -> None:
    """Create parent directory for a file path if it is non-empty."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def read_2ch_iq(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads dual-channel IQ data from a binary file."""
    try:
        data = np.fromfile(filename, dtype=np.complex64)
        ch0 = data[0::2]
        ch1 = data[1::2]
        return ch0, ch1
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None, None

def pair_bursts(bursts0, bursts1, max_center_diff):
    """Pairs bursts from two channels based on center proximity."""
    pairs = []
    for b0 in bursts0:
        c0 = (b0[0] + b0[1]) / 2
        best_b1, min_diff = None, float('inf')
        for b1 in bursts1:
            c1 = (b1[0] + b1[1]) / 2
            diff = abs(c0 - c1)
            if diff < max_center_diff and diff < min_diff:
                min_diff, best_b1 = diff, b1
        if best_b1:
            pairs.append((b0, best_b1))
    return pairs

def detect_signal_bursts_v2(iq_data: np.ndarray, sample_rate: float, threshold_db: float = 10.0,
                            hysteresis_db: float = 3.0, smoothing_ms: float = 0.2, min_burst_us: float = 100.0,
                            merge_gap_us: float = 50.0, **kwargs) -> Tuple[List[List[int]], dict]:
    """V2 power-based burst detector."""
    if iq_data is None or len(iq_data) == 0:
        return [], {}
    power_lin = np.abs(iq_data) ** 2
    tau_samples = max(1, int(smoothing_ms * 1e-3 * sample_rate))
    beta = 1.0 / float(tau_samples)
    b, a = [beta], [1.0, -(1.0 - beta)]
    envelope_lin = signal.lfilter(b, a, power_lin)
    noise_floor_lin = np.percentile(envelope_lin, kwargs.get("noise_percentile", 20.0))
    noise_floor_lin = max(noise_floor_lin, 1e-18)
    enter_thr_lin = noise_floor_lin * (10.0 ** (threshold_db / 10.0))
    exit_thr_lin = noise_floor_lin * (10.0 ** ((threshold_db - hysteresis_db) / 10.0))
    N = len(envelope_lin)
    bursts: list[list[int]] = []
    in_burst = False
    start_idx = 0
    for i in range(N):
        val = envelope_lin[i]
        if not in_burst:
            if val >= enter_thr_lin:
                in_burst, start_idx = True, i
        else:
            if val < exit_thr_lin:
                in_burst = False
                bursts.append([start_idx, i - 1])
    if in_burst:
        bursts.append([start_idx, N - 1])
    
    merge_gap_samples = int(max(0, merge_gap_us) * 1e-6 * sample_rate)
    merged: list[list[int]] = []
    if bursts:
        merged.append(bursts[0])
        for bseg in bursts[1:]:
            if bseg[0] - merged[-1][1] <= merge_gap_samples:
                merged[-1][1] = max(merged[-1][1], bseg[1])
            else:
                merged.append(bseg)
    bursts = merged
    
    min_len_samples = int(max(0, min_burst_us) * 1e-6 * sample_rate)
    filtered = [b for b in bursts if (b[1] - b[0] + 1) >= min_len_samples]
    
    return filtered, {}

def split_frame_by_gaps(seg0: np.ndarray, seg1: np.ndarray, sample_rate: float,
                        gap_threshold: float, min_gap_us: float, min_subburst_us: float) -> List[Tuple[int, int]]:
    """Splits a frame into sub-bursts based on low-power gaps."""
    if len(seg0) == 0:
        return []
    joint_envelope = np.minimum(np.abs(seg0), np.abs(seg1))
    is_gap = joint_envelope < gap_threshold
    min_gap_samples = int(min_gap_us * 1e-6 * sample_rate)
    if min_gap_samples < 1: min_gap_samples = 1
    
    cut_points = [0]
    in_gap, gap_start = False, 0
    for i, v in enumerate(is_gap):
        if v and not in_gap:
            in_gap, gap_start = True, i
        elif not v and in_gap:
            in_gap = False
            if i - gap_start >= min_gap_samples:
                cut_points.append(gap_start + (i - gap_start) // 2)
    cut_points.append(len(seg0))
    
    subbursts = []
    min_subburst_samples = int(min_subburst_us * 1e-6 * sample_rate)
    cut_points = sorted(list(set(cut_points)))
    for i in range(len(cut_points) - 1):
        s, e = cut_points[i], cut_points[i+1] - 1
        if (e - s + 1) >= min_subburst_samples:
            subbursts.append((s, e))
    return subbursts

# =============================================================================
#  CORE CALCULATION (V6 - WITH CFO CORRECTION)
# =============================================================================

def compute_phase_with_cfo_correction(s1: np.ndarray, s2: np.ndarray, sample_rate: float) -> Tuple[float, float, float]:
    """
    Estimates and corrects for CFO by linearly fitting the instantaneous phase.
    Returns: (CFO-Corrected Phase [deg], Estimated CFO [Hz], Mean Correlation Power)
    """
    if len(s1) < 10:  # Require a minimum number of points for a stable fit
        return float('nan'), float('nan'), 0.0

    # 1. Calculate instantaneous phase difference
    instant_phase_rad = np.angle(s1 * np.conj(s2))
    
    # 2. Unwrap the phase to handle -pi to pi wrap-around
    unwrapped_phase_rad = np.unwrap(instant_phase_rad)
    
    # 3. Perform a 1st-degree polynomial (linear) fit to the phase trend
    time_axis = np.arange(len(unwrapped_phase_rad))
    slope, intercept = np.polyfit(time_axis, unwrapped_phase_rad, 1)
    
    # 4. Convert slope to frequency offset in Hz
    # slope is in radians/sample. Multiply by sample_rate to get rad/sec. Divide by 2*pi for Hz.
    cfo_hz = slope * sample_rate / (2 * np.pi)
    
    # 5. The intercept is the CFO-corrected phase at t=0 of the burst. Convert to degrees.
    corrected_phase_deg = np.rad2deg(intercept)
    
    # 6. Calculate a length-independent measure of correlation strength (mean of power-like product)
    mean_corr_power = np.mean(np.abs(s1) * np.abs(s2))
    
    return corrected_phase_deg, cfo_hz, mean_corr_power

# =============================================================================
#  PLOTTING FUNCTIONS
# =============================================================================

def generate_output_plots(phases, cfos, corrs, output_dir):
    """Generates all analysis plots: Polar, Summary, and CFO Histogram."""
    if len(phases) == 0:
        print("No data to plot.")
        return

    # Ensure the output directory exists (output_dir is a directory, not a file)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. CFO-Corrected Polar Plot ---
    corr_lin = np.asarray(corrs)
    corr_db = 10.0 * np.log10(corr_lin + 1e-18) # Use 10*log10 for power-like quantities
    phase_rad = np.deg2rad(np.mod(np.asarray(phases), 360.0))
    
    fig1 = plt.figure(figsize=(10, 10))
    axp = fig1.add_subplot(111, projection='polar')
    sc = axp.scatter(phase_rad, corr_db, c=corr_lin, cmap='viridis', s=30, alpha=0.8, edgecolors='k', linewidth=0.5)
    cbar = fig1.colorbar(sc, pad=0.1, shrink=0.8)
    cbar.set_label('|s1|*|s2| (Mean Power)', fontsize=12)
    axp.set_title(f'CFO-Corrected Polar Plot ({len(phases)} points)', fontsize=16, pad=20)
    THEORETICAL_PHASE = 72
    axp.plot([np.deg2rad(THEORETICAL_PHASE)]*2, axp.get_ylim(), color='r', ls='--', lw=2, label=f'Theoretical LOS ({THEORETICAL_PHASE}°)')
    axp.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.1))
    fig1.savefig(os.path.join(output_dir, "polar_plot_cfo_corrected.png"), dpi=150)
    plt.close(fig1)

    # --- 2. CFO-Corrected Summary Plot ---
    x_axis = np.arange(len(phases))
    fig2, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig2.suptitle('CFO-Corrected Analysis Across Sub-Bursts', fontsize=16)
    axs[0].plot(x_axis, corr_db, "o-")
    axs[0].set_ylabel("Correlation Power (dB)")
    axs[0].grid(True)
    axs[0].set_title("Magnitude vs. Sub-Burst Index")
    axs[1].plot(x_axis, [(p + 180) % 360 - 180 for p in phases], "o-", color='C1') # Wrap phase to -180,180
    axs[1].set_ylabel("CFO-Corrected Phase (deg)")
    axs[1].set_xlabel("Sub-Burst Index")
    axs[1].set_ylim(-180, 180)
    axs[1].grid(True)
    fig2.savefig(os.path.join(output_dir, "summary_plot_cfo_corrected.png"), dpi=150)
    plt.close(fig2)

    # --- 3. CFO Histogram ---
    fig3 = plt.figure(figsize=(10, 6))
    plt.hist(cfos, bins=75, edgecolor='k')
    plt.title('Distribution of Estimated CFOs', fontsize=16)
    plt.xlabel('Frequency Offset (Hz)')
    plt.ylabel('Number of Sub-Bursts')
    plt.grid(True)
    fig3.savefig(os.path.join(output_dir, "cfo_histogram.png"), dpi=150)
    plt.close(fig3)
    
    print(f"Generated all plots in directory: {output_dir}")

# =============================================================================
#  MAIN EXECUTION
# =============================================================================

def main():
    # --- Parameters ---
    FILENAME = "2ch_iq_data.bin"
    SAMPLE_RATE = 4e6
    OUTPUT_DIR = "results_v6_cfo"
    # TEMP: analyze only pair #1 and print its overall phase and duration
    ONLY_PAIR_INDEX = 1  # 1-based index; set to None to analyze all
    
    # Detection & Segmentation Parameters
    COARSE_THRESHOLD_DB = 14.0
    COARSE_MERGE_GAP_US = 20.0
    MIN_BURST_US_COARSE = 5.0
    GAP_THRESHOLD = 0.05
    MIN_GAP_US = 5.0
    MIN_SUBBURST_US = 40.0
    
    # Quality & Analysis Parameters
    MIN_MEAN_AMPLITUDE = 0.1
    MIN_SAMPLES_FOR_CFO_FIT = 50 # A good fit needs enough points

    # --- Data Processing Pipeline ---
    ch0, ch1 = read_2ch_iq(FILENAME)
    if ch0 is None:
        print("Failed to read IQ data. Exiting.")
        return

    # Step 1: Coarse burst detection
    bursts0, _ = detect_signal_bursts_v2(ch0, sample_rate=SAMPLE_RATE, threshold_db=COARSE_THRESHOLD_DB, merge_gap_us=COARSE_MERGE_GAP_US, min_burst_us=MIN_BURST_US_COARSE)
    bursts1, _ = detect_signal_bursts_v2(ch1, sample_rate=SAMPLE_RATE, threshold_db=COARSE_THRESHOLD_DB, merge_gap_us=COARSE_MERGE_GAP_US, min_burst_us=MIN_BURST_US_COARSE)
    
    # Step 2: Pair bursts from both channels
    pairs = pair_bursts(bursts0, bursts1, max_center_diff=int(0.3e-3 * SAMPLE_RATE))
    print(f"Found {len(pairs)} initial coarse frames.")

    # TEMP path: if ONLY_PAIR_INDEX is set, report only that pair's overall phase and duration
    if ONLY_PAIR_INDEX is not None:
        if len(pairs) < ONLY_PAIR_INDEX:
            print(f"Requested pair #{ONLY_PAIR_INDEX} not available (only {len(pairs)} pairs).")
            return
        b0, b1 = pairs[ONLY_PAIR_INDEX - 1]
        s, e = max(b0[0], b1[0]), min(b0[1], b1[1])
        total_len = e - s + 1
        dur_sec = total_len / SAMPLE_RATE
        seg0, seg1 = ch0[s:e+1], ch1[s:e+1]
        phase_deg, cfo_hz, mean_corr_power = compute_phase_with_cfo_correction(seg0, seg1, SAMPLE_RATE)
        print("-" * 60)
        print(f"Pair #{ONLY_PAIR_INDEX}: overlap=[{s},{e}], samples={total_len}, duration={dur_sec*1e6:.2f} us")
        print(f"Overall CFO-corrected phase (Ch0 vs Ch1): {phase_deg:.2f} deg")
        print(f"Estimated CFO: {cfo_hz:.1f} Hz")
        print(f"Mean correlation power (|s0|*|s1| mean): {mean_corr_power:.4f}")
        print("-" * 60)
        return

    # Step 3: Iterate, segment, filter, and analyze (all pairs)
    corrected_phases, cfo_estimates, correlations = [], [], []

    for pidx, (b0, b1) in enumerate(pairs):
        s, e = max(b0[0], b1[0]), min(b0[1], b1[1])
        if (e - s + 1) < (MIN_SUBBURST_US * 1e-6 * SAMPLE_RATE):
            continue
        
        seg0, seg1 = ch0[s:e+1], ch1[s:e+1]
        
        # Segment the coarse frame into sub-bursts
        subbursts = split_frame_by_gaps(seg0, seg1, sample_rate=SAMPLE_RATE, gap_threshold=GAP_THRESHOLD, min_gap_us=MIN_GAP_US, min_subburst_us=MIN_SUBBURST_US)

        for ls, le in subbursts:
            sub0, sub1 = seg0[ls:le+1], seg1[ls:le+1]
            
            # Quality filtering
            if len(sub0) < MIN_SAMPLES_FOR_CFO_FIT:
                continue
            if np.mean(0.5 * (np.abs(sub0) + np.abs(sub1))) < MIN_MEAN_AMPLITUDE:
                continue

            # Step 4: Perform CFO correction and get results
            phase, cfo, corr = compute_phase_with_cfo_correction(sub0, sub1, SAMPLE_RATE)
            
            if not np.isnan(phase):
                corrected_phases.append(phase)
                cfo_estimates.append(cfo)
                correlations.append(corr)

    print(f"Kept and analyzed {len(corrected_phases)} high-quality sub-bursts.")
    
    # Step 5: Generate all output plots
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_output_plots(corrected_phases, cfo_estimates, correlations, OUTPUT_DIR)

if __name__ == "__main__":
    main()