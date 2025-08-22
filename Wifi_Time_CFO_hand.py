import os
from typing import List, Tuple
import finufft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =============================================================================
#  依赖函数 (与之前版本相同)
# =============================================================================

def read_2ch_iq(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """从文件中读取双通道IQ数据"""
    try:
        data = np.fromfile(filename, dtype=np.complex64)
        ch0 = data[0::2]
        ch1 = data[1::2]
        return ch0, ch1
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None, None

def pair_bursts(bursts0, bursts1, max_center_diff):
    """基于中心点距离配对两个通道的突发"""
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

def compute_phase_difference(s1: np.ndarray, s2: np.ndarray, max_lag_samples: int = 0) -> Tuple[float, int, complex]:
    """
    计算相位差 (V4版：相关性值已除以长度N，使其成为不受长度影响的平均值)
    """
    assert len(s1) == len(s2)
    n_samples = len(s1)
    if n_samples == 0:
        return float("nan"), 0, 0j

    def corr_at_lag(lag: int) -> complex:
        if lag == 0:
            c = np.sum(s1 * np.conj(s2))
        elif lag > 0:
            c = np.sum(s1[lag:] * np.conj(s2[:-lag]))
        else:
            L = -lag
            c = np.sum(s1[:-L] * np.conj(s2[L:]))
        
        effective_len = n_samples - abs(lag)
        return c / effective_len if effective_len > 0 else 0j

    if max_lag_samples <= 0:
        c0 = corr_at_lag(0)
        return float(np.rad2deg(np.angle(c0))), 0, c0

    best_lag, best_val, best_mag = 0, 0j, -1.0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        c = corr_at_lag(lag)
        m = abs(c)
        if m > best_mag:
            best_mag, best_val, best_lag = m, c, lag
            
    return float(np.rad2deg(np.angle(best_val))), best_lag, best_val

def detect_signal_bursts_v2(iq_data: np.ndarray, sample_rate: float, threshold_db: float = 10.0,
    hysteresis_db: float = 3.0, smoothing_ms: float = 0.2, min_burst_us: float = 100.0,
    merge_gap_us: float = 50.0, pre_guard_us: float = 0.0, post_guard_us: float = 0.0,
    noise_percentile: float = 20.0, edge_refine_fraction: float = 0.25,
    edge_search_ms: float = 0.1):
    """V2版本突发检测器 (完整版)"""
    if iq_data is None or len(iq_data) == 0:
        return [], {}
    power_lin = np.abs(iq_data) ** 2
    tau_samples = max(1, int(smoothing_ms * 1e-3 * sample_rate))
    beta = 1.0 / float(tau_samples)
    b, a = [beta], [1.0, -(1.0 - beta)]
    envelope_lin = signal.lfilter(b, a, power_lin)
    noise_floor_lin = np.percentile(envelope_lin, noise_percentile)
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
                in_burst = True
                start_idx = i
        else:
            if val < exit_thr_lin:
                in_burst = False
                end_idx = i - 1
                bursts.append([start_idx, end_idx])
    if in_burst:
        bursts.append([start_idx, N - 1])
    if not bursts and np.all(envelope_lin >= enter_thr_lin):
        bursts = [[0, N - 1]]
    merge_gap_samples = int(max(0, merge_gap_us) * 1e-6 * sample_rate)
    merged: list[list[int]] = []
    if bursts:
        merged.append(bursts[0])
        for bseg in bursts[1:]:
            last = merged[-1]
            if bseg[0] - last[1] <= merge_gap_samples:
                last[1] = max(last[1], bseg[1])
            else:
                merged.append(bseg)
    bursts = merged
    min_len_samples = int(max(0, min_burst_us) * 1e-6 * sample_rate)
    filtered: list[list[int]] = []
    for s, e in bursts:
        if e - s + 1 >= min_len_samples:
            filtered.append([s, e])
    return filtered, {}

def split_frame_by_gaps(seg0: np.ndarray, seg1: np.ndarray, sample_rate: float,
    gap_threshold: float, min_gap_us: float, min_subburst_us: float) -> List[Tuple[int, int]]:
    """基于间隙的分割算法 (完整版)"""
    if len(seg0) == 0:
        return []
    joint_envelope = np.minimum(np.abs(seg0), np.abs(seg1))
    is_gap = joint_envelope < gap_threshold
    min_gap_samples = int(min_gap_us * 1e-6 * sample_rate)
    if min_gap_samples < 1: min_gap_samples = 1
    cut_points = [0]
    in_gap = False
    gap_start = 0
    for i in range(len(is_gap)):
        if is_gap[i] and not in_gap:
            in_gap, gap_start = True, i
        elif not is_gap[i] and in_gap:
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

def plot_correlation_summary_db(subburst_times, corr_mags, corr_phases, output_path):
    """
    生成dB版本的摘要图 (V10: 使用校正后的相位数据)
    """
    if len(corr_mags) == 0:
        print("No correlation data to plot for summary.")
        return

    mags_db = 20.0 * np.log10(np.asarray(corr_mags) + 1e-18)
    phases = np.asarray(corr_phases) # Now contains corrected phases
    x_axis = np.asarray(subburst_times)
    
    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.suptitle('Correlation Analysis Across Sub-Bursts (CFO Corrected)', fontsize=16) # Title updated
    
    axs[0].plot(x_axis, mags_db, "o-")
    axs[0].set_ylabel("Correlation Magnitude (dB)")
    axs[0].grid(True)
    axs[0].set_title("Magnitude (dB) vs. Time")
    
    # Text box now reflects stats of the corrected phase
    text_str = (f"N = {len(mags_db)}\nMedian Mag = {np.median(mags_db):.2f} dB\n"
                f"Mean Phase = {np.mean(phases):.2f}°\nStd Phase = {np.std(phases):.2f}°")
    axs[0].text(0.02, 0.95, text_str, transform=axs[0].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axs[1].plot(x_axis, phases, "o-", color='C1')
    axs[1].set_ylabel("Phase (deg)")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylim(-180, 180)
    axs[1].set_yticks(np.arange(-180, 181, 90))
    axs[1].grid(True)
    axs[1].set_title("CFO Corrected Phase vs. Time") # Sub-title updated
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Generated CFO-corrected summary plot: {output_path}")

def plot_polar_db(corr_mags, corr_phases, output_path):
    """
    生成dB版本的极坐标图 (V10: 使用校正后的相位数据)
    """
    if len(corr_mags) == 0:
        print("No correlation data to plot for polar.")
        return
        
    mags_lin = np.asarray(corr_mags)
    mags_db = 20.0 * np.log10(mags_lin + 1e-18)
    # Now uses corrected phases
    theta_rad = np.deg2rad(np.mod(np.asarray(corr_phases), 360.0))
    
    fig = plt.figure(figsize=(10, 10))
    axp = fig.add_subplot(111, projection='polar')
    sc = axp.scatter(theta_rad, mags_db, c=mags_lin, cmap='viridis', s=30, alpha=0.8, edgecolors='k', linewidth=0.5)
    cbar = fig.colorbar(sc, pad=0.1, shrink=0.8)
    cbar.set_label('|Correlation| (Normalized, Linear Scale)', fontsize=12)
    r_min, r_max = np.floor(np.min(mags_db) / 5) * 5, np.ceil(np.max(mags_db) / 5) * 5
    axp.set_rlim(max(r_min, -80), r_max)
    r_ticks = np.linspace(max(r_min, -80), r_max, num=5)
    axp.set_rgrids(r_ticks, labels=[f"{int(tick)} dB" for tick in r_ticks], angle=22.5, fontsize=10)
    THEORETICAL_PHASE = 72 # This might need re-evaluation after CFO correction
    axp.plot([np.deg2rad(THEORETICAL_PHASE)]*2, axp.get_ylim(), color='r', ls='--', lw=2, label=f'Theoretical LOS ({THEORETICAL_PHASE}°)')
    
    axp.set_title(f'CFO Corrected Phase-Correlation Polar Plot ({len(corr_mags)} points)', fontsize=16, pad=20)
    axp.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Generated CFO-corrected polar plot: {output_path}")

# =============================================================================
#  主函数 (V10 - 增加CFO校正)
# =============================================================================
def main():
    # --- 参数设置 ---
    FILENAME = "2ch_iq_data.bin"
    SAMPLE_RATE = 4e6
    
    # <<< V10 MODIFICATION: CFO Correction Parameter >>>
    # 您可以在这里手动调整CFO补偿值 (单位: Hz)
    CFO_HZ_CORRECTION = 0
    
    COARSE_THRESHOLD_DB = 14.0
    COARSE_MERGE_GAP_US = 20.0
    MIN_BURST_US_COARSE = 5.0
    GAP_THRESHOLD = 0.05
    MIN_GAP_US = 5.0
    MIN_SUBBURST_US = 40.0
    MIN_MEAN_AMPLITUDE = 0.1
    MIN_STRONG_SAMPLES = 64
    MAX_LAG_SAMPLES = 1
    
    # <<< V10 MODIFICATION: Update output filenames >>>
    POLAR_PATH = "correlation_polar_V10_CFO_Corrected.png"
    SUMMARY_PATH = "correlation_summary_db_V10_CFO_Corrected.png"

    TARGET_PAIRS: List[int] = [] 

    # --- 数据处理流程 ---
    for path in [POLAR_PATH, SUMMARY_PATH]:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
    ch0, ch1 = read_2ch_iq(FILENAME)
    if ch0 is None:
        return

    bursts0, _ = detect_signal_bursts_v2(ch0, sample_rate=SAMPLE_RATE, threshold_db=COARSE_THRESHOLD_DB, merge_gap_us=COARSE_MERGE_GAP_US, min_burst_us=MIN_BURST_US_COARSE)
    bursts1, _ = detect_signal_bursts_v2(ch1, sample_rate=SAMPLE_RATE, threshold_db=COARSE_THRESHOLD_DB, merge_gap_us=COARSE_MERGE_GAP_US, min_burst_us=MIN_BURST_US_COARSE)
    all_pairs = pair_bursts(bursts0, bursts1, max_center_diff=int(0.3e-3 * SAMPLE_RATE))
    print(f"Found {len(all_pairs)} initial coarse frames (large frames) in total.")

    pairs_to_process = []
    if not TARGET_PAIRS:
        print("Processing all detected pairs.")
        pairs_to_process = list(enumerate(all_pairs, 1))
    else:
        # ... (code for selecting pairs remains the same)
        pass
    
    corr_mags, corr_phases, corr_lags = [], [], []
    subburst_times = []

    for pidx, (b0, b1) in pairs_to_process:
        s, e = max(b0[0], b1[0]), min(b0[1], b1[1])
        # ... (code for frame info printing remains the same)
        
        seg0, seg1 = ch0[s:e+1], ch1[s:e+1]
        subbursts = split_frame_by_gaps(seg0, seg1, sample_rate=SAMPLE_RATE, gap_threshold=GAP_THRESHOLD, min_gap_us=MIN_GAP_US, min_subburst_us=MIN_SUBBURST_US)

        for ls, le in subbursts:
            sub0, sub1 = seg0[ls:le+1], seg1[ls:le+1]
            if len(sub0) < MIN_STRONG_SAMPLES or np.mean(0.5 * (np.abs(sub0) + np.abs(sub1))) < MIN_MEAN_AMPLITUDE:
                continue

            phi_deg, lag, cval = compute_phase_difference(sub0, sub1, max_lag_samples=MAX_LAG_SAMPLES)
            if not np.isnan(phi_deg):
                center_sample_abs = s + (ls + le) / 2.0
                time_s = center_sample_abs / SAMPLE_RATE
                
                # <<< V10 MODIFICATION START: Apply CFO Correction >>>
                phase_correction_due_to_cfo = time_s * CFO_HZ_CORRECTION * 360.0
                corrected_phi_deg = phi_deg - phase_correction_due_to_cfo
                
                # Wrap the corrected phase to the [-180, 180] range
                wrapped_corrected_phi = np.mod(corrected_phi_deg + 180.0, 360.0) - 180.0
                # <<< V10 MODIFICATION END >>>

                corr_mags.append(abs(cval))
                # Store the corrected and wrapped phase
                corr_phases.append(wrapped_corrected_phi) 
                corr_lags.append(lag)
                subburst_times.append(time_s)

    print(f"\nAnalysis complete. Kept {len(corr_mags)} total high-quality sub-bursts for plotting.")
    
    # Both plotting functions will now use the corrected phase data
    plot_polar_db(corr_mags, corr_phases, POLAR_PATH)
    plot_correlation_summary_db(subburst_times, corr_mags, corr_phases, SUMMARY_PATH)


if __name__ == "__main__":
    main()