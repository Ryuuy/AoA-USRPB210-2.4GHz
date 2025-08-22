import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# import finufft # Removed as requested

# =============================================================================
#  Manual NDFT Function (Provided by User)
# =============================================================================
def manual_ndft(signal_x: np.ndarray, time_t: np.ndarray, freq_range: list, num_freq_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    直接计算非均匀离散傅里叶变换（NDFT）。

    该函数通过求和公式直接计算频谱，不使用FFT或插值技巧。
    
    Args:
        signal_x (np.ndarray): 包含信号幅度的复数数组，对应于每个时间点。
        time_t (np.ndarray): 包含非均匀采样时间点的浮点数数组。
                              其长度必须与 signal_x 相同。
        freq_range (list or np.ndarray): 包含频率范围的列表或数组，例如 [-5000, 5000]。
        num_freq_points (int): 频率轴上的抽头（计算点）数量。
    
    Returns:
        tuple: 包含两个 NumPy 数组的元组。
               第一个数组是计算出的频谱，第二个数组是对应的频率轴。
    """
    if len(signal_x) != len(time_t):
        raise ValueError("signal_x and time_t must have the same length.")

    frequencies = np.linspace(freq_range[0], freq_range[1], num_freq_points)
    
    # 使用 NumPy 的广播功能进行向量化计算，这比 for 循环更快
    # 指数部分的公式是 e^(-j * 2 * pi * f * t)
    exponent_matrix = np.exp(-1j * 2 * np.pi * frequencies.reshape(-1, 1) * time_t)
    
    spectrum = np.dot(exponent_matrix, signal_x)
    
    return spectrum, frequencies

# =============================================================================
#  CFO Estimation using Manual NDFT (New Function)
# =============================================================================
def estimate_cfo_manual_ndft(
    subburst_times: np.ndarray,
    corr_values: np.ndarray,
    num_freq_points: int = 16384 # 使用高分辨率以精确寻找峰值
) -> float:
    """
    通过手动计算高分辨率NDFT频谱来估算CFO。
    """
    print("\n--- Running CFO Estimation with Manual NDFT ---")
    
    if len(corr_values) < 2:
        print("Warning: Signal is too short for CFO estimation.")
        return 0.0
    
    # --- 1. 计算频率轴参数 ---
    time_span = subburst_times[-1] - subburst_times[0]
    if time_span <= 0:
        print("Warning: Time duration is zero or negative.")
        return 0.0
        
    num_points = len(subburst_times)
    # 等效采样率决定了无模糊频率范围
    equivalent_samplerate = num_points / time_span
    
    # 定义搜索峰值的频率范围
    freq_range = [-4, 0]
    
    print(f"Time span: {time_span:.4f}s, Points: {num_points}, Equivalent Fs: {equivalent_samplerate:.2f} Hz.")
    print(f"Frequency search range: [{freq_range[0]:.2f}, {freq_range[1]:.2f}] Hz.")
    print(f"Spectrum resolution points: {num_freq_points}")

    # --- 2. 执行手动NDFT ---
    spectrum, freq_axis = manual_ndft(
        signal_x=corr_values.astype(np.complex128),
        time_t=subburst_times,
        freq_range=freq_range,
        num_freq_points=num_freq_points
    )
    
    # --- 3. 寻找峰值频率 ---
    fft_magnitude = np.abs(spectrum)
    spectrum_db = 20 * np.log10(fft_magnitude + 1e-12)
    
    peak_index = np.argmax(spectrum_db)
    estimated_cfo = freq_axis[peak_index]

    print(f"Manual NDFT Complete. Estimated CFO = {estimated_cfo:.4f} Hz")

    # --- 4. 绘制结果频谱 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(freq_axis, spectrum_db - np.max(spectrum_db), '-') # 高分辨率频谱使用实线
    ax.axvline(x=estimated_cfo, color='r', linestyle='--', label=f'Peak: {estimated_cfo:.2f} Hz')
    
    ax.set_title('CFO Spectrum (Manual NDFT)', fontsize=16)
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('Normalized Power (dB)', fontsize=14)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("cfo_manual_ndft_spectrum.png", dpi=150)
    plt.show()
    plt.close(fig)

    return estimated_cfo

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
    """计算相位差"""
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
    """V2版本突发检测器"""
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
    """基于间隙的分割算法"""
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

def plot_correlation_summary_db(subburst_times, corr_mags, corr_phases, output_path, title_suffix=""):
    """生成摘要图"""
    if len(corr_mags) == 0:
        print("No correlation data to plot for summary.")
        return

    mags_db = 20.0 * np.log10(np.asarray(corr_mags) + 1e-18)
    phases = np.asarray(corr_phases)
    x_axis = np.asarray(subburst_times)
    
    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(f'Correlation Analysis Across Sub-Bursts{title_suffix}', fontsize=16)
    
    axs[0].plot(x_axis, mags_db, "o-")
    axs[0].set_ylabel("Correlation Magnitude (dB)")
    axs[0].grid(True)
    axs[0].set_title("Magnitude (dB) vs. Time")
    
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
    axs[1].set_title(f"Phase vs. Time{title_suffix}")
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Generated summary plot: {output_path}")

def plot_polar_db(corr_mags, corr_phases, output_path, title_suffix=""):
    """生成极坐标图"""
    if len(corr_mags) == 0:
        print("No correlation data to plot for polar.")
        return
        
    mags_lin = np.asarray(corr_mags)
    mags_db = 20.0 * np.log10(mags_lin + 1e-18)
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
    THEORETICAL_PHASE = 72
    axp.plot([np.deg2rad(THEORETICAL_PHASE)]*2, axp.get_ylim(), color='r', ls='--', lw=2, label=f'Theoretical LOS ({THEORETICAL_PHASE}°)')
    
    axp.set_title(f'Phase-Correlation Polar Plot{title_suffix} ({len(corr_mags)} points)', fontsize=16, pad=20)
    axp.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.1))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Generated polar plot: {output_path}")

# =============================================================================
#  Main Function (Modified)
# =============================================================================
def main():
    # --- Parameters ---
    FILENAME = "2ch_iq_data.bin"
    SAMPLE_RATE = 4e6
    
    COARSE_THRESHOLD_DB = 14.0
    COARSE_MERGE_GAP_US = 20.0
    MIN_BURST_US_COARSE = 5.0
    GAP_THRESHOLD = 0.05
    MIN_GAP_US = 5.0
    MIN_SUBBURST_US = 40.0
    MIN_MEAN_AMPLITUDE = 0.1
    MIN_STRONG_SAMPLES = 64
    MAX_LAG_SAMPLES = 1
    
    POLAR_PATH_UNCORRECTED = "correlation_polar_uncorrected.png"
    SUMMARY_PATH_UNCORRECTED = "correlation_summary_db_uncorrected.png"
    POLAR_PATH_CORRECTED = "correlation_polar_corrected.png"
    SUMMARY_PATH_CORRECTED = "correlation_summary_db_corrected.png"

    # --- Data Processing ---
    ch0, ch1 = read_2ch_iq(FILENAME)
    if ch0 is None:
        return

    # --- Step 1: Find all high-quality sub-bursts and gather their data ---
    print("\n--- Step 1: Gathering all correlation points ---")
    
    bursts0, _ = detect_signal_bursts_v2(ch0, sample_rate=SAMPLE_RATE, threshold_db=COARSE_THRESHOLD_DB, merge_gap_us=COARSE_MERGE_GAP_US, min_burst_us=MIN_BURST_US_COARSE)
    bursts1, _ = detect_signal_bursts_v2(ch1, sample_rate=SAMPLE_RATE, threshold_db=COARSE_THRESHOLD_DB, merge_gap_us=COARSE_MERGE_GAP_US, min_burst_us=MIN_BURST_US_COARSE)
    all_pairs = pair_bursts(bursts0, bursts1, max_center_diff=int(0.3e-3 * SAMPLE_RATE))
    
    subburst_times = []
    corr_values = []      # For NDFT (complex)
    corr_mags = []        # For plotting
    raw_corr_phases = []  # For plotting (uncorrected)

    for pidx, (b0, b1) in enumerate(all_pairs, 1):
        s, e = max(b0[0], b1[0]), min(b0[1], b1[1])
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
                
                subburst_times.append(time_s)
                corr_values.append(cval)
                corr_mags.append(abs(cval))
                raw_corr_phases.append(phi_deg)

    if not corr_values:
        print("No high-quality sub-bursts found. Exiting.")
        return
        
    print(f"Found {len(corr_values)} high-quality correlation points.")

    # --- Step 2: Plot the original, uncorrected data ---
    print("\n--- Step 2: Plotting uncorrected results ---")
    
    raw_phases_wrapped = [np.mod(p + 180, 360) - 180 for p in raw_corr_phases]
    plot_polar_db(corr_mags, raw_phases_wrapped, POLAR_PATH_UNCORRECTED, title_suffix=" (Uncorrected)")
    plot_correlation_summary_db(subburst_times, corr_mags, raw_phases_wrapped, SUMMARY_PATH_UNCORRECTED, title_suffix=" (Uncorrected)")
    
    # --- Step 3: Estimate CFO using the new Manual NDFT method ---
    # <<< THIS IS THE MODIFIED PART >>>
    estimated_cfo_hz = estimate_cfo_manual_ndft(
        np.array(subburst_times),
        np.array(corr_values)
    )
    
    # --- Step 4: Correct the phases and plot the corrected data ---
    print("\n--- Step 4: Applying CFO correction and plotting corrected results ---")
    
    corrected_phases = []
    for i, time_s in enumerate(subburst_times):
        phase_correction_due_to_cfo = time_s * estimated_cfo_hz * 360.0
        corrected_phi = raw_corr_phases[i] - phase_correction_due_to_cfo
        corrected_phases.append(np.mod(corrected_phi + 180, 360) - 180)

    plot_polar_db(corr_mags, corrected_phases, POLAR_PATH_CORRECTED, title_suffix=" (CFO Corrected)")
    plot_correlation_summary_db(subburst_times, corr_mags, corrected_phases, SUMMARY_PATH_CORRECTED, title_suffix=" (CFO Corrected)")

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()