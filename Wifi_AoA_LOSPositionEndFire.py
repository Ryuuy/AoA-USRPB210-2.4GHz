import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =============================================================================
#  SECTION 1: Core Signal Processing Functions (User Provided)
#  (This section remains unchanged as per your request)
# =============================================================================

def manual_ndft(signal_x: np.ndarray, time_t: np.ndarray, freq_range: list, num_freq_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """直接计算非均匀离散傅里叶变换（NDFT）。"""
    if len(signal_x) != len(time_t):
        raise ValueError("signal_x and time_t must have the same length.")
    frequencies = np.linspace(freq_range[0], freq_range[1], num_freq_points)
    exponent_matrix = np.exp(-1j * 2 * np.pi * frequencies.reshape(-1, 1) * time_t)
    spectrum = np.dot(exponent_matrix, signal_x)
    return spectrum, frequencies

def estimate_cfo_manual_ndft(
    subburst_times: np.ndarray,
    corr_values: np.ndarray,
    num_freq_points: int = 16384
) -> float:
    """通过手动计算高分辨率NDFT频谱来估算CFO。"""
    print("\n--- Running CFO Estimation with Manual NDFT ---")
    if len(corr_values) < 2: return 0.0
    time_span = subburst_times[-1] - subburst_times[0]
    if time_span <= 0: return 0.0
    num_points = len(subburst_times)
    equivalent_samplerate = num_points / time_span
    freq_range = [-3, 3]
    print(f"Time span: {time_span:.4f}s, Points: {num_points}, Equivalent Fs: {equivalent_samplerate:.2f} Hz.")
    print(f"Frequency search range: [{freq_range[0]:.2f}, {freq_range[1]:.2f}] Hz.")

    spectrum, freq_axis = manual_ndft(
        signal_x=corr_values.astype(np.complex128),
        time_t=subburst_times,
        freq_range=freq_range,
        num_freq_points=num_freq_points
    )
    
    peak_index = np.argmax(np.abs(spectrum))
    estimated_cfo = freq_axis[peak_index]
    print(f"Manual NDFT Complete. Estimated CFO = {estimated_cfo:.4f} Hz")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    spectrum_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
    ax.plot(freq_axis, spectrum_db - np.max(spectrum_db), '-')
    ax.axvline(x=estimated_cfo, color='r', linestyle='--', label=f'Peak: {estimated_cfo:.2f} Hz')
    ax.set_title('CFO Spectrum (Manual NDFT)', fontsize=16)
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('Normalized Power (dB)', fontsize=14)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("cfo_manual_ndft_spectrum.png", dpi=150)
    return estimated_cfo

def read_2ch_iq(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """从文件中读取双通道IQ数据"""
    try:
        data = np.fromfile(filename, dtype=np.complex64)
        return data[0::2], data[1::2]
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
        if best_b1: pairs.append((b0, best_b1))
    return pairs

def compute_phase_difference(s1: np.ndarray, s2: np.ndarray, max_lag_samples: int = 0) -> Tuple[float, int, complex]:
    """计算相位差"""
    assert len(s1) == len(s2)
    n_samples = len(s1)
    if n_samples == 0: return float("nan"), 0, 0j
    def corr_at_lag(lag: int) -> complex:
        if lag == 0: c = np.sum(s1 * np.conj(s2))
        elif lag > 0: c = np.sum(s1[lag:] * np.conj(s2[:-lag]))
        else: c = np.sum(s1[:lag] * np.conj(s2[-lag:]))
        effective_len = n_samples - abs(lag)
        return c / effective_len if effective_len > 0 else 0j
    if max_lag_samples <= 0:
        c0 = corr_at_lag(0)
        return float(np.rad2deg(np.angle(c0))), 0, c0
    best_lag, best_val, best_mag = 0, 0j, -1.0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        c = corr_at_lag(lag)
        m = abs(c)
        if m > best_mag: best_mag, best_val, best_lag = m, c, lag
    return float(np.rad2deg(np.angle(best_val))), best_lag, best_val

def detect_signal_bursts_v2(iq_data: np.ndarray, sample_rate: float, threshold_db: float = 10.0,
    hysteresis_db: float = 3.0, smoothing_ms: float = 0.2, min_burst_us: float = 100.0,
    merge_gap_us: float = 50.0, pre_guard_us: float = 0.0, post_guard_us: float = 0.0,
    noise_percentile: float = 20.0, edge_refine_fraction: float = 0.25,
    edge_search_ms: float = 0.1):
    """V2版本突发检测器"""
    if iq_data is None or len(iq_data) == 0: return [], {}
    power_lin = np.abs(iq_data) ** 2
    tau_samples = max(1, int(smoothing_ms * 1e-3 * sample_rate))
    beta = 1.0 / float(tau_samples)
    b, a = [beta], [1.0, -(1.0 - beta)]
    envelope_lin = signal.lfilter(b, a, power_lin)
    noise_floor_lin = np.percentile(envelope_lin, noise_percentile)
    if noise_floor_lin <= 0: noise_floor_lin = 1e-18
    enter_thr_lin = noise_floor_lin * (10.0 ** (threshold_db / 10.0))
    exit_thr_lin = noise_floor_lin * (10.0 ** ((threshold_db - hysteresis_db) / 10.0))
    N, bursts, in_burst, start_idx = len(envelope_lin), [], False, 0
    for i, val in enumerate(envelope_lin):
        if not in_burst and val >= enter_thr_lin: in_burst, start_idx = True, i
        elif in_burst and val < exit_thr_lin:
            in_burst = False
            bursts.append([start_idx, i - 1])
    if in_burst: bursts.append([start_idx, N - 1])
    if not bursts and np.all(envelope_lin >= enter_thr_lin): bursts = [[0, N - 1]]
    merge_gap_samples = int(max(0, merge_gap_us) * 1e-6 * sample_rate)
    merged = []
    if bursts:
        merged.append(bursts[0])
        for bseg in bursts[1:]:
            if bseg[0] - merged[-1][1] <= merge_gap_samples: merged[-1][1] = max(merged[-1][1], bseg[1])
            else: merged.append(bseg)
    bursts = merged
    min_len_samples = int(max(0, min_burst_us) * 1e-6 * sample_rate)
    filtered = [[s, e] for s, e in bursts if e - s + 1 >= min_len_samples]
    return filtered, {}

def split_frame_by_gaps(seg0: np.ndarray, seg1: np.ndarray, sample_rate: float,
    gap_threshold: float, min_gap_us: float, min_subburst_us: float) -> List[Tuple[int, int]]:
    """基于间隙的分割算法"""
    if len(seg0) == 0: return []
    is_gap = np.minimum(np.abs(seg0), np.abs(seg1)) < gap_threshold
    min_gap_samples = int(min_gap_us * 1e-6 * sample_rate)
    if min_gap_samples < 1: min_gap_samples = 1
    cut_points, in_gap, gap_start = [0], False, 0
    for i, g in enumerate(is_gap):
        if g and not in_gap: in_gap, gap_start = True, i
        elif not g and in_gap:
            in_gap = False
            if i - gap_start >= min_gap_samples: cut_points.append(gap_start + (i - gap_start) // 2)
    cut_points.append(len(seg0))
    min_subburst_samples = int(min_subburst_us * 1e-6 * sample_rate)
    subbursts = []
    cut_points = sorted(list(set(cut_points)))
    for i in range(len(cut_points) - 1):
        s, e = cut_points[i], cut_points[i+1] - 1
        if (e - s + 1) >= min_subburst_samples: subbursts.append((s, e))
    return subbursts

# =============================================================================
#  SECTION 2: AoA Specific Functions (Upgraded and Modified)
# =============================================================================

def calculate_fine_corrections(
    subburst_times: np.ndarray,
    corr_mags: np.ndarray, 
    coarse_corrected_unwrapped_phases: np.ndarray, 
    theoretical_phase_deg: float
) -> Tuple[float, float]:
    """
    【融合版】一个函数完成两件事：
    1. 通过分析LOS簇的相位斜率，计算出精细的残余CFO。
    2. 在应用了精细CFO修正后，再用高斯加权法计算最终的相位偏移(offset)。
    """
    print("\n--- Step 3: Calculating Fine Corrections (Residual CFO & Phase Offset) ---")
    if len(corr_mags) < 10: # 需要足够的数据点
        print("Warning: Not enough points for fine correction. Skipping.")
        return 0.0, 0.0

    # --- Part 1: 计算残余CFO (Residual CFO) ---

    # 步骤 1a: 使用5dB规则筛选出LOS候选点
    corr_mags_db = 20 * np.log10(corr_mags + 1e-12)
    max_mag_db = np.max(corr_mags_db)
    high_power_mask = (corr_mags_db >= max_mag_db - 3.0)
    candidate_indices = np.where(high_power_mask)[0]
    
    N = len(candidate_indices)
    print(f"Found N={N} high-power points for analysis.")

    # 步骤 1b: 检查是否有足够的点来计算斜率 (前后各5个点，至少需要10个)
    if N < 10:
        print("Warning: Not enough high-power points (< 10) for slope refinement. Residual CFO set to 0.")
        residual_cfo_hz = 0.0
    else:
        # 提取高能量点对应的时间和未包裹相位
        los_times = subburst_times[candidate_indices]
        los_phases = coarse_corrected_unwrapped_phases[candidate_indices]
        
        # 计算前后5个点的平均时间和平均相位
        t_start = np.mean(los_times[:8])
        phi_start = np.mean(los_phases[:8])
        t_end = np.mean(los_times[-8:])
        phi_end = np.mean(los_phases[-8:])
        print(t_start,phi_start,t_end,phi_end)
        delta_phi_deg = phi_end - phi_start
        delta_phi_deg = (delta_phi_deg + 180 + 360) % 360 - 180
        delta_t_s = t_end - t_start
        print(delta_phi_deg,delta_t_s)
        if delta_t_s < 1e-6:
            print("Warning: Time difference is too small. Residual CFO set to 0.")
            residual_cfo_hz = 0.0
        else:
            # 频率 = 相位变化 / 时间变化。因为相位是度，所以要除以360
            residual_cfo_hz = (delta_phi_deg / delta_t_s) / 360.0
            print(f"Calculated residual CFO from slope: {residual_cfo_hz:.4f} Hz")

    # --- Part 2: 计算最终的相位偏移 (Phase Offset) ---

    # 步骤 2a: 应用刚刚算出的残余CFO，得到完全校正的相位
    residual_phase_correction = subburst_times * residual_cfo_hz * 360.0
    fully_corrected_phases = coarse_corrected_unwrapped_phases - residual_phase_correction
    
    # 步骤 2b: 在完全校正的相位上，运行高斯加权算法来找offset
    # 注意：候选点还是之前根据幅度选出的那些，但现在它们的相位值更新了
    candidate_fully_corrected_phases = fully_corrected_phases[candidate_indices]
    candidate_mags = corr_mags[candidate_indices]
    
    # 找到绝对最强点，用它最新的相位作为高斯中心
    anchor_idx = np.argmax(corr_mags)
    anchor_phase = fully_corrected_phases[anchor_idx]
    print("candidate_fully_corrected_phases",candidate_fully_corrected_phases,len(candidate_fully_corrected_phases))
    sigma_deg = 360.0 / (2 * 1.96)
    phase_diff = (candidate_fully_corrected_phases - anchor_phase + 180) % 360 - 180
    gaussian_weights = np.exp(-0.5 * (phase_diff / sigma_deg)**2)
    
    # 用更新后的相位计算相量
    candidate_phasors = candidate_mags * np.exp(1j * np.deg2rad(candidate_fully_corrected_phases))
    
    mean_phasor = np.sum(gaussian_weights[:, np.newaxis] * candidate_phasors) / np.sum(gaussian_weights)
    cluster_center_phase_deg = np.rad2deg(np.angle(mean_phasor))
    print("cluster_center_phase_deg",cluster_center_phase_deg)
    offset = cluster_center_phase_deg - theoretical_phase_deg
    
    print(f"Final Gaussian weighted cluster center: {cluster_center_phase_deg:.2f}°.")
    print(f"Final calculated phase offset to be subtracted: {offset:.2f}°")
    # plt.figure(figsize=(12, 8))
    # plt.subplot(2, 1, 1)
    # plt.plot(coarse_corrected_unwrapped_phases, 'b-', label='Coarse Corrected Phases')
    # plt.title('Coarse Corrected Unwrapped Phases')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Phase (degrees)')
    # plt.grid(True)
    # plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.plot(fully_corrected_phases, 'r-', label='Fully Corrected Phases')
    # plt.title('Fully Corrected Phases')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Phase (degrees)')
    # plt.grid(True)
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
    return fully_corrected_phases-offset


def phase_to_aoa_broadside(corrected_phases_deg: np.ndarray, d_antenna: float, frequency_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    【升级版】将校准后的相位差数组转换为到达角(AoA)数组。
    同时返回arcsin的参数，用于后续筛选。
    """
    c = 299792458.0
    lambda_wave = c / frequency_hz
    corrected_phases_rad = np.deg2rad(corrected_phases_deg)
    print("corrected_phases_rad",np.min(corrected_phases_rad),np.max(corrected_phases_rad))

    # 计算 arcsin 的参数，但不进行裁剪
    sin_theta_arg = corrected_phases_rad * lambda_wave / (2 * np.pi * d_antenna)
    print("sin_theta_arg",np.min(sin_theta_arg),np.max(sin_theta_arg))
    # 对有效范围内的参数计算AoA，无效部分暂时标记为nan
    # 这样做可以避免裁剪，同时保留原始信息
    valid_mask = np.abs(sin_theta_arg) <= 1.0
    aoa_deg = np.full_like(sin_theta_arg, np.nan) # 创建一个填满nan的数组
    
    aoa_rad_valid = np.arcsin(sin_theta_arg[valid_mask])
    aoa_deg[valid_mask] = np.rad2deg(aoa_rad_valid)
    
    return aoa_deg, sin_theta_arg

def plot_calibrated_phase_polar(corr_mags, corr_phases, output_path, title_suffix=""):
    """生成相位校准后的极坐标图 (范围 -pi 到 pi)。"""
    if len(corr_mags) == 0: return
    mags_lin = np.asarray(corr_mags)
    mags_db = 20.0 * np.log10(mags_lin + 1e-18)
    
    # 输入的 corr_phases 已经被包裹在[-180, 180], 这里转换为[-pi, pi]
    theta_rad = np.deg2rad(np.asarray(corr_phases))

    fig = plt.figure(figsize=(10, 10))
    axp = fig.add_subplot(111, projection='polar')
    sc = axp.scatter(theta_rad, mags_db, c=mags_lin, cmap='viridis', s=30, alpha=0.8)
    cbar = fig.colorbar(sc, pad=0.1, shrink=0.8)
    cbar.set_label('|Correlation| (Linear Scale)', fontsize=12)
    
    axp.annotate('LOS Calibrated to 0', xy=(0, np.max(mags_db)), xytext=(np.pi/4, np.max(mags_db) + 10),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2),
                 ha='center', va='center', fontsize=12, color='red')
                 
    # --- MODIFIED PART ---
    # 1. 设置我们想要显示的刻度标签
    angle_ticks_rad = np.array([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4])
    angle_tick_labels = [r'$-\pi$', r'$-3\pi/4$', r'$-\pi/2$', r'$-\pi/4$', '0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$']
    axp.set_xticks(angle_ticks_rad)
    axp.set_xticklabels(angle_tick_labels)

    # 2. 【关键修复】明确设置极坐标的角度范围为 -pi 到 pi，以显示完整的圆
    axp.set_thetalim(-np.pi, np.pi)
    # -------------------
    
    axp.set_title(f'Phase-Correlation Polar Plot{title_suffix} ({len(corr_mags)} points)', fontsize=16, pad=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Generated calibrated phase plot: {output_path}")

def plot_aoa_pattern(corr_mags, aoa_degrees, output_path, model_type='broadside'):
    """
    【最终版】生成AoA方向图，能自动适应Broadside和End-fire模型。
    - 对于End-fire，它会将0°（LOS方向）放在图的正右方，并显示[0, 180°]的范围。
    """
    if len(corr_mags) == 0: return
    fig = plt.figure(figsize=(10, 10))
    ax_aoa = fig.add_subplot(111, projection='polar')
    
    sc = ax_aoa.scatter(np.deg2rad(aoa_degrees), corr_mags, c=corr_mags, cmap='viridis', s=30)
    
    cbar = fig.colorbar(sc, pad=0.1, shrink=0.8)
    cbar.set_label('|Correlation| (Linear Scale)', fontsize=12)
    ax_aoa.set_xlabel("Angle of Arrival (°)")
    ax_aoa.grid(True)
    
    # 根据模型类型，自动调整图表的坐标系和标题
    if model_type.lower() == 'endfire':
        # --- End-fire 专用绘图设置 ---
        # 将图表的 '0°' 刻度放在正右方 (East)，这代表天线的轴线方向 (LOS方向)
        ax_aoa.set_theta_zero_location('E') 
        
        # 设置角度从0度开始，逆时针增加 (标准数学表示)
        ax_aoa.set_theta_direction(1)      
        
        # 将图表的显示范围设置为 [0, 180度] (半圆)
        ax_aoa.set_thetalim(0, np.pi)      
        
        ax_aoa.set_title(f'Angle of Arrival (End-fire) ({len(aoa_degrees)} points)', fontsize=16, pad=20)
    else: 
        # --- Broadside 专用绘图设置 ---
        ax_aoa.set_theta_zero_location('N') # 将 '0°' 刻度放在正上方 (North)
        ax_aoa.set_theta_direction(-1)     # 顺时针为正角度
        ax_aoa.set_thetalim(-np.pi/2, np.pi/2) # 设置范围为 [-90, +90度]
        ax_aoa.set_title(f'Angle of Arrival (Broadside) ({len(aoa_degrees)} points)', fontsize=16, pad=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Generated AoA pattern plot: {output_path}")

def phase_to_aoa_endfire(corrected_phases_deg: np.ndarray, d_antenna: float, frequency_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    【End-fire版】将校准后的相位差数组转换为到达角(AoA)数组。
    """
    c = 299792458.0
    lambda_wave = c / frequency_hz
    corrected_phases_rad = np.deg2rad(corrected_phases_deg)

    max_phase_rad = (2 * np.pi * d_antenna) / lambda_wave
    cos_theta_arg = corrected_phases_rad / max_phase_rad
    
    valid_mask = np.abs(cos_theta_arg) <= 1.0
    aoa_deg = np.full_like(cos_theta_arg, np.nan)
    
    aoa_rad_valid = np.arccos(cos_theta_arg[valid_mask])
    aoa_deg[valid_mask] = np.rad2deg(aoa_rad_valid)
    
    return aoa_deg, cos_theta_arg

# =============================================================================
#  SECTION 3: Main Execution
# =============================================================================
def main():
    # --- Experiment Setup and Parameters ---
    FILENAME = "2ch_iq_data.bin"
    SAMPLE_RATE = 4e6
    ANTENNA_SPACING_M = 0.026
    CENTER_FREQ_HZ = 2.455e9
    THEORETICAL_PHASE_DEG = 76.6

    # --- Signal Detection Parameters ---
    COARSE_THRESHOLD_DB = 14.0
    COARSE_MERGE_GAP_US = 20.0
    MIN_BURST_US_COARSE = 5.0
    GAP_THRESHOLD = 0.05
    MIN_GAP_US = 5.0
    MIN_SUBBURST_US = 40.0
    MIN_MEAN_AMPLITUDE = 0.1
    MIN_STRONG_SAMPLES = 64
    MAX_LAG_SAMPLES = 1

    # --- Step 1: Read Data and Extract Correlation Points ---
    print("\n--- Step 1: Reading IQ data and finding correlation points ---")
    ch0, ch1 = read_2ch_iq(FILENAME)
    if ch0 is None: return

    bursts0, _ = detect_signal_bursts_v2(ch0, sample_rate=SAMPLE_RATE, threshold_db=COARSE_THRESHOLD_DB, merge_gap_us=COARSE_MERGE_GAP_US, min_burst_us=MIN_BURST_US_COARSE)
    bursts1, _ = detect_signal_bursts_v2(ch1, sample_rate=SAMPLE_RATE, threshold_db=COARSE_THRESHOLD_DB, merge_gap_us=COARSE_MERGE_GAP_US, min_burst_us=MIN_BURST_US_COARSE)
    all_pairs = pair_bursts(bursts0, bursts1, max_center_diff=int(0.3e-3 * SAMPLE_RATE))
    
    subburst_times, corr_values = [], []
    for pidx, (b0, b1) in enumerate(all_pairs, 1):
        s, e = max(b0[0], b1[0]), min(b0[1], b1[1])
        seg0, seg1 = ch0[s:e+1], ch1[s:e+1]
        subbursts = split_frame_by_gaps(seg0, seg1, sample_rate=SAMPLE_RATE, gap_threshold=GAP_THRESHOLD, min_gap_us=MIN_GAP_US, min_subburst_us=MIN_SUBBURST_US)
        for ls, le in subbursts:
            sub0, sub1 = seg0[ls:le+1], seg1[ls:le+1]
            if len(sub0) < MIN_STRONG_SAMPLES or np.mean(0.5 * (np.abs(sub0) + np.abs(sub1))) < MIN_MEAN_AMPLITUDE: continue
            phi_deg, lag, cval = compute_phase_difference(sub0, sub1, max_lag_samples=MAX_LAG_SAMPLES)
            if not np.isnan(phi_deg):
                subburst_times.append((s + (ls + le) / 2.0) / SAMPLE_RATE)
                corr_values.append(cval)

    if not corr_values:
        print("No high-quality sub-bursts found. Exiting.")
        return
        
    subburst_times = np.array(subburst_times)
    corr_values = np.array(corr_values)
    corr_mags = np.abs(corr_values)
    raw_corr_phases = np.rad2deg(np.angle(corr_values))
    print(f"Found {len(corr_values)} high-quality correlation points.")

# --- Step 2: Coarse CFO Estimation and Correction ---
    estimated_cfo_hz = estimate_cfo_manual_ndft(subburst_times, corr_values)
    
    # 得到粗略校正后的、未包裹的相位
    phase_correction_due_to_cfo = subburst_times * -1.78 * 360.0
    coarse_corrected_unwrapped_phases = raw_corr_phases - phase_correction_due_to_cfo
    print("coarse_corrected_unwrapped_phases",np.max(coarse_corrected_unwrapped_phases))
    coarse_corrected_unwrapped_phases = (coarse_corrected_unwrapped_phases + 180) % 360 - 180
    print("coarse_corrected_unwrapped_phases",np.max(coarse_corrected_unwrapped_phases))
    # --- Step 3: Fine Corrections (Residual CFO & Phase Offset) ---
    # 调用我们融合后的新函数
    final_calibrated_wrapped = calculate_fine_corrections(
        subburst_times,
        corr_mags,
        coarse_corrected_unwrapped_phases,
        THEORETICAL_PHASE_DEG
    )


    # --- Step 5: Plot Final Calibrated Phase Polar Diagram ---
    plot_calibrated_phase_polar(
        corr_mags, 
        final_calibrated_wrapped, 
        "correlation_polar_phase_calibrated.png",
        title_suffix=" (Phase Calibrated to 0)"
    )

    # --- Step 6: Calculate AoA and Filter Invalid Points ---
    print("\n--- Step 6: Calculating AoA and filtering invalid points ---")
    aoa_degrees, sin_theta_arg = phase_to_aoa_broadside(final_calibrated_wrapped, ANTENNA_SPACING_M, CENTER_FREQ_HZ)
    
    valid_mask = ~np.isnan(aoa_degrees)
    valid_aoa = aoa_degrees[valid_mask]
    valid_mags = corr_mags[valid_mask]
    print(valid_aoa)
    num_total = len(aoa_degrees)
    num_valid = len(valid_aoa)
    print(f"Total points: {num_total}, Physically valid points for AoA: {num_valid} ({(num_valid/num_total)*100:.1f}%)")
    print(f"Filtered out {num_total - num_valid} points with phase differences too large for the broadside model.")
    
    # --- Step 7: Plot the Filtered, Valid AoA Pattern ---
# ... main函数前面的所有代码保持您现有的不变 ...

    # --- Step 7: Plot the Filtered, Valid AoA Pattern ---
    plot_aoa_pattern(valid_mags, valid_aoa, "aoa_pattern_endfire.png", model_type='endfire')

    print("\nProcessing complete.")
if __name__ == "__main__":
    main()