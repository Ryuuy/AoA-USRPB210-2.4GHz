import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

def inspect_iq_file(filename, sample_rate, num_channels=2, dtype=np.complex64):
    """检查IQ数据文件的大小、样本总数和录制时长，并返回总样本数。"""
    print(f"--- 文件属性检查: {filename} ---")
    try:
        file_size_bytes = os.path.getsize(filename)
        print(f"文件大小: {file_size_bytes / 1e6:.2f} MB")
        bytes_per_sample_per_channel = np.dtype(dtype).itemsize
        total_samples_per_channel = file_size_bytes / (bytes_per_sample_per_channel * num_channels)
        if total_samples_per_channel != int(total_samples_per_channel):
             print("警告: 文件大小与样本格式不完全匹配，可能文件不完整。")
        total_samples_per_channel = int(total_samples_per_channel)
        duration_seconds = total_samples_per_channel / sample_rate
        print(f"每个通道的样本数: {total_samples_per_channel}")
        print(f"预估录制时长: {duration_seconds:.2f} 秒")
        print("------------------------------------\n")
        return total_samples_per_channel
    except FileNotFoundError:
        print(f"错误: 文件 '{filename}' 未找到。")
        return 0

def read_2ch_iq(filename, num_channels=2):
    """从一个标准的双通道交错IQ二进制文件中读取数据。"""
    try:
        raw_data = np.fromfile(filename, dtype=np.complex64)
        if raw_data.size == 0: return None, None
        reshaped_data = raw_data.reshape(-1, num_channels)
        return reshaped_data[:, 0], reshaped_data[:, 1]
    except FileNotFoundError: return None, None

def detect_signal_bursts(iq_data, threshold_db=10, window_len=1000, debounce_len=5000):
    """使用移动平均功率检测IQ数据中的信号脉冲。"""
    power = np.abs(iq_data)**2
    mov_avg_power = np.convolve(power, np.ones(window_len)/window_len, mode='same')
    noise_floor = np.median(mov_avg_power)
    threshold = noise_floor * (10**(threshold_db/10))
    detections = (mov_avg_power > threshold)
    diff = np.diff(detections.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    if len(starts) == 0 and len(ends) == 0 and np.all(detections):
        return [[0, len(iq_data)]]
    if len(starts) > len(ends):
        ends = np.append(ends, len(iq_data) - 1)
    if len(starts) < len(ends):
        starts = np.insert(starts, 0, 0)
    
    bursts = []
    if len(starts) > 0:
        last_end = starts[0]
        for i in range(len(starts)):
            if starts[i] - last_end < debounce_len:
                if len(bursts) > 0:
                    bursts[-1][1] = ends[i]
                else:
                    bursts.append([starts[i], ends[i]])
            else:
                bursts.append([starts[i], ends[i]])
            last_end = ends[i]
    return bursts

def detect_signal_bursts_v2(
    iq_data: np.ndarray,
    sample_rate: float,
    threshold_db: float = 10.0,
    hysteresis_db: float = 3.0,
    smoothing_ms: float = 0.2,
    min_burst_us: float = 100.0,
    merge_gap_us: float = 50.0,
    pre_guard_us: float = 0.0,
    post_guard_us: float = 0.0,
    noise_percentile: float = 20.0,
    edge_refine_fraction: float = 0.25,
    edge_search_ms: float = 0.1,
):
    """
    Robust power-based burst detector with causal smoothing, hysteresis, and optional edge refinement.

    Parameters
    - threshold_db: detection threshold above noise floor (in dB)
    - hysteresis_db: exit threshold is lower than enter threshold by this amount (in dB)
    - smoothing_ms: EMA time-constant for envelope smoothing (milliseconds)
    - min_burst_us: discard bursts shorter than this duration (microseconds)
    - merge_gap_us: merge neighboring bursts separated by smaller gaps than this (microseconds)
    - pre_guard_us/post_guard_us: extend each kept burst by this margin (microseconds)
    - noise_percentile: percentile of the smoothed envelope used as noise floor estimate
    - edge_refine_fraction: for optional edge tuning, the level between noise and in-burst median
    - edge_search_ms: search range for edge refinement (milliseconds)

    Returns
    - bursts: List[[start_idx, end_idx]]
    - debug: Dict with keys 'envelope_db', 'enter_threshold_db', 'exit_threshold_db', 'noise_floor_db'
    """
    if iq_data is None or len(iq_data) == 0:
        return [], {
            "envelope_db": np.array([]),
            "enter_threshold_db": None,
            "exit_threshold_db": None,
            "noise_floor_db": None,
        }

    power_lin = np.abs(iq_data) ** 2

    # Causal EMA smoothing via IIR filter (vectorized, fast)
    tau_samples = max(1, int(smoothing_ms * 1e-3 * sample_rate))
    beta = 1.0 / float(tau_samples)  # approximately 1/tau
    b = [beta]
    a = [1.0, -(1.0 - beta)]
    envelope_lin = signal.lfilter(b, a, power_lin)

    # Robust noise floor
    noise_floor_lin = np.percentile(envelope_lin, noise_percentile)
    noise_floor_lin = max(noise_floor_lin, 1e-18)

    # Two-threshold hysteresis
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

    # If always above threshold, return whole range
    if len(bursts) == 0 and np.all(envelope_lin >= enter_thr_lin):
        bursts = [[0, N - 1]]

    # Merge nearby bursts
    merge_gap_samples = int(max(0, merge_gap_us) * 1e-6 * sample_rate)
    merged: list[list[int]] = []
    for bseg in bursts:
        if not merged:
            merged.append(bseg)
        else:
            last = merged[-1]
            if bseg[0] - last[1] <= merge_gap_samples:
                last[1] = max(last[1], bseg[1])
            else:
                merged.append(bseg)
    bursts = merged

    # Min length filter and guards
    min_len_samples = int(max(0, min_burst_us) * 1e-6 * sample_rate)
    pre_guard = int(max(0, pre_guard_us) * 1e-6 * sample_rate)
    post_guard = int(max(0, post_guard_us) * 1e-6 * sample_rate)

    filtered: list[list[int]] = []
    for s, e in bursts:
        if e - s + 1 >= min_len_samples:
            s2 = max(0, s - pre_guard)
            e2 = min(N - 1, e + post_guard)
            filtered.append([s2, e2])
    bursts = filtered

    # Optional edge refinement to reduce extra leading/trailing samples
    edge_search_samples = int(max(0, edge_search_ms) * 1e-3 * sample_rate)
    if bursts and 0.0 < edge_refine_fraction < 1.0:
        refined: list[list[int]] = []
        for s, e in bursts:
            seg = power_lin[max(0, s - edge_search_samples):min(N, e + edge_search_samples + 1)]
            offset = max(0, s - edge_search_samples)
            if len(seg) == 0:
                refined.append([s, e])
                continue
            inburst_med = float(np.median(power_lin[s:e+1])) if e >= s else float(np.median(seg))
            target = noise_floor_lin + edge_refine_fraction * (inburst_med - noise_floor_lin)

            # Left refine
            left_cond = seg > target
            left_idxs = np.flatnonzero(left_cond)
            if left_idxs.size > 0:
                s_ref = offset + int(left_idxs[0])
            else:
                s_ref = s

            # Right refine
            right_idxs = left_idxs
            if right_idxs.size > 0:
                e_ref = offset + int(right_idxs[-1])
            else:
                e_ref = e

            refined.append([max(0, s_ref), min(N - 1, e_ref)])
        bursts = refined

    debug = {
        "envelope_db": 10.0 * np.log10(envelope_lin + 1e-18),
        "enter_threshold_db": 10.0 * np.log10(enter_thr_lin + 1e-18),
        "exit_threshold_db": 10.0 * np.log10(exit_thr_lin + 1e-18),
        "noise_floor_db": 10.0 * np.log10(noise_floor_lin + 1e-18),
    }
    return bursts, debug

def pair_bursts(bursts0, bursts1, max_center_diff=1000):
    """智能匹配两个通道的脉冲列表。"""
    pairs = []
    candidates1 = list(bursts1)
    for b0_start, b0_end in bursts0:
        midpoint0 = b0_start + (b0_end - b0_start) / 2
        best_match = None
        smallest_diff = float('inf')
        for i, (b1_start, b1_end) in enumerate(candidates1):
            midpoint1 = b1_start + (b1_end - b1_start) / 2
            diff = abs(midpoint0 - midpoint1)
            if diff < max_center_diff and diff < smallest_diff:
                smallest_diff = diff
                best_match = (i, [b1_start, b1_end])
        if best_match is not None:
            match_index, b1 = best_match
            pairs.append( ([b0_start, b0_end], b1) )
            del candidates1[match_index]
    return pairs

def analyze_burst_full(ch0_burst, ch1_burst, sample_rate, center_freq, antenna_spacing_m, min_len_samples: int = 2048):
    """
    方法A: 对完整的脉冲数据进行分析。
    返回一个包含6个分析结果的元组。
    """
    min_len = min(len(ch0_burst), len(ch1_burst))
    if min_len < min_len_samples:
        return (np.nan,)*6
        
    s1 = ch0_burst[:min_len]
    s2 = ch1_burst[:min_len]
    
    correlation = signal.correlate(s1, s2, mode='full')
    lags = signal.correlation_lags(len(s1), len(s2), mode='full')
    delay_in_samples = lags[np.argmax(np.abs(correlation))]

    instant_phase_diff_rad = np.unwrap(np.angle(s1 * np.conj(s2)))
    sample_indices = np.arange(len(instant_phase_diff_rad))
    poly_coeffs = np.polyfit(sample_indices, instant_phase_diff_rad, 1)
    fit_line = np.polyval(poly_coeffs, sample_indices)
    
    detrended_phase_rad = instant_phase_diff_rad - fit_line
    static_phase_rad = poly_coeffs[1]
    
    c = 299792458.0
    wavelength = c / center_freq
    arg = (static_phase_rad * wavelength) / (2 * np.pi * antenna_spacing_m)
    aoa_deg = np.nan
    if abs(arg) <= 1:
        aoa_deg = np.rad2deg(np.arcsin(arg))
        
    return (delay_in_samples, instant_phase_diff_rad, fit_line, 
            detrended_phase_rad, static_phase_rad, aoa_deg)

def analyze_burst_golden_segment(ch0_burst, ch1_burst, sample_rate, center_freq, antenna_spacing_m, min_len_samples: int = 2048):
    """
    方法B: 使用两遍分析法，先找到“黄金信号段”，再对其进行分析。
    """
    power_envelope = np.convolve(np.abs(ch0_burst)**2, np.ones(200)/200, mode='same')
    peak_power = np.max(power_envelope)
    high_snr_mask = (power_envelope > peak_power * 0.5)
    high_snr_indices = np.where(high_snr_mask)[0]
    
    if len(high_snr_indices) < min_len_samples:
        return (np.nan,)*6, None
        
    golden_start, golden_end = high_snr_indices[0], high_snr_indices[-1]
    
    iq0_golden = ch0_burst[golden_start:golden_end]
    iq1_golden = ch1_burst[golden_start:golden_end]
    
    results = analyze_burst_full(
        iq0_golden,
        iq1_golden,
        sample_rate,
        center_freq,
        antenna_spacing_m,
        min_len_samples=min_len_samples,
    )
    
    return results, (golden_start, golden_end)