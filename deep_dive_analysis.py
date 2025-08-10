import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sdr_analysis_tools import (inspect_iq_file, 
                                read_2ch_iq, 
                                detect_signal_bursts,
                                pair_bursts,
                                analyze_burst_full,
                                analyze_burst_golden_segment)

def create_and_show_comparison_plot(full_burst_results, golden_segment_results,
                                    iq0_full, iq1_full, golden_indices,
                                    config, burst_num_pair):
    """
    为单个脉冲对，创建包含两种方法结果的对比图，并直接显示。
    """
    fig, axs = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(f'Analysis Comparison for Burst Pair #{burst_num_pair}', fontsize=20)

    # --- 数据解包（方法A始终解包；方法B按条件解包） ---
    delay_full, raw_phase_full, fit_full, detrended_full, static_phase_full_rad, aoa_full = full_burst_results
    
    # ================= TOP ROW: Method A (Full Burst Analysis) =================
    phase_deg_full = np.rad2deg(static_phase_full_rad)
    axs[0, 0].set_title(f'Method A: Full Burst (Phase: {phase_deg_full:.2f} deg, AoA: {aoa_full:.2f} deg)')
    
    # 图A1: 时间域功率包络与一次分割门限
    env_full = np.convolve(np.abs(iq0_full)**2, np.ones(200)/200, mode='same')
    env_full_db = 10*np.log10(env_full + 1e-12)
    noise_floor_lin = np.median(env_full)
    thresh_db = 10*np.log10(noise_floor_lin + 1e-12) + config['detection_threshold_db']
    axs[0, 0].plot(env_full_db, label='Power Env (dBFS)')
    axs[0, 0].axhline(thresh_db, color='r', linestyle='--', label=f'Thresh: +{config["detection_threshold_db"]} dB')
    axs[0, 0].set_xlabel('Sample Index'); axs[0, 0].set_ylabel('Power (dBFS)'); axs[0, 0].grid(True); axs[0, 0].legend()
    
    # 图A2: 原始相位斜坡
    axs[0, 1].plot(np.rad2deg(raw_phase_full), label='Raw Phase')
    axs[0, 1].plot(np.rad2deg(fit_full), 'r--', label='Linear Fit')
    axs[0, 1].set_xlabel('Sample Index'); axs[0, 1].set_ylabel('Phase (Deg)'); axs[0, 1].grid(True); axs[0, 1].legend()

    # 图A3: 去趋势后的相位
    axs[0, 2].plot(np.rad2deg(detrended_full))
    axs[0, 2].axhline(0, color='r', linestyle='--')
    axs[0, 2].set_xlabel('Sample Index'); axs[0, 2].set_ylabel('Phase Fluctuation (Deg)'); axs[0, 2].grid(True)

    # ================= BOTTOM ROW: Method B (Golden Segment Analysis 预览) =================
    if golden_segment_results is not None and golden_indices is not None and np.isfinite(golden_segment_results[4]):
        delay_golden, raw_phase_golden, fit_golden, detrended_golden, static_phase_golden_rad, aoa_golden = golden_segment_results
        phase_deg_golden = np.rad2deg(static_phase_golden_rad)
        axs[1, 0].set_title(f'Method B: Golden Segment (Phase: {phase_deg_golden:.2f} deg, AoA: {aoa_golden:.2f} deg)')

        # 图B1: 时间域功率包络与黄金段高亮
        axs[1, 0].plot(env_full_db, label='Power Env (dBFS)')
        axs[1, 0].axvspan(golden_indices[0], golden_indices[1], color='g', alpha=0.3, label='Golden Segment')
        axs[1, 0].axhline(thresh_db, color='r', linestyle='--', label='Thresh')
        axs[1, 0].set_xlabel('Sample Index'); axs[1, 0].set_ylabel('Power (dBFS)'); axs[1, 0].grid(True); axs[1, 0].legend()

        # 图B2: 黄金段的相位斜坡
        axs[1, 1].plot(np.rad2deg(raw_phase_golden), label='Raw Phase')
        axs[1, 1].plot(np.rad2deg(fit_golden), 'r--', label='Linear Fit')
        axs[1, 1].set_xlabel('Sample Index (in Golden Segment)'); axs[1, 1].set_ylabel('Phase (Deg)'); axs[1, 1].grid(True); axs[1, 1].legend()

        # 图B3: 黄金段去趋势后的相位
        axs[1, 2].plot(np.rad2deg(detrended_golden))
        axs[1, 2].axhline(0, color='r', linestyle='--')
        axs[1, 2].set_xlabel('Sample Index (in Golden Segment)'); axs[1, 2].set_ylabel('Phase Fluctuation (Deg)'); axs[1, 2].grid(True)
    else:
        for j in range(3):
            axs[1, j].text(0.5, 0.5, 'Golden Segment analysis failed\n(burst too short or noisy)', 
                           ha='center', va='center', fontsize=14, color='red')
            axs[1, j].set_xticks([]); axs[1, j].set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- 核心修改：直接显示图像并等待用户关闭 ---
    plt.show()

if __name__ == '__main__':
    # --- 1. 配置区 ---
    config = {
        "filename": "2ch_iq_data.bin", "sample_rate": 4e6, "center_freq": 2.41e9,
        "antenna_spacing_m": 0.025, "detection_threshold_db": 15
    }
    
    # --- 2. 检查并读取数据 ---
    total_samples = inspect_iq_file(config['filename'], config['sample_rate'])
    if total_samples <= 0: exit()
    print("Loading full data...")
    ch0_samples, ch1_samples = read_2ch_iq(config['filename'])
    if ch0_samples is None: exit()
    
    # --- 3. 脉冲检测与匹配 ---
    bursts_ch0 = detect_signal_bursts(ch0_samples, threshold_db=config['detection_threshold_db'])
    bursts_ch1 = detect_signal_bursts(ch1_samples, threshold_db=config['detection_threshold_db'])
    matched_pairs = pair_bursts(bursts_ch0, bursts_ch1)
    print(f"Successfully matched {len(matched_pairs)} pairs.")

    # --- 4. 交互式地逐个分析匹配的脉冲 ---
    print("\n--- Starting Interactive Analysis for Matched Bursts ---")
    
    all_phases_full = []
    all_phases_golden = []
    
    for i, (b0, b1) in enumerate(matched_pairs):
        print(f"\nPress Enter to analyze next pair ({i+1}/{len(matched_pairs)}) or type 'q' to quit...")
        user_input = input()
        if user_input.lower() == 'q':
            print("Quitting analysis loop.")
            break
            
        iq0_full = ch0_samples[b0[0]:b0[1]]
        iq1_full = ch1_samples[b1[0]:b1[1]]

        # 运行方法A（一次分割预览）
        full_results = analyze_burst_full(
            iq0_full,
            iq1_full,
            sample_rate=config["sample_rate"],
            center_freq=config["center_freq"],
            antenna_spacing_m=config["antenna_spacing_m"],
        )
        golden_results, golden_indices = analyze_burst_golden_segment(
            iq0_full,
            iq1_full,
            sample_rate=config["sample_rate"],
            center_freq=config["center_freq"],
            antenna_spacing_m=config["antenna_spacing_m"],
        )

        # 当方法A有效时绘图；方法B结果按条件显示
        if np.isfinite(full_results[4]):
            all_phases_full.append(np.rad2deg(full_results[4]))
            if golden_results is not None and np.isfinite(golden_results[4]):
                all_phases_golden.append(np.rad2deg(golden_results[4]))
            create_and_show_comparison_plot(full_results, golden_results, iq0_full, iq1_full, golden_indices, config, i+1)
        else:
            print(f"Skipping pair #{i+1}, data quality too low for full-burst analysis.")

    # --- 5. (可选) 在所有交互分析结束后，显示最终的对比直方图 ---
    if all_phases_full:
        print("\nDisplaying final summary histogram...")
        plt.figure(figsize=(14, 7))
        
        plt.subplot(1, 2, 1)
        plt.hist(all_phases_full, bins=90, range=(-180, 180))
        plt.title("Phase Difference Dist. (Method A: Full Burst)")
        plt.xlabel("Phase Difference (Degrees)")
        plt.ylabel("Count")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()