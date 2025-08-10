import numpy as np
import matplotlib.pyplot as plt
import os
from sdr_analysis_tools import (inspect_iq_file, 
                                read_2ch_iq, 
                                detect_signal_bursts,
                                pair_bursts)

def get_robust_phase_and_power(ch0_burst, ch1_burst):
    S1 = np.fft.fft(ch0_burst); S2 = np.fft.fft(ch1_burst)
    dc_ignore_bins = 20
    peak_bin = np.argmax(np.abs(S1[dc_ignore_bins:])) + dc_ignore_bins
    peak_val1 = S1[peak_bin]; peak_val2 = S2[peak_bin]
    phase_diff_rad = np.angle(peak_val1 * np.conj(peak_val2))
    phase_diff_deg = np.rad2deg(phase_diff_rad)
    power_dbfs = 10 * np.log10(np.mean(np.abs(ch0_burst)**2))
    return phase_diff_deg, power_dbfs

if __name__ == '__main__':
    config = { "filename": "2ch_iq_data.bin", "sample_rate": 4e6, "detection_threshold_db": 15 }
    total_samples = inspect_iq_file(config['filename'], config['sample_rate']);
    if total_samples <= 0: exit()
    print("Loading full data..."); ch0_samples, ch1_samples = read_2ch_iq(config['filename'])
    if ch0_samples is None: exit()
    print("\nDetecting bursts on both channels...")
    bursts_ch0 = detect_signal_bursts(ch0_samples, threshold_db=config['detection_threshold_db'])
    bursts_ch1 = detect_signal_bursts(ch1_samples, threshold_db=config['detection_threshold_db'])
    print(f"Detected {len(bursts_ch0)} on Ch0 and {len(bursts_ch1)} on Ch1.")
    matched_pairs = pair_bursts(bursts_ch0, bursts_ch1)
    print(f"Successfully matched {len(matched_pairs)} pairs.")
    print("\n--- Analyzing all matched bursts ---")
    all_phases = []; all_powers = []
    for i, (b0, b1) in enumerate(matched_pairs):
        if (b0[1] - b0[0] < 2048) or (b1[1] - b1[0] < 2048): continue
        iq0 = ch0_samples[b0[0]:b0[1]]; iq1 = ch1_samples[b1[0]:b1[1]]
        phase, power = get_robust_phase_and_power(iq0, iq1)
        all_phases.append(phase); all_powers.append(power)
    if not all_phases:
        print("\nNo valid analysis results to plot.")
    else:
        print(f"\nPlotting Power vs. Phase scatter plot for {len(all_phases)} valid bursts...")
        plt.figure(figsize=(12, 8)); plt.scatter(all_phases, all_powers, alpha=0.6)
        plt.title("Power vs. Phase Difference for All Matched Bursts")
        plt.xlabel("Phase Difference (Degrees)"); plt.ylabel("Average Power (dBFS)")
        plt.grid(True); plt.xlim(-180, 180); plt.show()