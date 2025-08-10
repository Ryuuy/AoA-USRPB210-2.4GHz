import argparse
import numpy as np
from typing import List, Tuple

# 注意：这个脚本依赖于 sdr_analysis_tools.py 中的 read_2ch_iq 和 detect_signal_bursts_v2
from sdr_analysis_tools import read_2ch_iq, detect_signal_bursts_v2


def format_dbfs(x_lin: float) -> float:
    return 10.0 * np.log10(max(x_lin, 1e-18))


def summarize_segment(iq: np.ndarray) -> Tuple[float, float, float]:
    power = np.abs(iq) ** 2
    return (
        format_dbfs(float(np.mean(power))),
        format_dbfs(float(np.max(power))),
        format_dbfs(float(np.median(power))),
    )


def print_samples_block(label: str, data: np.ndarray, max_print: int = 256):
    n = len(data)
    head = min(n, max_print // 2)
    tail = min(n - head, max_print - head)
    print(f"{label}: total={n} samples")
    if n <= max_print:
        for i in range(n):
            v = data[i]
            print(f"  idx={i:6d}  re={v.real:+.5f}  im={v.imag:+.5f}  |v|={abs(v):.5f}")
    else:
        for i in range(head):
            v = data[i]
            print(f"  idx={i:6d}  re={v.real:+.5f}  im={v.imag:+.5f}  |v|={abs(v):.5f}")
        print("  ...")
        for i in range(n - tail, n):
            v = data[i]
            print(f"  idx={i:6d}  re={v.real:+.5f}  im={v.imag:+.5f}  |v|={abs(v):.5f}")


def main():
    p = argparse.ArgumentParser(description="Print time-domain waveforms for detected bursts")
    p.add_argument("--filename", default="2ch_iq_data.bin")
    p.add_argument("--sample_rate", type=float, default=4e6)
    p.add_argument("--threshold_db", type=float, default=10.0)
    # ... (其他 argparse 参数)
    args = p.parse_args()

    ch0, ch1 = read_2ch_iq(args.filename)
    if ch0 is None:
        print(f"Error: cannot read IQ file '{args.filename}'")
        return

    bursts, debug = detect_signal_bursts_v2(
        ch0,
        sample_rate=args.sample_rate,
        threshold_db=args.threshold_db,
        # ... (其他参数传递)
    )

    sr = args.sample_rate
    print(f"Detected {len(bursts)} bursts")
    if len(bursts) == 0:
        return

    for bi, (s, e) in enumerate(bursts[: args.max_bursts]):
        dur_s = (e - s + 1) / sr
        mid = s + (e - s) // 2
        print("\n" + "=" * 60)
        print(f"Burst #{bi+1}: start={s} end={e}  len={e-s+1}  dur={dur_s*1e6:.2f} us")
        # ... (其他打印逻辑)

if __name__ == "__main__":
    main()