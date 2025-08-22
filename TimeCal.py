import os
import numpy as np

# --- 参数设置 ---
# 请确保这些参数与生成文件的采集脚本中的参数完全一致
FILENAME = "2ch_iq_data.bin"
SAMPLE_RATE = 4e6
NUM_CHANNELS = 2
DTYPE = np.complex64 # 数据类型为 complex64 (I/Q各为float32)

def calculate_duration():
    """
    计算并打印IQ数据文件的录制时长。
    """
    try:
        # 步骤 1: 获取文件总大小 (字节)
        file_size_bytes = os.path.getsize(FILENAME)
        if file_size_bytes == 0:
            print(f"错误: 文件 '{FILENAME}' 为空，无法计算。")
            return

        # 步骤 2: 获取单个复数样本的大小 (字节)
        # complex64 = 2 * float32 = 2 * 4 bytes = 8 bytes
        bytes_per_sample = np.dtype(DTYPE).itemsize

        # 步骤 3: 计算文件中所有通道的复数样本总数
        total_complex_samples = file_size_bytes / bytes_per_sample

        # 步骤 4: 计算每个通道的样本数
        samples_per_channel = total_complex_samples / NUM_CHANNELS

        # 步骤 5: 计算时长 (秒)
        duration_seconds = samples_per_channel / SAMPLE_RATE

        # --- 清晰地打印出所有计算步骤和最终结果 ---
        print("=" * 45)
        print(f"文件分析: '{FILENAME}'")
        print("=" * 45)
        print(f"  - 文件总大小: {file_size_bytes} 字节")
        print(f"  - 采样率: {SAMPLE_RATE / 1e6} MHz")
        print(f"  - 通道数: {NUM_CHANNELS}")
        print(f"  - 单个样本大小: {bytes_per_sample} 字节")
        print("-" * 45)
        print(f"  - 总复数样本数 (所有通道): {int(total_complex_samples)}")
        print(f"  - 每通道样本数: {int(samples_per_channel)}")
        print("=" * 45)
        print(f"最终计算时长: {duration_seconds:.6f} 秒")
        print("=" * 45)

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{FILENAME}'。请确保文件存在于当前目录。")
    except Exception as e:
        print(f"计算过程中发生错误: {e}")

if __name__ == "__main__":
    calculate_duration()