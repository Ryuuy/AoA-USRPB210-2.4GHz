import numpy as np
import uhd
import threading
import time

# --- 1. 参数与配置 ---
SDR_ARGS = "serial=321D889,num_recv_frames=1024" # 您的B210序列号, 并增加了接收缓冲区
SAMPLE_RATE = 4e6    # 采样率 (4 MS/s)
CENTER_FREQ = 2.455e9 # 中心频率
GAIN = 50            # 增益
NUM_CHANNELS = 2     # 双通道
FILENAME = "2ch_iq_data.bin" # 输出文件名

def acquire_and_save(stop_event):
    """主采集函数：配置USRP，进行定时启动，并以二进制交错格式连续保存双通道数据"""
    print(f"开始双通道数据采集，采样率: {SAMPLE_RATE/1e6} MHz, 中心频率: {CENTER_FREQ/1e9} GHz")

    # 使用二进制追加模式'ab'打开文件
    with open(FILENAME, "ab") as f:
        try:
            usrp = uhd.usrp.MultiUSRP(SDR_ARGS)
            
            # --- 全局参数设置 ---
            usrp.set_rx_rate(SAMPLE_RATE)
            usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(CENTER_FREQ))
            
            # --- 逐通道参数设置 (最佳实践) ---
            print(f"配置通道0: 增益={GAIN} dB, 天线=RF A/RX2")
            usrp.set_rx_gain(GAIN, 0)
            usrp.set_rx_antenna("RX2", 0)
            
            print(f"配置通道1: 增益={GAIN} dB, 天线=RF B/RX2")
            usrp.set_rx_gain(GAIN, 1)
            usrp.set_rx_antenna("RX2", 1)
            
            # --- 数据流配置 ---
            st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            st_args.channels = list(range(NUM_CHANNELS)) # 配置为 [0, 1]
            streamer = usrp.get_rx_stream(st_args)

            recv_buffer = np.zeros((NUM_CHANNELS, 8192), dtype=np.complex64)
            metadata = uhd.types.RXMetadata()

            # --- 设置一个未来的、确定的启动时间以确保通道同步 ---
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
            stream_cmd.stream_now = False
            start_time = usrp.get_time_now() + uhd.libpyuhd.types.time_spec(0.2)
            stream_cmd.time_spec = start_time
            streamer.issue_stream_cmd(stream_cmd)
            
            # --- 预热和冲刷管道 ---
            print("正在预热数据流...")
            streamer.recv(recv_buffer, metadata, timeout=0.5)
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                 print(f"预热时出现一个预料内的错误: {metadata.strerror()} (可忽略)")

            print(f"串流已稳定！开始写入文件... 按 Ctrl+C 停止。")
            
            while not stop_event.is_set():
                samps = streamer.recv(recv_buffer, metadata)
                if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                    print(metadata.strerror())
                # 使用转置来获得正确的样本交错顺序
                f.write(recv_buffer.T.tobytes())

        except Exception as e:
            print(f"线程内发生异常: {e}")
        finally:
            # 发送停止指令
            stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            if 'streamer' in locals() and streamer:
                streamer.issue_stream_cmd(stop_cmd)
            print("\n数据流已停止。")


if __name__ == "__main__":
    # 每次运行前清空旧文件
    with open(FILENAME, "wb") as f:
        pass

    stop_event = threading.Event()
    acq_thread = threading.Thread(target=acquire_and_save, args=(stop_event,))

    print("启动采集线程...")
    acq_thread.start()

    try:
        # 等待线程自然结束或用户中断
        acq_thread.join()
    except KeyboardInterrupt:
        print("\n主线程收到中断信号，正在停止采集线程...")
        stop_event.set()

    # 确保线程完全退出
    acq_thread.join()
    print("程序已完全退出。")