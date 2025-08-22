import numpy as np
import uhd
import threading
import time

# ... (其他参数保持不变) ...
SDR_ARGS = "serial=321D889,num_recv_frames=1024"
SAMPLE_RATE = 4e6
CENTER_FREQ = 2.455e9
GAIN = 50
NUM_CHANNELS = 2
FILENAME = "2ch_iq_data.bin"

def acquire_and_save(stop_event):
    """主采集函数：配置USRP，进行定时启动，并以二进制交错格式连续保存双通道数据"""
    print(f"开始双通道数据采集，采样率: {SAMPLE_RATE/1e6} MHz, 中心频率: {CENTER_FREQ/1e9} GHz")

    total_samps_written = 0
    acquisition_start_time = None
    
    # <<< MODIFICATION: 将 uhd 对象声明为 None，以便在 finally 中检查 >>>
    usrp = None
    streamer = None

    with open(FILENAME, "ab") as f:
        try:
            usrp = uhd.usrp.MultiUSRP(SDR_ARGS)
            # ... (其他配置保持不变)
            usrp.set_rx_rate(SAMPLE_RATE)
            usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(CENTER_FREQ))
            usrp.set_rx_gain(GAIN, 0)
            usrp.set_rx_antenna("RX2", 0)
            usrp.set_rx_gain(GAIN, 1)
            usrp.set_rx_antenna("RX2", 1)
            st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            st_args.channels = list(range(NUM_CHANNELS))
            streamer = usrp.get_rx_stream(st_args)
            recv_buffer = np.zeros((NUM_CHANNELS, 8192), dtype=np.complex64)
            metadata = uhd.types.RXMetadata()
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
            stream_cmd.stream_now = False
            start_time = usrp.get_time_now() + uhd.libpyuhd.types.time_spec(0.2)
            stream_cmd.time_spec = start_time
            streamer.issue_stream_cmd(stream_cmd)
            print("正在预热数据流...")
            streamer.recv(recv_buffer, metadata, timeout=0.5)
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                 print(f"预热时出现一个预料内的错误: {metadata.strerror()} (可忽略)")

            print(f"串流已稳定！开始写入文件... 按 Ctrl+C 停止。")
            
            while not stop_event.is_set():
                samps = streamer.recv(recv_buffer, metadata)
                
                if acquisition_start_time is None and samps > 0:
                    acquisition_start_time = time.monotonic()
                
                total_samps_written += samps

                if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                    print(metadata.strerror())
                
                f.write(recv_buffer.T.tobytes())

        except Exception as e:
            print(f"线程内发生异常: {e}")
        finally:
            # --- 关键修复：手动、有序地停止和销毁UHD对象 ---
            print("\n正在安全关闭数据流和设备...")
            
            # 1. 停止数据流
            if streamer:
                stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
                streamer.issue_stream_cmd(stop_cmd)
            
            # 打印统计信息
            if acquisition_start_time is not None:
                acquisition_end_time = time.monotonic()
                measured_duration = acquisition_end_time - acquisition_start_time
                theoretical_duration = total_samps_written / SAMPLE_RATE
                
                print("\n--- 采集统计 ---")
                print(f"总计采集样本数 (每通道): {total_samps_written}")
                print(f"实际采集时间: {measured_duration:.6f} 秒")
                print(f"理论采集时间: {theoretical_duration:.6f} 秒")
                print("------------------")

            # 2. 显式删除对象，触发C++析构函数以释放硬件
            #    这个操作对于防止C++底层崩溃至关重要
            if streamer:
                del streamer
            if usrp:
                del usrp
            
            print("设备已安全关闭。")
# ... (主线程部分 __main__ 保持不变) ...
if __name__ == "__main__":
    with open(FILENAME, "wb") as f:
        pass
    stop_event = threading.Event()
    acq_thread = threading.Thread(target=acquire_and_save, args=(stop_event,))
    print("启动采集线程...")
    acq_thread.start()
    try:
        acq_thread.join()
    except KeyboardInterrupt:
        print("\n主线程收到中断信号，正在停止采集线程...")
        stop_event.set()
    acq_thread.join()
    print("程序已完全退出。")