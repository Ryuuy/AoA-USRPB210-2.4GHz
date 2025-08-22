import numpy as np
import uhd
import threading
import time

# =============================================================================
#  --- 1. 参数设置 ---
# =============================================================================

# -- 运行模式控制 --
# 设置为数字 (例如 5.0)，程序将运行指定秒数后自动停止。
# 设置为 None，程序将连续采集，直到您按下 Ctrl+C。
RUN_DURATION_SECONDS = 3.0

# -- SDR设备参数 --
SDR_ARGS = "serial=321D889,num_recv_frames=1024"
SAMPLE_RATE = 4e6
CENTER_FREQ = 2.455e9
GAIN = 50
NUM_CHANNELS = 2
FILENAME = "2ch_iq_data.bin"

# =============================================================================
#  --- 2. 子线程采集循环 (无需修改) ---
# =============================================================================

def acquire_loop(streamer, stop_event, file_handle):
    """
    这是一个精简的采集循环函数，它只负责接收和写入数据。
    硬件资源(streamer)由调用它的主线程管理。
    """
    # ... (这个函数和上一版完全一样，无需修改)
    total_samps_written = 0
    acquisition_start_time = None
    
    recv_buffer = np.zeros((NUM_CHANNELS, 8192), dtype=np.complex64)
    metadata = uhd.types.RXMetadata()

    print("串流已稳定！子线程开始接收数据...")
    if RUN_DURATION_SECONDS is None:
        print("模式：连续采集。按 Ctrl+C 停止。")
    else:
        print(f"模式：固定时长。将自动运行 {RUN_DURATION_SECONDS} 秒。")


    while not stop_event.is_set():
        try:
            samps = streamer.recv(recv_buffer, metadata, timeout=0.1)
            if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                if metadata.error_code == uhd.types.RXMetadataErrorCode.timeout:
                    continue
                else:
                    print(f"Stream error: {metadata.strerror()}")
                    continue

            if acquisition_start_time is None and samps > 0:
                acquisition_start_time = time.monotonic()
            
            total_samps_written += samps
            file_handle.write(recv_buffer[:, :samps].T.tobytes())

        except Exception as e:
            print(f"采集线程发生异常: {e}")
            break

# =============================================================================
#  --- 3. 主函数 (整合了两种模式) ---
# =============================================================================

def main():
    """
    主函数负责所有硬件的创建、配置和最终的销毁。
    并根据 RUN_DURATION_SECONDS 的设置选择运行模式。
    """
    with open(FILENAME, "wb") as f:
        pass

    stop_event = threading.Event()
    usrp = None
    streamer = None
    acq_thread = None

    try:
        # --- 硬件创建和配置 (通用) ---
        print("主线程：开始配置硬件...")
        usrp = uhd.usrp.MultiUSRP(SDR_ARGS)
        # ... (配置代码和之前一样)
        usrp.set_rx_rate(SAMPLE_RATE)
        usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(CENTER_FREQ))
        usrp.set_rx_gain(GAIN, 0)
        usrp.set_rx_antenna("RX2", 0)
        usrp.set_rx_gain(GAIN, 1)
        usrp.set_rx_antenna("RX2", 1)
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = list(range(NUM_CHANNELS))
        streamer = usrp.get_rx_stream(st_args)
        
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = False
        start_time = usrp.get_time_now() + uhd.libpyuhd.types.time_spec(0.2)
        stream_cmd.time_spec = start_time
        streamer.issue_stream_cmd(stream_cmd)
        
        dummy_buffer = np.zeros((NUM_CHANNELS, 1024), dtype=np.complex64)
        streamer.recv(dummy_buffer, uhd.types.RXMetadata(), timeout=0.5)

        # --- 启动子线程，并根据模式选择等待方式 ---
        with open(FILENAME, "ab") as file_handle:
            acq_thread = threading.Thread(target=acquire_loop, args=(streamer, stop_event, file_handle))
            acq_thread.start()
            
            # <<< 这是新的逻辑分支 >>>
            if RUN_DURATION_SECONDS is not None and RUN_DURATION_SECONDS > 0:
                # 固定时长模式
                acq_thread.join(timeout=RUN_DURATION_SECONDS)
            else:
                # 连续采集模式
                acq_thread.join()

    except KeyboardInterrupt:
        print("\n主线程收到中断信号...")
        
    except Exception as e:
        print(f"主线程发生异常: {e}")

    finally:
        # --- 通用的、安全关闭流程 ---
        print("\n主线程：开始执行关闭流程...")
        stop_event.set()
        
        if acq_thread and acq_thread.is_alive():
            acq_thread.join()

        print("主线程：正在安全关闭数据流和设备...")
        if streamer:
            stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            streamer.issue_stream_cmd(stop_cmd)
        
        if 'streamer' in locals() and streamer:
            del streamer
        if 'usrp' in locals() and usrp:
            del usrp
            
        print("设备已安全关闭，程序已完全退出。")

if __name__ == "__main__":
    main()