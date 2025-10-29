import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import hilbert, butter, filtfilt, find_peaks
import os
plt.rcParams['font.family'] = 'Arial'


def estimate_noise_level_normalized(signal):
    """在无采样频率情况下估计信号的噪声水平（归一化方法）"""
    # 对信号进行FFT以分析频率成分
    n = len(signal)
    yf = fft(signal)
    xf_normalized = np.linspace(0.0, 0.5, n // 2, endpoint=False)
    psd = np.abs(yf[:n // 2]) ** 2  # 功率谱密度估计

    # 假设高频部分主要是噪声（归一化频率 > 0.35）
    noise_mask = xf_normalized > 0.35

    # 计算噪声功率和总功率
    noise_power = np.mean(psd[noise_mask]) if np.any(noise_mask) else np.mean(psd)
    total_power = np.mean(psd)
    noise_ratio = noise_power / total_power

    # 分类噪声水平
    if noise_ratio < 0.2:
        noise_level = "低噪声"
        noise_category = 0
    elif noise_ratio < 0.5:
        noise_level = "中噪声"
        noise_category = 1
    else:
        noise_level = "高噪声"
        noise_category = 2

    print(f"噪声评估: {noise_level} (噪声功率比: {noise_ratio:.2f})")
    return noise_category, noise_ratio


def adaptive_filter_normalized(signal, noise_category):
    """根据噪声水平应用不同的归一化滤波策略"""
    # 基于归一化频率的滤波器参数（0到0.5之间）
    if noise_category == 0:  # 低噪声
        # 轻度滤波，保留更多细节
        low_norm = 0.1
        high_norm = 0.4
        order = 3
        #print(f"应用低噪声滤波策略: 归一化频率 {low_norm:.1f}-{high_norm:.1f}, 3阶")
    elif noise_category == 1:  # 中噪声
        # 中度滤波，平衡细节和噪声去除
        low_norm = 0.15
        high_norm = 0.35
        order = 4
        #print(f"应用中噪声滤波策略: 归一化频率 {low_norm:.1f}-{high_norm:.1f}, 4阶")
    else:  # 高噪声
        # 强滤波，优先去除噪声
        low_norm = 0.2
        high_norm = 0.3
        order = 5
        #print(f"应用高噪声滤波策略: 归一化频率 {low_norm:.1f}-{high_norm:.1f}, 5阶")

    # 设计并应用带通滤波器
    b, a = butter(order, [low_norm, high_norm], btype='band')
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal, (low_norm, high_norm)


def remove_dc_component(signal):
    """去除信号的直流分量"""
    dc_component = np.mean(signal)
    signal_no_dc = signal - dc_component
    return signal_no_dc, dc_component


def plot_envelope_spectrum_no_fs(csv_file_path, peak_threshold=0,name=''):
    """
    在没有采样频率的情况下，进行噪声区分和过滤的包络谱分析

    参数:
    csv_file_path: CSV文件路径
    peak_threshold: 峰值显示阈值
    """
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件 '{csv_file_path}' 不存在")
        return
    filepath = ''
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)

        if 'ai1' not in df.columns:
            print("错误: CSV文件中未找到'ai1'列")
            #print(f"文件中的列名: {', '.join(df.columns)}")
            return

        # 提取原始信号
        raw_signal = df['ai1'].values
        n = len(raw_signal)
        #print(f"成功读取数据，共 {n} 个样本点")
        #print("注意: 未提供采样频率，所有频率将以归一化形式展示 (0到0.5，相对于采样频率的比例)")

        # 去除直流分量
        signal, dc_component = remove_dc_component(raw_signal)
        #print(f"已去除直流分量: {dc_component:.6f}")

        import numpy as np
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))  # 设置画布大小
        plt.plot(signal)

        plt.title('电机振动图', fontsize=14)
        plt.xlabel('', fontsize=12)
        plt.ylabel('位移 (振幅)', fontsize=12)

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()

        raw_filepath = './output/dianji_cls/' + name + '_vib.png'
        plt.savefig(raw_filepath, dpi=300, bbox_inches='tight')

        # 评估噪声水平（归一化方法）
        noise_category, noise_ratio = estimate_noise_level_normalized(signal)

        # 根据噪声水平应用自适应滤波（归一化频率）
        filtered_signal, cutoff_norm = adaptive_filter_normalized(signal, noise_category)

        plt.figure(figsize=(10, 4))  # 设置画布大小
        plt.plot(signal)

        plt.title('', fontsize=14)
        plt.xlabel('', fontsize=12)
        #plt.ylabel('', fontsize=12)
        plt.yticks([])  # 去掉

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        #plt.legend()

        filepath='./output/dianji_cls/' + name + '_wave.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

        #
        # # 希尔伯特变换与包络提取 先用此方案
        # analytic_signal = hilbert(filtered_signal)
        # envelope = np.abs(analytic_signal)
        #
        # # 包络谱计算（使用归一化频率 先用此方案）
        # yf = fft(envelope)
        # xf_normalized = np.linspace(0.0, 0.5, n // 2, endpoint=False)
        # amplitude = 2.0 / n * np.abs(yf[:n // 2])
        #
        # # 动态调整峰值阈值，高噪声时提高阈值
        # adjusted_threshold = peak_threshold * (1 + min(noise_ratio, 1.0))
        # #print(f"调整后的峰值检测阈值: {adjusted_threshold:.2f}")
        #
        # # 寻找显著峰值
        # peaks, _ = find_peaks(amplitude, height=adjusted_threshold * np.max(amplitude))
        # peak_freqs_norm = xf_normalized[peaks]
        # peak_amps = amplitude[peaks]
        #
        # # 检查是否有时间信息，尝试估计采样频率（如果可能）
        # fs_estimated = None
        # time_columns = [col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
        # if time_columns:
        #     #print(f"\n检测到可能的时间列: {time_columns[0]}")
        #     time_data = df[time_columns[0]].values
        #
        #     if len(time_data) > 1 and not np.allclose(time_data, time_data[0]):
        #         # 计算时间差（假设时间单位为秒）
        #         dt_avg = np.mean(np.diff(time_data))
        #         if dt_avg > 0:
        #             fs_estimated = 1.0 / dt_avg
        #
        #             peak_freqs_actual = peak_freqs_norm * fs_estimated
        #
        # plt.figure(figsize=(12, 6))
        # # 包络谱  此处还是用归一化吧
        # plt.plot(xf_normalized[1:], amplitude[1:], linewidth=1.5)
        # # plt.title('振动信号包络谱（归一化频率）', fontsize=14)
        # # plt.ylabel('振幅', fontsize=12)
        # plt.title('', fontsize=14)
        # plt.ylabel('', fontsize=12)
        #
        # # 标记显著峰值
        # #plt.scatter(peak_freqs_norm, peak_amps, color='red', s=50, zorder=3, label='显著特征频率')
        # for i, (freq_norm, amp) in enumerate(zip(peak_freqs_norm, peak_amps)):
        #     # 显示归一化频率，如果可能的话也显示实际频率估计值
        #     if fs_estimated:
        #         label = f"{freq_norm:.3f} (≈{peak_freqs_actual[i]:.1f} Hz)"
        #     else:
        #         label = f"{freq_norm:.3f}"
        #
        #     plt.annotate(label,
        #                      (freq_norm, amp),
        #                      xytext=(10, 10),
        #                      textcoords='offset points',
        #                      arrowprops=dict(arrowstyle='->', color='black'))
        #
        # plt.grid(True, alpha=0.3)
        # plt.xlim(0, 0.1)  # 显示较低的频率范围，通常包含主要特征
        # plt.yticks([]) #不展示
        # #plt.legend()
        # plt.tight_layout()
        # filepath='./output/dianji_cls/' + name + '_wave.png'
        # plt.savefig(filepath, dpi=300, bbox_inches='tight')
        #plt.show()

        # print("\n显著特征频率:")
        # if fs_estimated:
        #     for freq_norm, freq_actual in zip(peak_freqs_norm, peak_freqs_actual):
        #         print(f"- 归一化频率: {freq_norm:.3f}, 估计实际频率: {freq_actual:.2f} Hz")
        # else:
        #     for freq_norm in peak_freqs_norm:
        #         print(f"- 归一化频率: {freq_norm:.3f} (相对采样频率的比例)")

    except Exception as e:
        print(f"处理数据时发生错误: {str(e)}")

    return filepath,raw_filepath


if __name__ == "__main__":
    csv_file = "E:/ai-dataset/motor_fault_detect/train/negative_samples/06c280a2-917c-49e4-9cf0-5a5dab14b281_F.csv"
    #csv_file = "E:/ai-dataset/motor_fault_detect/train/positive_samples/9496b2dd-cf29-4064-ba5f-f5011906eaae_B.csv"
    name=''
    plot_envelope_spectrum_no_fs(
                csv_file,
                peak_threshold=0.2,  # 基础峰值阈值
                name=name
            )
    from pathlib import Path
    #path='E:/ai-dataset/motor_fault_detect/train/positive_samples'
    #path = 'E:/ai-dataset/motor_fault_detect/train/negative_samples'
    #path = 'E:/ai-dataset/motor_fault_detect/validation/negative_samples'
    # path = 'E:/ai-dataset/motor_fault_detect/validation/positive_samples'
    # root_path = Path(path)
    # i=0
    # for item in root_path.iterdir():
    #
    #     name = item.name
    #     if '_F' in name:
    #         i += 1
    #         print(i)
    #         plot_envelope_spectrum_no_fs(
    #             item,
    #             peak_threshold=0.2,  # 基础峰值阈值
    #             name=name
    #         )
    #     if i>=25:
    #         break
    # path = './images'
    # root_path = Path(path)
    # for item in root_path.iterdir():
    #     if 'v_p' in item.name:
    #         print("images/" + item.name, 0)
    #     if 'v_n' in item.name:
    #         print("images/" + item.name, 1)
