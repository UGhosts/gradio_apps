import uuid

import gradio as gr
import time
import sys
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis
import numpy as np
import pandas as pd
from paddlex import create_model

BASE_DIR = Path(__file__).parent.parent
from utils.app_utils import AppUtils as util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plt = util.auto_config_chinese_font()

def vibration_analysis_core(csv_path,savepath):
    """
    振动数据核心分析方法（精简版）

    参数:
        csv_path: str - CSV文件路径（必填）

    返回:
        tuple - (综合分析图保存路径, 完整报告文本内容)
                若分析失败，返回 (None, 错误信息)
    """
    # 固定配置（可根据需求调整）
    VALUE_COL = "value"
    SAVE_DIR = savepath
    AUTO_ESTIMATE_FS = False
    SAMPLING_FREQ_DEFAULT = 1000
    KURTOSIS_WINDOW = 200
    RMS_WINDOW = 100
    # 健康状态阈值
    KURTOSIS_THRESHOLD = 0.5
    ACC_RMS_THRESHOLD = 1.0
    DISPLACEMENT_THRESHOLD = 0.001

    # ===================== 内部工具函数 =====================
    def custom_cumtrapz(y, dx=1.0, initial=0.0):
        """手动实现梯形累积积分"""
        n = len(y)
        result = np.zeros(n, dtype=np.float64)
        result[0] = initial
        for i in range(1, n):
            trapezoid_area = (y[i - 1] + y[i]) * dx / 2.0
            result[i] = result[i - 1] + trapezoid_area
        return result

    def load_vibration_data():
        """读取振动数据"""
        df = pd.read_csv(csv_path)
        if VALUE_COL not in df.columns:
            raise ValueError(f"CSV文件中未找到'{VALUE_COL}'列，请检查列名")
        return df[VALUE_COL].dropna().values

    def calculate_displacement(acc_data, sampling_freq):
        """从加速度积分计算位移"""
        velocity = custom_cumtrapz(acc_data, dx=1 / sampling_freq, initial=0)
        velocity -= np.mean(velocity)
        displacement = custom_cumtrapz(velocity, dx=1 / sampling_freq, initial=0)
        displacement -= np.mean(displacement)
        return displacement

    def calculate_fft(signal, sampling_freq):
        """FFT频域分析"""
        n = len(signal)
        yf = fft(signal)
        xf = fftfreq(n, 1 / sampling_freq)[:n // 2]
        yf_amp = 2.0 / n * np.abs(yf[:n // 2])
        return xf, yf_amp

    def calculate_rms(signal, window_size, sampling_freq):
        """计算滑动RMS"""
        if len(signal) <= window_size:
            return np.array([0]), np.array([np.sqrt(np.mean(np.square(signal)))])

        rms_values = []
        for i in range(len(signal) - window_size + 1):
            window = signal[i:i + window_size]
            rms_values.append(np.sqrt(np.mean(np.square(window))))

        rms_values = np.array(rms_values)
        rms_time = np.arange(len(rms_values)) / sampling_freq + (window_size / 2) / sampling_freq
        return rms_time, rms_values

    def calculate_kurtosis_features(acc_data, window_size, sampling_freq):
        """计算峭度特征"""
        global_kurt = kurtosis(acc_data, fisher=True)
        global_kurt_abs = kurtosis(acc_data, fisher=False)

        if len(acc_data) <= window_size:
            slide_kurt = np.array([global_kurt])
            kurt_time = np.array([len(acc_data) / (2 * sampling_freq)])
        else:
            slide_kurt = []
            for i in range(len(acc_data) - window_size + 1):
                window = acc_data[i:i + window_size]
                slide_kurt.append(kurtosis(window, fisher=True))
            slide_kurt = np.array(slide_kurt)
            kurt_time = np.arange(len(slide_kurt)) / sampling_freq + (window_size / 2) / sampling_freq

        return {
            "global_fisher_kurtosis": global_kurt,
            "global_absolute_kurtosis": global_kurt_abs,
            "sliding_kurtosis": slide_kurt,
            "sliding_kurtosis_time": kurt_time
        }

    def evaluate_health_status(acc_data, disp_data, kurt_features):
        """评估健康状态"""
        acc_rms = np.sqrt(np.mean(np.square(acc_data)))
        max_displacement = np.max(np.abs(disp_data))
        global_kurt = kurt_features['global_fisher_kurtosis']

        if (global_kurt < KURTOSIS_THRESHOLD and
                acc_rms < ACC_RMS_THRESHOLD and
                max_displacement < DISPLACEMENT_THRESHOLD):
            return {
                "status": "🟢 健康",
                "suggestion": "设备运行正常，建议继续按常规周期进行维护检查。",
                "acc_rms": acc_rms,
                "max_displacement": max_displacement
            }
        elif (global_kurt < KURTOSIS_THRESHOLD * 1.5 and
              acc_rms < ACC_RMS_THRESHOLD * 1.5 and
              max_displacement < DISPLACEMENT_THRESHOLD * 1.5):
            return {
                "status": "🟡 注意",
                "suggestion": "设备存在轻微异常振动，建议增加监测频率，密切关注状态变化。",
                "acc_rms": acc_rms,
                "max_displacement": max_displacement
            }
        else:
            return {
                "status": "🔴 异常",
                "suggestion": "设备振动指标严重超标，存在故障风险，建议立即停机检查！",
                "acc_rms": acc_rms,
                "max_displacement": max_displacement
            }

    def plot_and_save(acc_data, disp_data, time_axis, kurt_features, sampling_freq):
        """绘制并保存综合分析图"""
        os.makedirs(SAVE_DIR, exist_ok=True)

        # 设置绘图样式
        #plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = (18, 15)

        # 1. 加速度时域图
        plt.subplot(3, 2, 1)
        plt.plot(time_axis, acc_data, color='#2E86AB', linewidth=0.8, label='加速度')
        plt.text(0.02, 0.95, f'全局Fisher峭度: {kurt_features["global_fisher_kurtosis"]:.2f}',
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.title('加速度时域波形（含峭度标注）', fontsize=14, fontweight='bold')
        plt.xlabel('时间 (s)', fontsize=12)
        plt.ylabel('加速度 (m/s²)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # 2. 位移时域图
        plt.subplot(3, 2, 2)
        plt.plot(time_axis, disp_data, color='#A23B72', linewidth=0.8)
        plt.title('位移时域波形', fontsize=14, fontweight='bold')
        plt.xlabel('时间 (s)', fontsize=12)
        plt.ylabel('位移 (m)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 3. 加速度频域图
        plt.subplot(3, 2, 3)
        xf, yf_amp = calculate_fft(acc_data, sampling_freq)
        plt.plot(xf, yf_amp, color='#F18F01', linewidth=0.8)
        plt.title('加速度频域频谱', fontsize=14, fontweight='bold')
        plt.xlabel('频率 (Hz)', fontsize=12)
        plt.ylabel('幅值', fontsize=12)
        plt.xlim(0, sampling_freq / 2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 4. 加速度RMS趋势图
        plt.subplot(3, 2, 4)
        rms_time, rms_vals = calculate_rms(acc_data, RMS_WINDOW, sampling_freq)
        if len(rms_time) == len(rms_vals):
            plt.plot(rms_time, rms_vals, color='#C73E1D', linewidth=1)
        else:
            plt.plot(np.arange(len(rms_vals)) / sampling_freq, rms_vals, color='#C73E1D', linewidth=1)
        plt.title('加速度滑动RMS趋势', fontsize=14, fontweight='bold')
        plt.xlabel('时间 (s)', fontsize=12)
        plt.ylabel('加速度RMS (m/s²)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 5. 滑动峭度趋势图
        plt.subplot(3, 2, 5)
        if len(kurt_features["sliding_kurtosis_time"]) == len(kurt_features["sliding_kurtosis"]):
            plt.plot(kurt_features["sliding_kurtosis_time"],
                     kurt_features["sliding_kurtosis"],
                     color='#6A994E', linewidth=1)
        else:
            plt.plot(np.arange(len(kurt_features["sliding_kurtosis"])) / sampling_freq,
                     kurt_features["sliding_kurtosis"],
                     color='#6A994E', linewidth=1)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Fisher峭度基准线')
        plt.title('加速度滑动Fisher峭度趋势', fontsize=14, fontweight='bold')
        plt.xlabel('时间 (s)', fontsize=12)
        plt.ylabel('Fisher峭度', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # 6. 峭度分布直方图
        plt.subplot(3, 2, 6)
        plt.hist(kurt_features["sliding_kurtosis"], bins=min(50, len(kurt_features["sliding_kurtosis"])),
                 color='#7209B7', alpha=0.7)
        plt.axvline(x=kurt_features["global_fisher_kurtosis"],
                    color='red', linestyle='--',
                    label=f'全局峭度: {kurt_features["global_fisher_kurtosis"]:.2f}')
        plt.title('滑动峭度分布', fontsize=14, fontweight='bold')
        plt.xlabel('Fisher峭度值', fontsize=12)
        plt.ylabel('频次', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # 保存综合分析图
        #main_plot_path = os.path.join(SAVE_DIR, "vibration_analysis_with_kurtosis.png")
        #print(savepath)
        main_plot_path = f'{BASE_DIR}/output/dianji_cls/'+uuid.uuid4().hex+'.png'
        plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 生成峭度趋势图（保留但不返回路径，如需可扩展）
        # plt.figure(figsize=(12, 5))
        # if len(kurt_features["sliding_kurtosis_time"]) == len(kurt_features["sliding_kurtosis"]):
        #     plt.plot(kurt_features["sliding_kurtosis_time"],
        #              kurt_features["sliding_kurtosis"],
        #              color='#6A994E', linewidth=1)
        # else:
        #     plt.plot(np.arange(len(kurt_features["sliding_kurtosis"])) / sampling_freq,
        #              kurt_features["sliding_kurtosis"],
        #              color='#6A994E', linewidth=1)
        # plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Fisher峭度基准线')
        # plt.axhline(y=3, color='orange', linestyle='--', alpha=0.7, label='绝对峭度基准线(3)')
        # plt.title('加速度滑动Fisher峭度趋势', fontsize=14, fontweight='bold')
        # plt.xlabel('时间 (s)', fontsize=12)
        # plt.ylabel('Fisher峭度', fontsize=12)
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # kurtosis_plot_path = os.path.join(SAVE_DIR, "sliding_kurtosis_trend.png")
        # plt.savefig(kurtosis_plot_path, dpi=300, bbox_inches='tight')
        # plt.close()

        return main_plot_path

    def generate_report(acc_data, disp_data, kurt_features, health_status, sampling_freq):
        """生成完整报告文本"""
        # 计算补充统计值
        acc_rms = np.sqrt(np.mean(np.square(acc_data)))
        disp_rms = np.sqrt(np.mean(np.square(disp_data)))
        slide_kurt_max = np.max(kurt_features['sliding_kurtosis'])
        slide_kurt_min = np.min(kurt_features['sliding_kurtosis'])
        data_duration = len(acc_data) / sampling_freq
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 构建报告内容
        report_content = f"""【健康状态评估】
================================================================================
  状态: {health_status['status']}
  建议: {health_status['suggestion']}
  分析时间: {current_time}

【振动状态分析】
================================================================================
  1. 基础参数
     - 采样频率: {sampling_freq:.0f} Hz
     - 数据点数: {len(acc_data)} 个
     - 数据时长: {data_duration:.2f} s

  2. 加速度特征
     - 最大值: {np.max(acc_data):.4f} m/s²
     - 最小值: {np.min(acc_data):.4f} m/s²
     - 平均值: {np.mean(acc_data):.4f} m/s²
     - 均方根(RMS): {acc_rms:.4f} m/s²
     - 峰值因子: {np.max(np.abs(acc_data)) / acc_rms:.4f}

  3. 位移特征
     - 最大值: {np.max(disp_data):.6f} m
     - 最小值: {np.min(disp_data):.6f} m
     - 平均值: {np.mean(disp_data):.6f} m
     - 均方根(RMS): {disp_rms:.6f} m

  4. 峭度特征
     - 全局Fisher峭度 (减去3): {kurt_features['global_fisher_kurtosis']:.4f}
     - 全局绝对峭度: {kurt_features['global_absolute_kurtosis']:.4f}
     - 滑动峭度均值: {np.mean(kurt_features['sliding_kurtosis']):.4f}
     - 滑动峭度最大值: {slide_kurt_max:.4f}
     - 滑动峭度最小值: {slide_kurt_min:.4f}

  5. 阈值对比
     - 加速度RMS阈值: {ACC_RMS_THRESHOLD} m/s² (当前: {acc_rms:.4f} m/s²)
     - 位移阈值: {DISPLACEMENT_THRESHOLD} m (当前最大值: {np.max(np.abs(disp_data)):.6f} m)
     - Fisher峭度阈值: {KURTOSIS_THRESHOLD} (当前全局值: {kurt_features['global_fisher_kurtosis']:.4f})

================================================================================
报告结束
==============================================================================="""

        # 保存报告文件（可选，保留原功能）
        # report_path = os.path.join(SAVE_DIR, "vibration_analysis_report.txt")
        # with open(report_path, "w", encoding="utf-8") as f:
        #     f.write(report_content)

        return report_content

    # ===================== 主分析流程 =====================
    try:
        # 1. 加载数据
        print(f"正在读取振动数据: {csv_path}")
        acc_data = load_vibration_data()
        print(f"成功读取 {len(acc_data)} 个振动数据点")

        # 2. 确定采样频率
        sampling_freq = SAMPLING_FREQ_DEFAULT if not AUTO_ESTIMATE_FS else 1000
        print(f"采样频率确定为: {sampling_freq:.0f} Hz")

        # 3. 计算位移
        print("正在计算位移数据...")
        disp_data = calculate_displacement(acc_data, sampling_freq)

        # 4. 计算峭度特征
        print("正在计算峭度特征...")
        kurt_features = calculate_kurtosis_features(acc_data, KURTOSIS_WINDOW, sampling_freq)

        # 5. 评估健康状态
        print("正在评估设备健康状态...")
        health_status = evaluate_health_status(acc_data, disp_data, kurt_features)

        # 6. 绘制并保存综合分析图
        print("正在生成分析图表...")
        main_plot_path = plot_and_save(acc_data, disp_data, np.arange(len(acc_data)) / sampling_freq,
                                       kurt_features, sampling_freq)
        print(f"综合分析图已保存至: {main_plot_path}")

        # 7. 生成完整报告
        print("正在生成分析报告...")
        report_content = generate_report(acc_data, disp_data, kurt_features, health_status, sampling_freq)
        print("✅ 振动数据分析完成！")

        # 返回核心结果：综合图路径 + 报告文本
        return main_plot_path, report_content

    except Exception as e:
        error_msg = f"分析失败: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return None, error_msg


def process_input(selected_model_dir):


    """处理全局选中的测试文件，返回图表和结果"""
    time.sleep(1)
    preset_info = f"测试文件: {selected_preset}" if selected_preset else "未选择测试文件"
    model_info = f"模型目录: {selected_model_dir}"
    result =''
    # 检查是否选择了测试文件
    if not selected_preset:
        return None, f"错误: 请先选择一个测试文件\n{preset_info}\n{model_info}"
    else:
        model = create_model(model_name="TimesNet_cls", model_dir=selected_model_dir)
        filepath = selected_preset
        output = model.predict(filepath, batch_size=1)
        savepath = f"{BASE_DIR}/output/dianji_cls"  # 结果目录
        # 调用新的方法
        plot_path, report_content = vibration_analysis_core(selected_preset,savepath)
        for res in output:
            #res.print()  ## 打印预测的结构化输出
            #res.save_to_img(save_path=savepath)
            res.save_to_json(save_path=savepath)

            separator = os.sep
            # 为上传的图片生成唯一文件名
            json_filename = selected_preset.split(separator)[-1].split('.')[0] + '_res.json'
            img_name = selected_preset.split(separator)[-1].split('.')[0] + '_res.png'
            with open(savepath+"/"+json_filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(data['classification'])
        #return savepath+"/"+img_name, data['classification']
        return plot_path, report_content

def set_selected(file_path, buttons, file_paths):
    """更新选中状态，修改按钮样式并更新全局变量"""
    global selected_preset
    selected_preset = file_path

    # 返回所有按钮的样式更新列表
    # 对于每个按钮，如果它对应的文件路径与选中的文件路径相同，则设置为primary（高亮），否则设置为secondary（默认）
    return [gr.update(variant="primary" if fp == file_path else "secondary") for fp, btn in zip(file_paths, buttons)]


def create_interface():
    # 从dataset/目录动态读取CSV文件
    cwru_dir = os.path.join(os.path.dirname(__file__), "dataset", "dianji_cls")
    preset_files = {}

    # 确保使用绝对路径或者正确的相对路径
    if not os.path.exists(cwru_dir):
        # 尝试使用其他可能的路径
        alt_paths = [
            #"E:/ai-dataset/motor_fault_detect_/validation/positive_samples",
            f"{BASE_DIR}/dataset/dianji_cls",
            "./dataset/dianji_cls",
            "dataset/dianji_cls",
        ]
        for path in alt_paths:
            if os.path.exists(path):
                cwru_dir = path
                break

    # 获取目录下所有CSV文件
    if os.path.exists(cwru_dir):
        for file_name in os.listdir(cwru_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(cwru_dir, file_name)
                preset_files[file_path] = f"📄 {file_name}"

    # 如果没有找到文件，使用默认文件
    if not preset_files:
        preset_files = {"dataset/dianji_cls/t_n1.csv": "📄 t_n1.csv"}

    # 从model/dianji_model目录读取子目录作为模型选项
    model_dir = os.path.join(os.path.dirname(__file__), "model", "dianji_cls")
    model_options = []  # 将使用元组列表: [(子目录名称, 完整路径)]

    if not os.path.exists(model_dir):
        # 尝试使用其他可能的路径
        alt_model_paths = [
            f"{BASE_DIR}/model/dianji_cls",
            "./model/dianji_cls",
            "model/dianji_cls",
        ]
        for path in alt_model_paths:
            if os.path.exists(path):
                model_dir = path
                break

    # 获取目录下所有子目录
    if os.path.exists(model_dir):
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            if os.path.isdir(item_path):
                # 添加元组(显示文本, 实际值)
                model_options.append((item, item_path))

    # 如果没有找到模型目录，使用默认值
    # if not model_options:
    #     default_model_name = "Timesnet_cls"
    #     default_model_dir = os.path.join(model_dir, default_model_name)
    #     model_options.append((default_model_name, default_model_dir))

    with gr.Blocks(title="电机故障预测应用") as demo:
        gr.Markdown("# 🚀 电机故障预测应用")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 选择测试文件")

                # 动态创建文件按钮
                buttons = []
                file_paths = list(preset_files.keys())
                for file_path, display_text in preset_files.items():
                    btn = gr.Button(display_text, variant="secondary", size="lg")
                    buttons.append(btn)

                # 在创建完所有按钮后，为每个按钮绑定点击事件
                for i, file_path in enumerate(file_paths):
                    buttons[i].click(
                        fn=lambda path=file_path: set_selected(path, buttons, file_paths),
                        inputs=[],
                        outputs=buttons
                    )

                # 添加模型选择下拉框
                gr.Markdown("### 选择模型")
                model_dropdown = gr.Dropdown(
                    choices=model_options,
                    label="模型列表",
                    value=model_options[0][1] if model_options else ""  # 使用元组的第二个元素作为默认值
                )

                process_btn = gr.Button("处理", variant="primary")

            with gr.Column(scale=2):  # 扩大结果展示区域
                gr.Markdown("### 原始振动信号图")
                plot_output = gr.Image(label="数据曲线", type="pil")

                gr.Markdown("### 处理结果")
                output_text = gr.Textbox(label="预测结果", lines=6)

        # 处理按钮事件（返回图片和文本结果）
        process_btn.click(
            fn=process_input,
            inputs=[model_dropdown],
            outputs=[plot_output, output_text]
        )

    return demo


def main():
    # 从命令行参数获取端口号，如果未提供则使用默认端口7860
    port = 7861
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            if port < 1024 or port > 65535:
                print(f"警告：端口号 {port} 不在有效范围内(1024-65535)，将使用默认端口7860")
                port = 7860
        except ValueError:
            print(f"警告：无效的端口号参数 '{sys.argv[1]}'，将使用默认端口7860")

    demo = create_interface()
    demo.launch(allowed_paths=[f'{BASE_DIR}/output'],server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()