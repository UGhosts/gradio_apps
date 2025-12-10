import uuid

import gradio as gr
import time
import sys
import os
import json
import matplotlib.pyplot as plt

from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
from utils.app_utils import AppUtils as util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plt = util.auto_config_chinese_font()


import numpy as np
import pandas as pd
from scipy.stats import kurtosis, zscore
from scipy.signal import detrend
import matplotlib as mpl
from datetime import datetime
import os


def analyze_gearbox_vibration(csv_path, save_dir="./"):
    plt.set_loglevel('WARNING')

    # ---------------------- 2. 读取并分析4轴CSV数据 ----------------------
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"读取CSV文件失败：{str(e)}")

    # 自动识别关键列（可根据实际CSV格式调整）
    time_col = None
    axis_cols = []
    for col in df.columns:
        col_lower = col.lower()
        # 识别时间列
        if 'time' in col_lower or '时间' in col_lower:
            time_col = col
        # 识别4个轴的振动列
        elif any(key in col_lower for key in ['axis', '轴', '1', '2', '3', '4']) and len(axis_cols) < 4:
            axis_cols.append(col)

    # 若自动识别失败，手动指定
    if time_col is None:
        time_col = df.columns[0]  # 默认第一列为时间
    if len(axis_cols) < 4:
        axis_cols = df.columns[1:5]  # 默认第2-5列为4个轴的振动数据

    # 提取核心数据
    t = df[time_col].values  # 时间序列
    fs = 1 / np.mean(np.diff(t))  # 自动计算采样频率（Hz）

    # ---------------------- 3. 特征计算函数 ----------------------
    # 中心差分法计算加速度
    def cal_acceleration(displacement, dt):
        diff1 = np.zeros_like(displacement)  # 一阶导数（速度）
        diff1[1:-1] = (displacement[2:] - displacement[:-2]) / (2 * dt)
        diff1[0] = (displacement[1] - displacement[0]) / dt
        diff1[-1] = (displacement[-1] - displacement[-2]) / dt

        diff2 = np.zeros_like(diff1)  # 二阶导数（加速度）
        diff2[1:-1] = (diff1[2:] - diff1[:-2]) / (2 * dt)
        diff2[0] = (diff1[1] - diff1[0]) / dt
        diff2[-1] = (diff1[-1] - diff1[-2]) / dt
        return diff2

    # 优化的滑动峭度计算函数
    def cal_sliding_kurtosis_optimized(data, window_size, fs, method='acceleration'):
        """优化的峭度计算函数"""
        # 数据预处理：去趋势 + 标准化（关键！消除基线偏移）
        data_processed = detrend(data)  # 去趋势
        data_processed = zscore(data_processed)  # 标准化

        # 如果选择基于加速度计算峭度（更易体现冲击特征）
        if method == 'acceleration':
            dt = 1 / fs
            data_processed = cal_acceleration(data_processed, dt)
            data_processed = zscore(data_processed)  # 再次标准化

        kurt_vals = np.zeros_like(data_processed)
        half_window = window_size // 2

        # 遍历计算每个点的峭度（确保窗口有效）
        for i in range(len(data_processed)):
            start = max(0, i - half_window)
            end = min(len(data_processed), i + half_window)
            window_data = data_processed[start:end]

            # 确保窗口有足够数据（至少5个点）
            if len(window_data) >= 5:
                # 计算峭度（fisher=False：原始峭度）
                kurt_val = kurtosis(window_data, fisher=False)
                # 避免NaN/Inf，替换为合理值
                if np.isnan(kurt_val) or np.isinf(kurt_val):
                    kurt_vals[i] = 3.0  # 正态分布峭度值
                else:
                    kurt_vals[i] = kurt_val
            else:
                # 窗口过小时用全局峭度填充
                kurt_vals[i] = kurtosis(data_processed, fisher=False)

        return kurt_vals

    # ---------------------- 4. 批量处理4个轴 ----------------------
    axis_results = {}
    dt = 1 / fs  # 采样时间间隔

    # 优化窗口大小（关键！根据采样频率自适应）
    min_window_size = 30  # 最小窗口点数（避免窗口过小）
    window_size = max(int(0.15 * fs), min_window_size)  # 0.15秒窗口，至少30个点

    # 批量处理4个轴
    for idx, axis_col in enumerate(axis_cols, 1):
        displacement = df[axis_col].values  # 当前轴的位移数据

        # 计算加速度
        acceleration = cal_acceleration(displacement, dt)

        # 计算优化后的滑动峭度（基于加速度）
        kurt_vals = cal_sliding_kurtosis_optimized(
            displacement,
            window_size=window_size,
            fs=fs,
            method='acceleration'
        )

        # 存储结果
        axis_results[f'轴{idx}'] = {
            '位移': displacement,
            '加速度': acceleration,
            '峭度': kurt_vals,
            '位移最大值': np.max(displacement),
            '位移最小值': np.min(displacement),
            '加速度最大值': np.max(acceleration),
            '加速度最小值': np.min(acceleration),
            '峭度最大值': np.max(kurt_vals),
            '峭度最小值': np.min(kurt_vals),
            '加速度平均值': np.mean(np.abs(acceleration)),
            '峭度平均值': np.mean(kurt_vals)
        }

    # ---------------------- 5. 绘制4轴整合图 ----------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    colors = ['tab:blue', 'tab:green', 'tab:purple', 'tab:brown']

    # 遍历每个轴绘图
    for ax_idx, (axis_name, data) in enumerate(axis_results.items()):
        ax = axes[ax_idx]

        # 主Y轴：位移 + 加速度（缩放后）
        acc_max = np.max(np.abs(data['加速度']))
        disp_max = np.max(np.abs(data['位移']))
        acc_scale = disp_max / acc_max * 0.8 if acc_max != 0 else 1
        acc_scaled = data['加速度'] * acc_scale

        # 绘制位移
        ax.plot(t, data['位移'], color=colors[ax_idx], label=f'{axis_name} 位移', alpha=0.7, linewidth=1)
        # 绘制加速度
        ax.plot(t, acc_scaled, color='orange', label=f'{axis_name} 加速度（缩放×{acc_scale:.3f}）', alpha=0.8, linewidth=1)

        # 次Y轴：峭度
        ax2 = ax.twinx()
        kurt_scaled = data['峭度']
        ax2.plot(t, kurt_scaled, color='red', label=f'{axis_name} 峭度', linewidth=2, alpha=0.9)

        # 添加峭度基准线（正态分布峭度=3）
        ax2.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='正态分布峭度(3)')

        # 子图配置
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('位移 / 加速度（缩放后）', color=colors[ax_idx])
        ax2.set_ylabel('峭度', color='red')
        ax.set_title(f'{axis_name} 振动特征（位移+加速度+峭度）\n峭度范围：{data["峭度最小值"]:.2f} ~ {data["峭度最大值"]:.2f}')
        ax.tick_params(axis='y', labelcolor=colors[ax_idx])
        ax2.tick_params(axis='y', labelcolor='red')
        ax.grid(True, alpha=0.3)

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    # 整体标题
    fig.suptitle('4轴振动数据特征整合图（优化峭度计算）', fontsize=16, y=0.98)
    plt.tight_layout()

    # 保存图片
    os.makedirs(save_dir, exist_ok=True)
    #image_save_path = os.path.join(save_dir, '4轴振动特征整合图_优化版.png')
    image_save_path = f'{BASE_DIR}/output/chilun_cls'+uuid.uuid4().hex+'.png'
    plt.savefig(image_save_path, bbox_inches='tight', dpi=300)
    plt.close()  # 关闭图形，释放内存

    # ---------------------- 6. 生成预测报告 ----------------------
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data_points = len(t)

    # 健康状态判断
    def evaluate_health_status(axis_results):
        max_avg_acc = max([axis_results[axis]['加速度平均值'] for axis in axis_results])
        max_kurt = max([axis_results[axis]['峭度最大值'] for axis in axis_results])

        # 峭度>8通常表示有冲击（故障特征）
        if max_avg_acc < 6 and max_kurt < 10.5:
            return "🟢 健康", "设备运行正常，建议继续按常规周期进行维护检查。"
        elif max_avg_acc < 8 or (max_kurt >= 10.5 and max_kurt < 12):
            return "🟡 注意", "部分轴振动/峭度值接近警戒值，建议增加监测频率，密切关注设备运行状态。"
        else:
            return "🔴 异常", "设备振动/峭度值超出正常范围，存在故障风险（峭度异常提示冲击特征），建议立即停机检查。"

    health_status, health_suggestion = evaluate_health_status(axis_results)

    # 各轴正常范围
    normal_ranges = {
        '轴1': (3, 5),
        '轴2': (2, 7),
        '轴3': (4, 8),
        '轴4': (4, 8)
    }

    # 生成详细报告
    report = f"""================================================================================
齿轮箱预测报告（优化峭度分析）
================================================================================
分析时间: {current_time}
数据文件: {os.path.basename(csv_path)}
数据点数: {data_points}
峭度计算窗口: {window_size} 点

【健康状态评估】
--------------------------------------------------------------------------------
  状态: {health_status}
  建议: {health_suggestion}

【振动状态分析】
--------------------------------------------------------------------------------"""

    # 添加各轴详细分析
    for axis_name in axis_results:
        avg_acc = axis_results[axis_name]['加速度平均值']
        min_kurt = axis_results[axis_name]['峭度最小值']
        max_kurt = axis_results[axis_name]['峭度最大值']
        avg_kurt = axis_results[axis_name]['峭度平均值']
        max_disp = axis_results[axis_name]['位移最大值']
        min_disp = axis_results[axis_name]['位移最小值']
        max_acc = axis_results[axis_name]['加速度最大值']
        min_acc = axis_results[axis_name]['加速度最小值']
        normal_min, normal_max = normal_ranges[axis_name]

        # 峭度状态说明
        if max_kurt < 5:
            kurt_status = "正常（无明显冲击）"
        elif max_kurt < 10.5:
            kurt_status = "注意（轻微冲击特征）"
        else:
            kurt_status = "异常（明显冲击特征）"

        report += f"""
  {axis_name}平均振动: {avg_acc:.2f} m/s² (正常范围: {normal_min}-{normal_max})
  {axis_name}位移范围: {min_disp:.6f} ~ {max_disp:.6f} m
  {axis_name}加速度范围: {min_acc:.2f} ~ {max_acc:.2f} m/s²
  {axis_name}峭度范围: {min_kurt:.2f} ~ {max_kurt:.2f}（平均: {avg_kurt:.2f}）- {kurt_status}"""

    report += f"""
================================================================================
关键说明：
1. 峭度正常值（正态分布）= 3，峭度>5提示存在冲击，>10.5提示明显故障冲击
2. 本次分析基于{window_size / fs:.3f}滑动窗口计算峭度
3. 加速度单位为m/s²，位移单位为m
================================================================================
"""
    return image_save_path, report



def process_input(selected_model_dir):
    from paddlex import create_model

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
        savepath = f"{BASE_DIR}/output/chilun_cls"  # 结果目录
        img_path,rs = analyze_gearbox_vibration(selected_preset,savepath)
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

            class_map = {0: '正常', 1: '故障'}
            for res in data['classification']:
                cls = class_map[res['classid']]
                prob = res['score'] - 0.02
                rs += f"分类：{cls}，概率：{prob:.2f}"
            rs +="""报告结束
================================================================================"""
        #return savepath+"/"+img_name, data['classification']
        return img_path, rs

def set_selected(file_path, buttons, file_paths):
    """更新选中状态，修改按钮样式并更新全局变量"""
    global selected_preset
    selected_preset = file_path

    # 返回所有按钮的样式更新列表
    # 对于每个按钮，如果它对应的文件路径与选中的文件路径相同，则设置为primary（高亮），否则设置为secondary（默认）
    return [gr.update(variant="primary" if fp == file_path else "secondary") for fp, btn in zip(file_paths, buttons)]


def create_interface():
    # 从dataset/目录动态读取CSV文件
    cwru_dir = os.path.join(os.path.dirname(__file__), "dataset", "chilun_cls")
    preset_files = {}

    # 确保使用绝对路径或者正确的相对路径
    if not os.path.exists(cwru_dir):
        # 尝试使用其他可能的路径
        alt_paths = [
            #"E:/ai-dataset/motor_fault_detect_/validation/positive_samples",
            f"{BASE_DIR}/dataset/chilun_cls",
            "./dataset/chilun_cls",
            "dataset/chilun_cls",
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

    # 从model/chilun_model目录读取子目录作为模型选项
    model_dir = os.path.join(os.path.dirname(__file__), "model", "chilun_cls")
    model_options = []  # 将使用元组列表: [(子目录名称, 完整路径)]

    if not os.path.exists(model_dir):
        # 尝试使用其他可能的路径
        alt_model_paths = [
            f"{BASE_DIR}/model/chilun_cls",
            "./model/chilun_cls",
            "model/chilun_cls",
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

    with gr.Blocks(title="齿轮箱故障预测应用") as demo:
        gr.Markdown("# 🚀 齿轮箱故障预测应用")

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