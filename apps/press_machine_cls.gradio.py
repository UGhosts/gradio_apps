import uuid
import gradio as gr
import time
import sys
import os
import json
import matplotlib.pyplot as plt
os.environ.setdefault("PADDLE_DEVICE", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
from paddlex import create_model
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Setup paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model" / "press_machine_cls" / "model"
EXAMPLE_DIR = BASE_DIR / "dataset" / "press_machine_cls"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.app_utils import AppUtils as util
plt = util.auto_config_chinese_font()

# Constants
FEATURE_COLS = [
    '主电机温度',
    '高速轴轴承温度',
    '低速轴轴承温度',
    '导轨温度',
    '主电机电流',
    '主电机振动',
    '湿式离合器制动器油压',
    '液压保护预压力油压',
    '湿式离合器制动器流量',
    '液压保护气动泵进气压力',
    '润滑泵站次数',
    '滑块调节断轴检测'
]

FAULT_MAP = {
    0: '正常运行',
    1: '轴承损坏',
    2: '轴瓦发热',
    3: '导套发热',
    4: '齿轮失效',
    5: '气阀失效',  
    6: '管路泄露',  
    7: '泵阀失效',  
    8: '电路故障',  
    9: '元器件损坏',
    10: '电源故障'  
}

selected_preset = None

def preprocess_for_inference(csv_path, save_dir, window_len=96, step=96):
    """
    Read the original CSV, rename columns to dim_0...dim_11, 
    and save to temp files for inference.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"读取CSV文件失败：{str(e)}")

    # Check missing columns
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV缺少必要列: {missing}")

    # Rename columns
    new_col_names = {col: f'dim_{i}' for i, col in enumerate(FEATURE_COLS)}
    inference_df = df[FEATURE_COLS].rename(columns=new_col_names)
    
    if len(inference_df) == 0:
        raise ValueError("CSV无有效数据")

    if len(inference_df) < window_len:
        pad_count = window_len - len(inference_df)
        last_row = inference_df.iloc[-1].copy()
        pad_df = pd.DataFrame([last_row] * pad_count)
        inference_df = pd.concat([inference_df, pad_df], ignore_index=True)
        windows = [inference_df]
    else:
        windows = []
        max_start = len(inference_df) - window_len
        starts = list(range(0, max_start + 1, step))
        if starts[-1] != max_start:
            starts.append(max_start)
        for start in starts:
            windows.append(inference_df.iloc[start:start + window_len].copy())

    cols_order = ['group_id', 'time'] + [f'dim_{i}' for i in range(len(FEATURE_COLS))] + ['label']
    window_paths = []
    for idx, window in enumerate(windows):
        window['group_id'] = 0
        window['time'] = range(window_len)
        window['label'] = 0
        window = window[cols_order]
        temp_filename = f"temp_infer_{uuid.uuid4().hex}_{idx}.csv"
        temp_path = os.path.join(save_dir, temp_filename)
        window.to_csv(temp_path, index=False)
        window_paths.append(temp_path)

    return window_paths

def analyze_press_machine(csv_path, save_dir="./"):
    """
    Generate visualization and report for press machine data.
    """
    plt.set_loglevel('WARNING')
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"读取CSV文件失败：{str(e)}")

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Extract time column or use index
    if '时间戳' in df.columns:
        try:
            time_vals = pd.to_datetime(df['时间戳'])
        except:
            time_vals = df.index
    else:
        time_vals = df.index

    # Subplot 1: Temperatures
    ax = axes[0]
    temp_cols = ['主电机温度', '高速轴轴承温度', '低速轴轴承温度', '导轨温度']
    for col in temp_cols:
        if col in df.columns:
            ax.plot(time_vals, df[col], label=col)
    ax.set_title('温度监测')
    ax.set_ylabel('温度 (°C)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Subplot 2: Electrical & Vibration
    ax = axes[1]
    if '主电机电流' in df.columns:
        ax.plot(time_vals, df['主电机电流'], label='主电机电流 (A)', color='tab:blue')
    if '主电机振动' in df.columns:
        ax2 = ax.twinx()
        ax2.plot(time_vals, df['主电机振动'], label='主电机振动 (mm/s)', color='tab:red', alpha=0.7)
        ax2.set_ylabel('振动')
        ax2.legend(loc='upper right')
    ax.set_title('电流与振动')
    ax.set_ylabel('电流')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Subplot 3: Hydraulics
    ax = axes[2]
    hydro_cols = ['湿式离合器制动器油压', '液压保护预压力油压', '液压保护气动泵进气压力']
    for col in hydro_cols:
        if col in df.columns:
            ax.plot(time_vals, df[col], label=col)
    ax.set_title('液压系统')
    ax.set_ylabel('压力 (MPa)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Subplot 4: Others
    ax = axes[3]
    if '湿式离合器制动器流量' in df.columns:
        ax.plot(time_vals, df['湿式离合器制动器流量'], label='流量 (L/min)', color='tab:purple')
    if '润滑泵站次数' in df.columns:
        ax3 = ax.twinx()
        ax3.plot(time_vals, df['润滑泵站次数'], label='润滑次数', color='tab:green', linestyle='--')
        ax3.set_ylabel('次数')
        ax3.legend(loc='upper right')
    ax.set_title('流量与润滑')
    ax.set_ylabel('流量')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('时间')

    plt.tight_layout()
    
    # Save image
    os.makedirs(save_dir, exist_ok=True)
    image_save_path = f'{save_dir}/{uuid.uuid4().hex}.png'
    plt.savefig(image_save_path, bbox_inches='tight', dpi=150)
    plt.close()

    # Generate basic stats report
    report_lines = []
    report_lines.append(f"数据文件: {os.path.basename(csv_path)}")
    report_lines.append(f"数据点数: {len(df)}")
    report_lines.append("-" * 40)
    
    for col in FEATURE_COLS:
        if col in df.columns:
            mean_val = df[col].mean()
            max_val = df[col].max()
            min_val = df[col].min()
            report_lines.append(f"{col}: 均值={mean_val:.2f}, 范围=[{min_val:.2f}, {max_val:.2f}]")

    label_col = None
    if '故障名称' in df.columns:
        label_col = '故障名称'
    elif '故障类型' in df.columns:
        label_col = '故障类型'
    
    return image_save_path, "\n".join(report_lines)

def process_input(selected_model_dir):
    """处理全局选中的测试文件，返回图表和结果"""
    time.sleep(1) # Simulate processing time
    
    if not selected_preset:
        return None, "错误: 请先选择一个测试文件"
    
    if not selected_model_dir:
        return None, "错误: 请选择模型"

    output_dir = f"{BASE_DIR}/output/press_machine_cls"
    os.makedirs(output_dir, exist_ok=True)

    try:
        img_path, stats_report = analyze_press_machine(selected_preset, output_dir)
    except Exception as e:
        return None, f"分析数据失败: {str(e)}"

    # 2. Inference
    try:
        # Preprocess CSV for inference (rename columns)
        window_paths = preprocess_for_inference(selected_preset, output_dir)
        
        # Load model and predict
        model = create_model(model_name="TimesNet_cls", model_dir=selected_model_dir)
        
        # Process results
        # output is a list of dicts or similar structure
        pred_text = ""
        pred_records = []
        
        # Save raw results
        for window_path in window_paths:
            output = model.predict(window_path, batch_size=1)
            for _, res in enumerate(output):
                res.save_to_json(save_path=output_dir)
                base_name = os.path.basename(window_path).split('.')[0]
                json_path = os.path.join(output_dir, f"{base_name}_res.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if 'classification' in data:
                        pred_records.extend(data['classification'])
                    os.remove(json_path)
                else:
                    pred_text += "\n无法读取预测结果文件\n"
            os.remove(window_path)

        if pred_records:
            if len(pred_records) == 1:
                item = pred_records[0]
                class_id = item['classid']
                score = item['score']
                class_name = FAULT_MAP.get(class_id, f"未知故障({class_id})")
                pred_text += f"\n预测结果: {class_name}\n置信度: {score:.4f}\n"
            else:
                counts = {}
                best_scores = {}
                for item in pred_records:
                    class_id = item['classid']
                    score = item['score']
                    counts[class_id] = counts.get(class_id, 0) + 1
                    best_scores[class_id] = max(best_scores.get(class_id, 0.0), score)
                pred_text += f"\n窗口数: {len(window_paths)}\n"
                for class_id, count in sorted(counts.items(), key=lambda x: (-x[1], -best_scores.get(x[0], 0.0))):
                    class_name = FAULT_MAP.get(class_id, f"未知故障({class_id})")
                    pred_text += f"{class_name}: {count}次, 最高置信度: {best_scores[class_id]:.4f}\n"

    except Exception as e:
        pred_text = f"\n推理失败: {str(e)}"

    final_report = f"""========================================
冲压机故障诊断报告
========================================
{stats_report}

----------------------------------------
【AI诊断结果】
{pred_text}
========================================"""

    return img_path, final_report

def set_selected(file_path, buttons, file_paths):
    global selected_preset
    selected_preset = file_path
    return [gr.update(variant="primary" if fp == file_path else "secondary") for fp, btn in zip(file_paths, buttons)]

def create_interface():
    # Data directory
    data_dir = str(EXAMPLE_DIR)
    preset_files = {}
    
    # Scan for CSV files
    if os.path.exists(data_dir):
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.csv') and file_name.startswith('press_data'):
                file_path = os.path.join(data_dir, file_name)
                preset_files[file_path] = f"📄 {file_name}"


    model_options = []
    model_options.append(("PressMachine_Inference_v1", str(MODEL_DIR)))
    

    with gr.Blocks(title="冲压机故障预测应用") as demo:
        gr.Markdown("# 🏭 冲压机故障预测应用")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. 选择测试数据")
                
                buttons = []
                file_paths = list(preset_files.keys())
                # Limit to 10 files to avoid UI clutter if too many
                display_paths = file_paths[:10] 
                
                for file_path in display_paths:
                    display_text = preset_files[file_path]
                    btn = gr.Button(display_text, variant="secondary", size="sm")
                    buttons.append(btn)
                
                if len(file_paths) > 10:
                    gr.Markdown(f"*还有 {len(file_paths)-10} 个文件未显示*")

                # Bind clicks
                for i, file_path in enumerate(display_paths):
                    buttons[i].click(
                        fn=lambda path=file_path: set_selected(path, buttons, display_paths),
                        inputs=[],
                        outputs=buttons
                    )

                gr.Markdown("### 2. 选择模型")
                model_dropdown = gr.Dropdown(
                    choices=model_options,
                    label="模型列表",
                    value=model_options[0][1] if model_options else None,
                    interactive=True
                )
                
                process_btn = gr.Button("开始分析", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 监测数据可视化")
                plot_output = gr.Image(label="传感器曲线", type="filepath")
                
                gr.Markdown("### 📋 诊断报告")
                output_text = gr.Textbox(label="分析结果", lines=15)

        process_btn.click(
            fn=process_input,
            inputs=[model_dropdown],
            outputs=[plot_output, output_text]
        )

    return demo

def main():
    port = 7862
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
            
    demo = create_interface()
    demo.launch(allowed_paths=[f'{BASE_DIR}/output'],server_name="0.0.0.0", server_port=port, share=False)

if __name__ == "__main__":
    main()
