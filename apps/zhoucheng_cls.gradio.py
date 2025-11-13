import gradio as gr
import time
import os
import pandas as pd
from paddlex import create_model
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from PIL import Image
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置中文字体支持，确保负号能够正确显示
plt.rcParams["font.family"] = ["DejaVu Sans", "SimHei"]  # 优先使用能够正确显示负号的字体
# 全局变量记录选中的测试文件
selected_preset = None

class BearingCNN(nn.Module):
    def __init__(self, input_length, num_classes=5):
        super(BearingCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 计算卷积层输出大小
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            conv_output = self.conv_layers(dummy_input)
            conv_output_size = conv_output.view(1, -1).size(1)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def load_model_config(load_dir="../model/zhoucheng_cls/cnn"):
    """加载模型配置和标准化参数"""
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"未找到模型配置目录: {load_dir}")

    # 加载模型配置
    with open(os.path.join(load_dir, "model_config.json"), "r") as f:
        config = json.load(f)

    # 加载标准化参数（如果存在）
    scaler = None
    mean_path = os.path.join(load_dir, "mean.npy")
    std_path = os.path.join(load_dir, "std.npy")

    if os.path.exists(mean_path) and os.path.exists(std_path):
        scaler = StandardScaler()
        scaler.mean_ = np.load(mean_path)
        scaler.scale_ = np.load(std_path)

    return config["input_length"], config["num_classes"], scaler

def fft_transform(signals):
    """对振动信号进行FFT变换并归一化"""
    fft_results = []

    for signal in signals:
        # 计算FFT
        n = len(signal)
        fft_result = np.fft.fft(signal)
        # 取幅值并只保留正频率部分
        fft_mag = np.abs(fft_result)[:n // 2]

        # 归一化处理：除以信号长度进行幅度归一化
        fft_mag = fft_mag / n

        # 可选：进一步进行0-1归一化（避免除零错误）
        max_val = np.max(fft_mag)
        if max_val > 0:
            fft_mag = fft_mag / max_val

        fft_results.append(fft_mag)

    return np.array(fft_results)

def predict_new_data(model, scaler, new_signal, class_names, file_path):
    model.eval()
    try:
        # 重塑信号为二维数组，适应FFT处理
        signal = new_signal.reshape(1, -1)

        # 进行FFT变换
        fft_result = fft_transform(signal)

        # 绘制FFT结果图
        plt.figure(figsize=(10, 4))
        plt.plot(fft_result[0])
        plt.title('FFT频谱', fontsize=14)
        plt.xlabel('频率点', fontsize=12)
        plt.yticks([])
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        filepath = file_path + '_wave.png'
        filepath = filepath.replace('dataset','output')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        # 如果有标准化参数，则进行标准化
        if scaler is not None:
            fft_result = scaler.transform(fft_result)

        # 转换为Tensor并添加批次和通道维度
        input_tensor = torch.tensor(fft_result, dtype=torch.float32).unsqueeze(1).to(device)

        # 预测
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = output.cpu().numpy()[0]

        result_dict = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)}
        return result_dict

    except Exception as e:
        print(f"预测出错: {e}")
        return None

def predict_from_csv(model, scaler, class_names, file_path):
    """从CSV文件加载数据并预测"""
    try:
        # 读取文件
        df = pd.read_csv(file_path)
        first_column_name = df.columns[0]  # 获取第一列的列名
        signal = df[first_column_name].values  # 通过列名获取第一列数据
        return predict_new_data(model, scaler, signal, class_names, file_path)
    except Exception as e:
        print(f"从CSV文件预测出错: {e}")
        return None

def standalone_prediction(model_path, class_names, file_path):
    """独立的预测函数，可在其他脚本中调用"""

    # 加载模型配置和标准化参数
    input_length, num_classes, scaler = load_model_config()

    # 加载模型
    model = BearingCNN(input_length, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 预测
    result = predict_from_csv(model, scaler, class_names, file_path)
    return result



def plot_time_series(data, title="时序数据曲线"):
    """绘制时序曲线图"""
    plt.figure(figsize=(10, 4))
    # 假设数据包含'timestamp'和'value'列，根据实际格式调整
    plt.plot(data['Horizontal_vibration_signals'], 'b-', linewidth=2)
    plt.xlabel('时间')
    plt.ylabel('数值')
    plt.title(title)
    plt.xticks(rotation=45)

    # 设置y轴范围，确保能够显示负数
    if 'value' in data.columns:
        min_val = data['value'].min()
        max_val = data['value'].max()
        # 添加一些边距
        margin = (max_val - min_val) * 0.05
        plt.ylim(min_val - margin, max_val + margin)

    plt.tight_layout()


    # 保存到内存
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img


def process_input(selected_model_dir):
    """处理全局选中的测试文件，返回图表和结果"""
    time.sleep(1)
    preset_info = f"测试文件: {selected_preset}" if selected_preset else "未选择测试文件"
    model_info = f"模型目录: {selected_model_dir}"
    class_folders = ["ball", "inner", "keep", "ok", "outer"]



    # 检查是否选择了测试文件
    if not selected_preset:
        return None, f"错误: 请先选择一个测试文件\n{preset_info}\n{model_info}"
    else:
        data = pd.read_csv(selected_preset)
        # 绘制时序曲线图
        plot_title = f"时序曲线 - {os.path.basename(selected_preset)}"
        plot_img = plot_time_series(data, plot_title)
        rs = standalone_prediction(selected_model_dir + '/bearing_fault_5class_model.pth', class_folders,
                                   selected_preset)


        return plot_img,rs


def set_selected(file_path, buttons, file_paths):
    """更新选中状态，修改按钮样式并更新全局变量"""
    global selected_preset
    selected_preset = file_path

    # 返回所有按钮的样式更新列表
    # 对于每个按钮，如果它对应的文件路径与选中的文件路径相同，则设置为primary（高亮），否则设置为secondary（默认）
    return [gr.update(variant="primary" if fp == file_path else "secondary") for fp, btn in zip(file_paths, buttons)]


def create_interface():
    #
    cwru_dir = os.path.join(os.path.dirname(__file__), "dataset", "zhoucheng_cls")
    preset_files = {}

    # 确保使用绝对路径或者正确的相对路径
    if not os.path.exists(cwru_dir):
        # 尝试使用其他可能的路径
        alt_paths = [
            "../dataset/zhoucheng_cls",
            "./dataset/zhoucheng_cls",
            "dataset/zhoucheng_cls",
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

    model_dir = os.path.join(os.path.dirname(__file__), "model", "zhoucheng_cls")
    model_options = []  # 将使用元组列表: [(子目录名称, 完整路径)]

    if not os.path.exists(model_dir):
        # 尝试使用其他可能的路径
        alt_model_paths = [
            "../model/zhoucheng_cls",
            "./model/zhoucheng_cls",
            "model/zhoucheng_cls",
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
    if not model_options:
        default_model_name = "Timesnet_cls"
        default_model_dir = os.path.join(model_dir, default_model_name)
        model_options.append((default_model_name, default_model_dir))

    with gr.Blocks(title="西交-轴承故障诊断应用") as demo:
        gr.Markdown("# 🚀 西交-轴承故障诊断应用")

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
                gr.Markdown("### 时序曲线图")
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
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()