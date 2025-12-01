import gradio as gr
import time
import os
import pandas as pd
from paddlex import create_model
from io import BytesIO, StringIO
from PIL import Image
import sys
from utils.app_utils import AppUtils as util
# 设置中文字体支持，确保负号能够正确显示
plt = util.auto_config_chinese_font()
# 全局变量记录选中的测试文件
selected_preset = None
import logging
# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_time_series(data, title="时序数据曲线"):
    """绘制时序曲线图"""
    plt.figure(figsize=(10, 4))
    # 假设数据包含'timestamp'和'value'列，根据实际格式调整
    plt.plot(data['time'], data['value'], 'b-', linewidth=2)
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

    # 检查是否选择了测试文件
    if not selected_preset:
        return None, f"错误: 请先选择一个测试文件\n{preset_info}\n{model_info}"
    
    # 检查paddle相关模块是否可用
    if not paddle_available:
        data = pd.read_csv(selected_preset)
        plot_title = f"时序曲线 - {os.path.basename(selected_preset)}"
        plot_img = plot_time_series(data, plot_title)
        
        # 提供友好的错误提示
        error_msg = ("错误: 缺少必要的PaddlePaddle相关依赖!\n" 
                    "请按照以下步骤安装所需依赖:\n" 
                    "1. 打开environment_win.yml文件\n" 
                    "2. 移除以下三行前面的注释符号(#):\n" 
                    "   - paddlepaddle-gpu==3.0.0\n" 
                    "   - paddlets==1.1.0\n" 
                    "   - paddlex==3.0.2\n" 
                    "3. 运行命令: conda env update -f environment_win.yml\n" 
                    "4. 重新启动应用程序\n\n" 
                    f"{preset_info}\n{model_info}")
        return plot_img, error_msg
    
    try:
        data = pd.read_csv(selected_preset)
        # 绘制时序曲线图
        plot_title = f"时序曲线 - {os.path.basename(selected_preset)}"
        plot_img = plot_time_series(data, plot_title)
        model = create_model(model_name="TimesNet_cls", model_dir=selected_model_dir)
        output = model.predict(selected_preset, batch_size=1)

        # 保存预测结果并处理显示
        result_df = None
        for res in output:
            logging.info(res.print(json_format=True))
            res.save_to_csv(save_path="./output/cwru_cls/")
            res.save_to_json(save_path="./output/cwru_cls/res.json")

            # 将预测结果转换为DataFrame（根据实际返回格式调整）
            # 假设res包含classid和score字段
            if hasattr(res, 'classid') and hasattr(res, 'score'):
                result_df = pd.DataFrame({
                    '样本ID': range(len(res.classid)),
                    '类别ID': res.classid,
                    '置信度': [f"{score:.4f}" for score in res.score]  # 保留4位小数
                })
            elif isinstance(res, dict) and 'classification' in res:
                # 处理字典类型结果
                cls_data = res['classification']
                result_df = pd.DataFrame(cls_data).rename(columns={
                    'classid': '类别ID',
                    'score': '置信度'
                })
                result_df['置信度'] = result_df['置信度'].apply(lambda x: f"{x:.4f}")
                result_df.insert(0, '样本ID', result_df.index)

        # 格式化输出结果
        if result_df is not None:
            # 使用to_string()美化表格显示
            result_str = "预测结果：\n" + result_df.to_string(index=False)
        else:
            result_str = "未获取到有效预测结果"
        return plot_img, f"处理完成!\n{preset_info}\n{model_info}\n\n{result_str}"
    except Exception as e:
        # 处理其他可能的异常
        data = pd.read_csv(selected_preset)
        plot_title = f"时序曲线 - {os.path.basename(selected_preset)}"
        plot_img = plot_time_series(data, plot_title)
        return plot_img, f"处理过程中发生错误: {str(e)}\n{preset_info}\n{model_info}"


def set_selected(file_path, buttons, file_paths):
    """更新选中状态，修改按钮样式并更新全局变量"""
    global selected_preset
    selected_preset = file_path

    # 返回所有按钮的样式更新列表
    # 对于每个按钮，如果它对应的文件路径与选中的文件路径相同，则设置为primary（高亮），否则设置为secondary（默认）
    return [gr.update(variant="primary" if fp == file_path else "secondary") for fp, btn in zip(file_paths, buttons)]


def create_interface():
    # 从dataset/cwru_cls_test目录动态读取CSV文件
    cwru_dir = os.path.join(os.path.dirname(__file__), "dataset", "cwru_cls")
    preset_files = {}

    # 确保使用绝对路径或者正确的相对路径
    if not os.path.exists(cwru_dir):
        # 尝试使用其他可能的路径
        alt_paths = [
            "../dataset/cwru_cls",
            "./dataset/cwru_cls",
            "dataset/cwru_cls",
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
        preset_files = {"dataset/cwru_cls/cwru_cls_7.csv": "📄 cwru_cls_7.csv"}

    # 从model/cwru_cls目录读取子目录作为模型选项
    model_dir = os.path.join(os.path.dirname(__file__), "model", "cwru_cls")
    model_options = []  # 将使用元组列表: [(子目录名称, 完整路径)]

    if not os.path.exists(model_dir):
        # 尝试使用其他可能的路径
        alt_model_paths = [
            "../model/cwru_cls",
            "./model/cwru_cls",
            "model/cwru_cls",
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

    with gr.Blocks(title="轴承故障诊断应用") as demo:
        gr.Markdown("# 🚀 轴承故障诊断应用")

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
                output_text = gr.Textbox(label="结果信息", lines=10, interactive=False)

        # 处理按钮事件（返回图片和文本结果）
        process_btn.click(
            fn=process_input,
            inputs=[model_dropdown],
            outputs=[plot_output, output_text]
        )

    return demo


def main():
    # 从命令行参数获取端口号，如果未提供则使用默认端口7860
    port = 7860
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            if port < 1024 or port > 65535:
                logging.warning(f"警告：端口号 {port} 不在有效范围内(1024-65535)，将使用默认端口7860")
                port = 7860
        except ValueError:
            logging.warning(f"警告：无效的端口号参数 '{sys.argv[1]}'，将使用默认端口7860")

    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()