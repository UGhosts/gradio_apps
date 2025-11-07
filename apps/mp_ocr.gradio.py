import gradio as gr
import time
import os
import pandas as pd
from paddlex import create_model
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from PIL import Image
import sys
import json

os.environ["no_proxy"] = "localhost,127.0.0.1"

# 设置中文字体支持，确保负号能够正确显示
plt.rcParams["font.family"] = ["DejaVu Sans", "SimHei"]  # 优先使用能够正确显示负号的字体
# 全局变量记录上传的图片
uploaded_image = None  # 记录上传的图片


def process_input(selected_model_dir):
    """处理上传的图片，返回图表和结果"""
    # 确定使用上传的图片作为输入源
    input_source = uploaded_image

    # 验证输入
    if not input_source:
        return None, "错误: 请先上传一张图片"

    # 准备信息
    input_info = "上传的图片"
    model_info = f"模型目录: {selected_model_dir}"

    try:
        from paddlex import create_pipeline
        data = {}
        selected_model_dir = selected_model_dir + "/OCR.yaml"
        modeldir = selected_model_dir.replace('\\', '/')
        pipeline = create_pipeline(pipeline=modeldir)

        # 执行OCR识别
        output = pipeline.predict(
            input=input_source,
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        # 准备输出目录
        outdir = "../output/mp_ocr"
        os.makedirs(outdir, exist_ok=True)  # 确保输出目录存在

        # 保存结果
        for res in output:
            res.save_to_img(save_path=outdir)
            res.save_to_json(save_path=outdir)
            separator = os.sep
            # 为上传的图片生成唯一文件名
            timestamp = int(time.time())
            plot_title = input_source.split(separator)[-1].replace('.', '_ocr_res_img.')
            json_filename = input_source.split(separator)[-1].split('.')[0] + '_res.json'
            plot_img = os.path.join(outdir, plot_title)
            file_path = os.path.join(outdir, json_filename)

            # 读取JSON结果
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

        return plot_img, f"处理完成!\n {data['rec_texts']}"

    except Exception as e:
        return None, f"处理出错: {str(e)}\n{input_info}\n{model_info}"


def handle_image_upload(file):
    """处理图片上传，更新全局变量和预览"""
    global uploaded_image
    if file:
        uploaded_image = file.name  # 获取文件路径
        return Image.open(file.name)  # 返回PIL图像用于预览
    return None


def create_interface():
    # 从model/目录读取子目录作为模型选项
    model_dir = os.path.join(os.path.dirname(__file__), "model", "ocr")
    model_options = []  # 将使用元组列表: [(子目录名称, 完整路径)]

    if not os.path.exists(model_dir):
        # 尝试使用其他可能的路径
        alt_model_paths = [
            "../model/ocr",
            "./model/ocr",
            "model/ocr",
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

    with gr.Blocks(title="工业OCR") as demo:
        gr.Markdown("# 🚀 工业OCR")

        with gr.Row():
            with gr.Column(scale=1):
                # 图片上传区域（仅保留上传功能）
                gr.Markdown("### 上传图片")
                # 使用UploadButton替代Image组件，完全控制上传行为
                upload_button = gr.UploadButton(
                    "点击此处上传图片",
                    file_types=["image"],  # 仅允许图片类型
                    variant="secondary"
                )
                upload_preview = gr.Image(
                    label="上传预览",
                    type="pil",
                    interactive=False,
                    height=200,
                    width=300
                )

                # 绑定上传事件
                upload_button.upload(
                    fn=handle_image_upload,
                    inputs=[upload_button],
                    outputs=[upload_preview]
                )

                # 分隔线
                gr.Markdown("---")

                # 添加模型选择下拉框
                gr.Markdown("### 选择模型")
                model_dropdown = gr.Dropdown(
                    choices=model_options,
                    label="模型列表",
                    value=model_options[0][1] if model_options else ""  # 使用元组的第二个元素作为默认值
                )

                process_btn = gr.Button("处理", variant="primary")

            with gr.Column(scale=2):  # 扩大结果展示区域
                gr.Markdown("### OCR结果图")
                plot_output = gr.Image(label="OCR", type="pil")

                gr.Markdown("### 处理结果")
                output_text = gr.Textbox(label="识别结果", lines=6)

        # 处理按钮事件（返回图片和文本结果）
        process_btn.click(
            fn=process_input,
            inputs=[model_dropdown],
            outputs=[plot_output, output_text]
        )

    return demo


def main():
    # 从命令行参数获取端口号，如果未提供则使用默认端口7861
    port = 7861
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            if port < 1024 or port > 65535:
                print(f"警告：端口号 {port} 不在有效范围内(1024-65535)，将使用默认端口7861")
                port = 7861
        except ValueError:
            print(f"警告：无效的端口号参数 '{sys.argv[1]}'，将使用默认端口7861")

    demo = create_interface()
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, allowed_paths=[dataset_dir,'../output'])


if __name__ == "__main__":
    main()
