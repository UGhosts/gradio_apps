import gradio as gr
import time
import sys
import os
import json
import matplotlib.pyplot as plt
from paddlex import create_pipeline
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io

# 解决中文显示问题
#plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
#plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).parent.parent
# 临时注释（如果没有utils模块），可根据实际情况保留
from utils.app_utils import AppUtils as util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plt = util.auto_config_chinese_font()

# 全局变量
selected_preset = None


def preprocess_image(image_path):
    """
    图片预处理：
    1. 截取中间72%的面积（按宽高各取中间85%，0.85*0.85≈0.72）
    2. 缩放至2MB以内，分辨率宽高不超过1200
    3. 保存回原路径覆盖原文件
    """
    # 读取图片
    try:
        img = cv2.imread(image_path)
        if img is None:
            # 尝试用PIL读取（兼容更多格式）
            with Image.open(image_path) as pil_img:
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 获取原图尺寸
        h, w = img.shape[:2]
        print(f"原图尺寸: {w}x{h}")

        # 步骤1：截取中间72%面积（宽高各取中间85%，0.85*0.85≈0.72）
        crop_ratio = 0.85  # 单维度裁剪比例
        # 计算裁剪坐标
        x1 = int(w * (1 - crop_ratio) / 2)
        y1 = int(h * (1 - crop_ratio) / 2)
        x2 = int(w - x1)
        y2 = int(h - y1)
        # 裁剪图片
        cropped_img = img[y1:y2, x1:x2]
        crop_h, crop_w = cropped_img.shape[:2]
        print(f"裁剪后尺寸: {crop_w}x{crop_h}")

        # 步骤2：缩放限制（宽高≤1200）
        max_size = 1200
        scale = 1.0
        if crop_w > max_size or crop_h > max_size:
            # 计算缩放比例
            scale = min(max_size / crop_w, max_size / crop_h)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            # 缩放图片
            cropped_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_AREA)


        # 步骤3：控制文件大小≤2MB
        # 先保存到内存缓冲区，逐步调整质量
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 99]  # 默认高质量
        ext = os.path.splitext(image_path)[1].lower()

        # 根据文件扩展名选择编码格式
        if ext in ['.png', '.PNG']:
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 6]  # PNG压缩级别（0-9）
        elif ext in ['.webp', '.WEBP']:
            encode_params = [cv2.IMWRITE_WEBP_QUALITY, 99]

        # 循环调整直到文件大小≤2MB
        max_file_size = 2 * 1024 * 1024  # 2MB
        while True:
            # 保存到内存
            retval, buffer = cv2.imencode(ext, cropped_img, encode_params)
            file_size = len(buffer)

            if file_size <= max_file_size or encode_params[1] <= 10:
                break

            # 降低质量/提高压缩级别
            if ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.webp', '.WEBP']:
                encode_params[1] -= 5  # JPEG/WEBP降低质量
            elif ext in ['.png', '.PNG']:
                encode_params[1] += 1  # PNG提高压缩级别

        # 保存处理后的图片覆盖原文件
        with open(image_path, 'wb') as f:
            f.write(buffer)

        file_size_mb = len(buffer) / 1024 / 1024
        print(f"处理后文件大小: {file_size_mb:.2f}MB (质量/压缩级别: {encode_params[1]})")

        return True

    except Exception as e:
        print(f"图片预处理失败: {str(e)}")
        raise


def process_input(selected_model_dir):
    from paddlex import create_model

    """处理全局选中的测试文件，返回图表和结果"""
    time.sleep(1)
    preset_info = f"测试文件: {selected_preset}" if selected_preset else "未选择测试文件"
    model_info = f"模型目录: {selected_model_dir}"
    result = ''

    # 检查是否选择了测试文件
    if not selected_preset:
        return None, f"错误: 请先选择一个测试文件\n{preset_info}\n{model_info}"
    else:
        # ========== 新增：图片预处理 ==========
        try:
            print(f"\n开始预处理图片: {selected_preset}")
            preprocess_image(selected_preset)
            print("图片预处理完成")
        except Exception as e:
            return None, f"图片预处理失败: {str(e)}\n{preset_info}\n{model_info}"

        # ========== 原有OCR逻辑 ==========
        selected_model_dir = selected_model_dir + "/OCR.yaml"
        modeldir = selected_model_dir.replace('\\', '/')
        pipeline = create_pipeline(pipeline=modeldir)

        outdir = f"{BASE_DIR}/output/gangyin_ocr"
        os.makedirs(outdir, exist_ok=True)  # 确保输出目录存在

        # 执行OCR识别
        output = pipeline.predict(
            input=selected_preset,
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        # 处理识别结果
        all_results = []
        for res in output:
            res.print()  ## 打印预测的结构化输出
            res.save_to_img(save_path=outdir)
            res.save_to_json(save_path=outdir)

            separator = os.sep
            # 为上传的图片生成唯一文件名
            json_filename = selected_preset.split(separator)[-1].split('.')[0] + '_res.json'
            img_name = selected_preset.split(separator)[-1].split('.')[0] + '_ocr_res_img.' + \
                       selected_preset.split(separator)[-1].split('.')[1]

            # 读取JSON结果
            json_path = os.path.join(outdir, json_filename)
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    all_results.append(data)

        # 拼接结果
        final_img_path = os.path.join(outdir, img_name) if img_name else None
        final_result = json.dumps(all_results, ensure_ascii=False, indent=2) if all_results else "无识别结果"

        return final_img_path, final_result


def set_selected(file_path, buttons, file_paths):
    """更新选中状态，修改按钮样式并更新全局变量"""
    global selected_preset
    selected_preset = file_path

    # 返回所有按钮的样式更新列表
    return [gr.update(variant="primary" if fp == file_path else "secondary") for fp, btn in zip(file_paths, buttons)]


def create_interface():
    # 从dataset/目录动态读取CSV文件
    cwru_dir = os.path.join(os.path.dirname(__file__), "dataset", "tujiaoji_cls")
    preset_files = {}

    # 确保使用绝对路径或者正确的相对路径
    if not os.path.exists(cwru_dir):
        # 尝试使用其他可能的路径
        alt_paths = [
            f"{BASE_DIR}/dataset/gangyin_ocr",
            "./dataset/gangyin_ocr",
            "dataset/gangyin_ocr",
        ]
        for path in alt_paths:
            if os.path.exists(path):
                cwru_dir = path
                break

    # 获取目录下所有图片文件（筛选常见图片格式）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif']
    if os.path.exists(cwru_dir):
        for file_name in os.listdir(cwru_dir):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in image_extensions:
                file_path = os.path.join(cwru_dir, file_name)
                preset_files[file_path] = f"📄 {file_name}"

    model_dir = os.path.join(os.path.dirname(__file__), "model", "ocr")
    model_options = []  # 将使用元组列表: [(子目录名称, 完整路径)]

    if not os.path.exists(model_dir):
        # 尝试使用其他可能的路径
        alt_model_paths = [
            f"{BASE_DIR}/model/ocr",
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

    with gr.Blocks(title="钢印ocr识别应用") as demo:
        gr.Markdown("# 🚀 钢印ocr识别应用")

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
                gr.Markdown("### ocr查看")
                plot_output = gr.Image(label="视图", type="pil", height=400, width=700)

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
    demo.launch(allowed_paths=[f'{BASE_DIR}/output'], server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()