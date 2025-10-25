import gradio as gr
import paddlex as pdx
import os,sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json
from utils.app_utils import AppUtils as util
from utils.app_utils import MultiDirectoryMonitor
from pathlib import Path
# --- 全局OCR实例 ---
ocr_instance = None

from pathlib import Path

# --- 模型目录配置 ---
BASE_DIR = Path(__file__).parent.parent
MODEL_BASE_DIR = BASE_DIR / "model" / "ele_metric_ocr" / "model"
RESTART_SIGNAL_FILENAME = ".restart_signal_ele_metric"
EXAMPLE_DIR = BASE_DIR / "model" / "ele_metric_ocr" / "example"


model_options = util.generate_paddlex_model_options(MODEL_BASE_DIR)


# --- 重启信号和文件监控处理 ---


def initialize_ocr(model_choice):
    """根据用户选择初始化PaddleX OCR模型"""
    global ocr_instance
    try:
        print(model_options)
        models_config = model_options[model_choice]
        
        # 第一次尝试初始化（如果失败会直接进入except）
        ocr_instance = pdx.create_pipeline(models_config)
        return "✓ 模型初始化成功"
        
    except Exception as first_error:
        # 如果第一次初始化失败，尝试从配置文件的同级目录加载.yaml文件
        try:
            current_path = Path(models_config).parent.parent / "pipeline"
            yaml_files = [file for file in current_path.iterdir() if file.suffix == ".yaml"]
            
            if not yaml_files:
                raise Exception("未找到可用的.yaml配置文件")
                
            # 尝试加载第一个.yaml文件（如果失败会抛出异常）
            ocr_instance = pdx.create_pipeline(str(yaml_files[0]))  # 确保传入字符串路径
            return "✓ 从备用.yaml文件初始化成功"
            
        except Exception as fallback_error:
            print(
                f"✗ 初始化模型失败:\n"
                f"- 主配置失败: {str(first_error)}\n"
                f"- 备用.yaml文件失败: {str(fallback_error)}"
            )
            return None


MAX_OCR_IMAGE_SIZE = 1280 

def resize_image_for_ocr(image, max_long_side=MAX_OCR_IMAGE_SIZE):
    """
    将图片等比例缩放到适合OCR处理的尺寸。
    """
    h, w = image.shape[:2] if len(image.shape) >= 2 else (0, 0)
    
    if h == 0 or w == 0:
        return image, 1.0

    # 如果图片尺寸已经小于等于阈值，则无需处理
    if h <= max_long_side and w <= max_long_side:
        return image, 1.0

    # 计算缩放比例
    if h > w:
        ratio = max_long_side / h
        new_h = max_long_side
        new_w = int(w * ratio)
    else:
        ratio = max_long_side / w
        new_w = max_long_side
        new_h = int(h * ratio)
        
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized_image, ratio

def draw_ocr_results(image_path, model_choice):
    global ocr_instance
    
    if not os.path.exists(image_path):
        return None, "错误: 图片未找到。"

    # 如果OCR实例不存在或模型发生变化，重新初始化
    if ocr_instance is None:
        initialize_ocr(model_choice)

    try:
        # 使用 OpenCV 读取原始图像
        original_image = cv2.imread(image_path)
        if original_image is None:
            return None, "错误: 无法读取图片文件。"
        processed_image, scale_ratio = resize_image_for_ocr(original_image)
        print(f"图片尺寸已从 {original_image.shape[:2]} 预处理为 {processed_image.shape[:2]}，缩放比例: {scale_ratio:.4f}")
        # 执行OCR识别 - 兼容不同的方法
        if hasattr(ocr_instance, 'predict'):
            # 使用您原来的predict方法
            result = ocr_instance.predict(processed_image)
        else:
            # 使用标准的ocr方法
            result = ocr_instance.ocr(processed_image, cls=True)
        
        try:
            # 尝试加载中文字体
            font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 25)
        except IOError:
            try:
                # 备选字体
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 25)
            except IOError:
                font = ImageFont.load_default()
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # 处理不同格式的结果
        result = list(result) 
        if result and len(result) > 0:
            # 检查是否是原来的predict方法返回的格式
            if isinstance(result[0], dict) and 'rec_texts' in result[0]:
                # 使用原来的处理逻辑
                page_result = result[0]
                rec_texts = page_result.get('rec_texts', [])
                rec_scores = page_result.get('rec_scores', [])
                rec_polys = page_result.get('rec_polys', [])
                
                # 同步筛选，只保留 rec_texts 长度 > 4 的项
                filtered_data = [
                    (text, score, poly) 
                    for text, score, poly in zip(rec_texts, rec_scores, rec_polys) 
                    if len(text) > 3 and (len(text) < 7 or '.' in text)
                ]

                # 第二步：如果结果超过2个，进一步筛选以0开头的
                if len(filtered_data) > 2:
                    zero_start_data = [(text, score, poly) for text, score, poly in filtered_data if text.startswith('0')]
                    if zero_start_data:
                        filtered_data = zero_start_data

                # 第三步：如果仍有多个结果，保留第一个
                if len(filtered_data) > 1:
                    filtered_data = [filtered_data[0]]
                
                # 解包回各自的列表
                rec_texts, rec_scores, rec_polys = zip(*filtered_data) if filtered_data else ([], [], [])
                
                output_img = page_result['doc_preprocessor_res'].get('output_img')
                if len(output_img.shape) == 3 and output_img.shape[2] == 3:
                    # 假设是BGR格式，转换为RGB
                    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(output_img)
                draw = ImageDraw.Draw(pil_image)
                
                if not (rec_texts and rec_scores and rec_polys):
                    return np.array(pil_image), "未识别到文本"
                
                recognition_count = 0
                for idx in range(len(rec_texts)):
                    text = rec_texts[idx].strip().lstrip('0')
                    confidence = rec_scores[idx]
                    points = rec_polys[idx]
                    
                    # 过滤低置信度结果
                    if not text or confidence < 0.5:
                        continue
                    
                    recognition_count += 1
                    color = colors[idx % len(colors)]
                    text_label = f'{text}   可信度: {confidence:.1%}'
                    
                    # 绘制边框
                    draw.polygon([tuple(p) for p in points], outline=color, width=3)
                    
                    # 绘制文本标签
                    text_position = (int(points[0][0]), max(0, int(points[0][1]) - 30))
                    padding = 5
                    
                    try:
                        # 计算文本背景框
                        text_bbox = draw.textbbox(text_position, text_label, font=font)
                        padded_bbox = [
                            text_bbox[0] - padding,
                            text_bbox[1] - padding,
                            text_bbox[2] + padding,
                            text_bbox[3] + padding
                        ]
                        draw.rectangle(padded_bbox, fill=(2, 166, 13))
                    except Exception as e:
                        # 简单背景框
                        text_width = len(text_label) * 12
                        text_height = 25
                        simple_bbox = [
                            text_position[0] - padding,
                            text_position[1] - padding,
                            text_position[0] + text_width + padding,
                            text_position[1] + text_height + padding
                        ]
                        draw.rectangle(simple_bbox, fill=(166, 43, 90))
                    
                    # 绘制文本
                    draw.text(text_position, text_label, fill=(255, 255, 255), font=font)
                    
            else:
                # 标准OCR结果格式处理
                # 转换为RGB格式的PIL图像
                pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                
                recognition_count = 0
                for idx, line in enumerate(result[0] if result[0] else []):
                    if len(line) >= 2:
                        points = np.array(line[0], dtype=np.int32)
                        text = line[1][0] if isinstance(line[1], tuple) else str(line[1])
                        confidence = line[1][1] if isinstance(line[1], tuple) and len(line[1]) > 1 else 1.0
                        
                        # 过滤低置信度结果
                        if confidence < 0.5 or not text.strip():
                            continue
                        
                        recognition_count += 1
                        color = colors[idx % len(colors)]
                        text_label = f'{text}  [{confidence:.1%}]'
                        
                        # 绘制边框
                        draw.polygon([tuple(p) for p in points], outline=color, width=2)
                        
                        # 绘制文本标签
                        text_position = (int(points[0][0]), max(0, int(points[0][1]) - 25))
                        
                        # 计算文本背景框
                        try:
                            bbox = draw.textbbox(text_position, text_label, font=font)
                            padding = 3
                            bg_bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]
                            draw.rectangle(bg_bbox, fill=(0, 0, 0, 180))
                        except:
                            # 简单背景框
                            text_width = len(text_label) * 10
                            bg_bbox = [text_position[0]-3, text_position[1]-3, 
                                     text_position[0]+text_width+3, text_position[1]+20]
                            draw.rectangle(bg_bbox, fill=(0, 0, 0))
                        
                        # 绘制文本
                        draw.text(text_position, text_label, fill=(255, 255, 255), font=font)
            
            status_msg = f"{text_label}" if recognition_count > 0 else "未识别到有效文本"
        else:
            # 如果没有识别结果，返回原始图像
            pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            status_msg = "未检测到文本"
        
        result_image = np.array(pil_image)
        return result_image, status_msg
            
    except Exception as e:
        print(f"OCR 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        try:
            # 返回原始图像和错误信息
            image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            return image, f"OCR 处理出错: {str(e)}"
        except:
            return None, f"OCR 处理出错: {str(e)}"

def ocr_image(image_path, model_choice):
    """主处理函数"""
    if image_path is None:
        return None, "请先上传或选择图片"
    return draw_ocr_results(image_path, model_choice)

def load_example(example_path):
    """加载示例图片"""
    return example_path

def clear_outputs():
    """清空所有输出"""
    return None, None, ""

def change_model(model_choice):
    """切换模型时的回调"""
    global ocr_instance
    ocr_instance = None  # 重置OCR实例，强制重新初始化
    return f"已选择模型: {model_choice}，下次识别时将自动加载"

def refresh_examples():
    """手动刷新示例图片列表"""
    load_example_images()
    examples = get_current_examples()
    status_msg = f"已刷新，找到 {len(EXAMPLE_IMAGES)} 张示例图片"
    if not examples:
        status_msg = "*没有找到示例图片，请在 ./examples/ 目录下添加图片文件*"
    return examples, status_msg

def create_gradio_interface():
    health_check_js = '''
    () => {
        let isConnected = true;
        setInterval(async () => {
            try {
                await fetch('/');
                if (!isConnected) {
                    console.log("成功重新连接到服务器，正在刷新页面...");
                    location.reload();
                }
                isConnected = true;
            } catch (e) {
                if (isConnected) {
                    console.log("与服务器的连接已断开，等待重新连接...");
                }
                isConnected = false;
            }
        }, 2000);
    }
    '''

    # --- 创建 Gradio 界面 ---
    with gr.Blocks(title="PaddleX 智能文字识别", theme=gr.themes.Default(), js=health_check_js) as iface:
        gr.Markdown("""
        # 🔍 电表读数OCR
        **功能特点：** 基于PaddleX框架的OCR识别、支持多种模型选择、提供示例图片、实时可视化识别结果
        """)
        
        with gr.Row():
            # 左侧：示例图片和模型选择
            with gr.Column(scale=1):
                gr.Markdown("### 🖼️ 示例图片")
                with gr.Row():
                    gr.Markdown("点击下方示例图片快速体验识别效果：")
                

                example_gallery = gr.Gallery(
                    value=util.get_current_examples(EXAMPLE_DIR),
                    label="点击选择示例",
                    show_label=False,
                    elem_id="example_gallery",
                    columns=4,
                    rows=1,
                    height=200,
                    allow_preview=False
                )
                    
                gr.Markdown("### 🤖 模型选择")
                model_selector = gr.Dropdown(
                        choices=list(model_options.keys()),
                        value=list(model_options.keys())[0] if model_options else None,
                    label="选择OCR模型",
                    info="支持PaddleX预训练模型和自定义模型"
                )

                with gr.Accordion("📖 使用说明", open=True):
                    gr.Markdown("""
                    **操作步骤：**
                    1. 选择合适的OCR模型
                    2. 上传图片或选择示例图片
                    3. 点击"开始识别"按钮
                    4. 查看识别结果和可视化标注
                    
                    **支持格式：** JPG, PNG, JPEG
                    
                    **基于框架：** PaddleX
                    """)
                
            # 右侧：上传和结果
            with gr.Column(scale=1):
                gr.Markdown("### 📤 上传图片")
                input_image = gr.Image(
                    type="filepath", 
                    label="上传图片",
                    height=200,
                    sources=['upload']
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 开始识别", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ 清空", variant="secondary")
                
                gr.Markdown("### 📋 识别结果")
                output_image = gr.Image(label="识别结果可视化", height=600)
                result_status = gr.Textbox(label="识别结果", interactive=False)

        # 事件绑定
        def select_example(evt: gr.SelectData):
            current_examples = util.get_current_examples(EXAMPLE_DIR)
            if current_examples and evt.index < len(current_examples):
                selected_path = current_examples[evt.index]
                return selected_path
            return None

        example_gallery.select(select_example, None, outputs=[input_image])
        
        submit_btn.click(
            fn=ocr_image,
            inputs=[input_image, model_selector],
            outputs=[output_image, result_status]
        )
        
        clear_btn.click(
            fn=clear_outputs,
            outputs=[input_image, output_image, result_status]
        )
        
        model_selector.change(
            fn=change_model,
            inputs=[model_selector]
        )
        return iface
def main():
    monitor_manager = MultiDirectoryMonitor(restart_signal_file_name=RESTART_SIGNAL_FILENAME)
    monitor_manager.add_directory(MODEL_BASE_DIR)
    monitor_manager.add_directory(EXAMPLE_DIR)
    if not monitor_manager.start_all():
        print("❌ 启动目录监控失败")
        return
    port = 7861
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            if port < 1024 or port > 65535:
                print(f"警告：端口号 {port} 不在有效范围内(1024-65535)，将使用默认端口{port}")
                port = port
        except ValueError:
            print(f"警告：无效的端口号参数 '{sys.argv[1]}'，将使用默认端口{port}")
    iface = create_gradio_interface()
    try:
        iface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
        )
    finally:
        # 应用关闭时停止监控
        monitor_manager.stop_all(join_threads=True)
    
    
# 启动应用
if __name__ == "__main__":
    main()
    
