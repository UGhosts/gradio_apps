import gradio as gr
from paddlex import create_model
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- 全局OCR实例 ---
ocr_instance = None

# --- 模型目录配置 ---
# PaddleX 模型通常保存在一个目录中，该目录包含 model.pdmodel, model.pdiparams, 和 model.yml 等文件
MODEL_IMAGES_DIR = "/home/software/gradio_apps/model/ele_metric_ocr/PP-OCRv5_mobile_det"
def generate_model_options(base_dir: str) -> dict:
    """
    动态扫描指定目录，自动生成PaddleX的模型配置字典。
    一个包含模型文件的子目录代表一个完整的、可加载的模型。
    """
    if not os.path.isdir(base_dir):
        print(f"警告: 模型根目录 '{base_dir}' 不存在。将返回空配置。")
        return {}
    
    final_options = {}
    for item_name in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item_name)
        # 生成一个对用户更友好的显示名称
        display_name = item_name.replace('_', ' ').replace('-', ' ')
        # 值直接就是模型的完整路径
        final_options[display_name] = item_path

    if not final_options:
        print(f"警告: 在 '{base_dir}' 目录中未找到任何有效的 PaddleX 模型。")

    return final_options

# 动态生成模型选项
MODEL_OPTIONS = generate_model_options(MODEL_IMAGES_DIR)

# --- 示例图片管理 (不变) ---
EXAMPLE_IMAGES_DIR = "/home/software/gradio_apps/dataset/ele_metric_ocr"
EXAMPLE_IMAGES = []

def load_example_images():
    """加载示例图片列表"""
    global EXAMPLE_IMAGES
    EXAMPLE_IMAGES = []
    if os.path.exists(EXAMPLE_IMAGES_DIR):
        for filename in sorted(os.listdir(EXAMPLE_IMAGES_DIR)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                EXAMPLE_IMAGES.append(os.path.join(EXAMPLE_IMAGES_DIR, filename))

load_example_images()

# --- 文件监控与应用重启 (不变) ---
RESTART_SIGNAL_FILE = ".restart_signal"

def trigger_restart():
    """创建重启信号文件并终止当前应用进程。"""
    print("检测到文件变化，正在触发应用重启...")
    with open(RESTART_SIGNAL_FILE, "w") as f:
        f.write("restart")
    monitor_manager.stop_all(join_threads=False)
    print("应用进程即将退出...")
    os._exit(0)

class DirectoryHandler(FileSystemEventHandler):
    def on_created(self, event): trigger_restart()
    def on_deleted(self, event): trigger_restart()
    def on_moved(self, event): trigger_restart()

class MultiDirectoryMonitor:
    """一个可以管理多个目录监控任务的类。"""
    def __init__(self):
        self._directories_to_watch = set()
        self._observers = []
    def add_directory(self, path: str):
        abs_path = os.path.abspath(path)
        if abs_path not in self._directories_to_watch:
            self._directories_to_watch.add(abs_path)
            print(f"目录已注册监控: {path}")
    def start_all(self):
        if self._observers: return
        handler = DirectoryHandler()
        for path in self._directories_to_watch:
            os.makedirs(path, exist_ok=True)
            observer = Observer()
            observer.schedule(handler, path, recursive=True)
            self._observers.append(observer)
            observer.start()
        print(f"✅ 已启动对 {len(self._observers)} 个目录的监控。")
    def stop_all(self, join_threads: bool = True):
        for observer in self._observers:
            if observer.is_alive(): observer.stop()
        if join_threads:
            for observer in self._observers: observer.join()
        self._observers = []
        print("✅ 所有监控任务已停止。")

monitor_manager = MultiDirectoryMonitor()

def get_current_examples():
    """获取当前示例图片列表（格式化为Gallery需要的格式）"""
    print(f"当前示例图片数量: {len(EXAMPLE_IMAGES)}")
    return [[path, ""] for path in EXAMPLE_IMAGES] if EXAMPLE_IMAGES else []

def initialize_ocr(model_choice):
    """根据用户选择初始化PaddleX OCR模型"""
    global ocr_instance
    try:
        if model_choice not in MODEL_OPTIONS:
            return f"✗ 未找到模型配置: {model_choice}"

        model_path = MODEL_OPTIONS[model_choice]
        if not os.path.isdir(model_path):
            return f"✗ 模型路径不存在: {model_path}"
        
        # 使用 PaddleX 加载模型，可以指定使用GPU
        ocr_instance= create_model(model_name=" PP-OCRv5_mobile_det", model_dir=model_path)
            
        return f"✓ 模型 {model_choice} 初始化成功"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"✗ 模型初始化失败: {str(e)}"

# --- 图像预处理 (不变) ---
MAX_OCR_IMAGE_SIZE = 1280 

def resize_image_for_ocr(image, max_long_side=MAX_OCR_IMAGE_SIZE):
    """
    将图片等比例缩放到适合OCR处理的尺寸。
    """
    h, w, _ = image.shape
    if h <= max_long_side and w <= max_long_side:
        return image, 1.0

    if h > w:
        ratio = max_long_side / h
        new_h, new_w = max_long_side, int(w * ratio)
    else:
        ratio = max_long_side / w
        new_w, new_h = max_long_side, int(h * ratio)
        
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image, ratio


def draw_ocr_results(image_path, model_choice):
    """
    使用 PaddleX 进行 OCR 并绘制结果。
    """
    global ocr_instance
    
    if not os.path.exists(image_path):
        return None, "错误: 图片未找到。"

    if ocr_instance is None:
        status = initialize_ocr(model_choice)
        if "失败" in status:
            return None, status

    original_image = cv2.imread(image_path)
    if original_image is None:
        return None, "错误: 无法读取图片文件。"

    try:
        # 1. 预处理图片用于OCR
        processed_image, scale_ratio = resize_image_for_ocr(original_image)
        print(f"图片尺寸已从 {original_image.shape[:2]} 预处理为 {processed_image.shape[:2]}，缩放比例: {scale_ratio:.4f}")
        
        # 2. 使用 PaddleX 执行OCR识别
        # PaddleX 的 predict 方法可以直接处理 numpy 数组
        result = ocr_instance.predict(processed_image)
        
        # 3. 准备绘制
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 25)
        except IOError:
            font = ImageFont.load_default()
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        # 在原始尺寸的图片上进行绘制
        pil_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        final_text_label = "未检测到文本"

        if result:
            # 4. 移植并适配原有的筛选逻辑
            # 第一步：初步筛选
            filtered_data = [
                item for item in result
                if len(item['text']) > 3 and (len(item['text']) < 7 or '.' in item['text'])
            ]

            # 第二步：如果结果超过2个，进一步筛选以0开头的
            if len(filtered_data) > 2:
                zero_start_data = [item for item in filtered_data if item['text'].startswith('0')]
                if zero_start_data:
                    filtered_data = zero_start_data

            # 第三步：如果仍有多个结果，保留第一个
            if len(filtered_data) > 1:
                filtered_data = [filtered_data[0]]

            # 5. 遍历筛选后的结果并绘制gr
            if not filtered_data:
                final_text_label = "未识别到有效文本"

            for idx, item in enumerate(filtered_data):
                text = item['text'].strip().lstrip('0')
                confidence = item['score']
                
                # 将检测框的坐标从缩放后的图像尺寸还原到原始图像尺寸
                points = np.array(item['polygon'])
                if scale_ratio != 1.0:
                    points = (points / scale_ratio).astype(np.int32)
                
                if not text or confidence < 0.5:
                    continue
                
                color = colors[idx % len(colors)]
                text_label = f'{text}   可信度: {confidence:.1%}'
                final_text_label = text_label # 更新状态文本

                # 绘制边框
                draw.polygon([tuple(p) for p in points], outline=color, width=3)
                
                # 绘制文本背景和文本
                text_position = (int(points[0][0]), max(0, int(points[0][1]) - 30))
                padding = 5
                try:
                    text_bbox = draw.textbbox(text_position, text_label, font=font)
                    padded_bbox = [
                        text_bbox[0] - padding, text_bbox[1] - padding,
                        text_bbox[2] + padding, text_bbox[3] + padding
                    ]
                    draw.rectangle(padded_bbox, fill=(2, 166, 13))
                except Exception:
                    text_width, text_height = len(text_label) * 12, 25
                    simple_bbox = [
                        text_position[0] - padding, text_position[1] - padding,
                        text_position[0] + text_width + padding, text_position[1] + text_height + padding
                    ]
                    draw.rectangle(simple_bbox, fill=(166, 43, 90))
                
                draw.text(text_position, text_label, fill=(255, 255, 255), font=font)
        
        result_image = np.array(pil_image)
        return result_image, final_text_label
            
    except Exception as e:
        print(f"OCR 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        # 返回原始图像和错误信息
        error_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        return error_image, f"OCR 处理出错: {str(e)}"

# --- Gradio 界面逻辑函数 (基本不变) ---
def ocr_image(image_path, model_choice):
    """主处理函数"""
    if image_path is None:
        return None, "请先上传或选择图片"
    if not MODEL_OPTIONS:
        return None, "错误：没有可用的模型。请检查模型目录配置。"
    return draw_ocr_results(image_path, model_choice)

def clear_outputs():
    """清空所有输出"""
    return gr.update(value=None), gr.update(value=None), gr.update(value="")

def change_model(model_choice):
    """切换模型时的回调"""
    global ocr_instance
    ocr_instance = None  # 重置OCR实例，强制在下次识别时重新初始化
    return f"已选择模型: {model_choice}，下次识别时将自动加载"

# health check JS (不变)
health_check_js = '''
() => {
    let isConnected = true;
    setInterval(async () => {
        try {
            await fetch('/app_id');
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

# --- 创建 Gradio 界面 (不变) ---
with gr.Blocks(title="PaddleX 智能文字识别", theme=gr.themes.Default(), js=health_check_js) as iface:
    gr.Markdown("""
    # 🔍 电表读数OCR (PaddleX版)
    **功能特点：** 支持多种PaddleX模型、提供示例图片、实时可视化识别结果、自动监控示例与模型目录
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ 示例图片")
            initial_examples = get_current_examples()
            example_gallery = gr.Gallery(
                value=initial_examples,
                label="点击选择示例", show_label=False, elem_id="example_gallery",
                columns=4, rows=1, height=200, allow_preview=False
            )
            
            if not initial_examples:
                gr.Markdown("<p style='color:orange;'>*没有找到示例图片，请在 ./examples/ 目录下添加图片文件*</p>")
            
            gr.Markdown("### 🤖 模型选择")
            model_selector = gr.Dropdown(
                choices=list(MODEL_OPTIONS.keys()),
                value=list(MODEL_OPTIONS.keys())[0] if MODEL_OPTIONS else None,
                label="选择OCR模型",
                info="自动扫描并加载指定目录下的PaddleX模型"
            )
            if not MODEL_OPTIONS:
                gr.Markdown("<p style='color:red;'>*警告：未找到任何有效的PaddleX模型，请检查模型目录！*</p>")

            with gr.Accordion("📖 使用说明", open=True):
                gr.Markdown("""
                **操作步骤：**
                1. 从下拉菜单选择一个OCR模型。
                2. 上传图片或点击一个示例图片。
                3. 点击"开始识别"按钮。
                4. 在右侧查看识别结果和可视化标注。
                """)

        with gr.Column(scale=1):
            gr.Markdown("### 📤 上传图片")
            input_image = gr.Image(type="filepath", label="上传图片", height=200, sources=['upload'])
            
            with gr.Row():
                submit_btn = gr.Button("🚀 开始识别", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ 清空", variant="secondary")
            
            gr.Markdown("### 📋 识别结果")
            output_image = gr.Image(label="识别结果可视化", height=600)
            result_status = gr.Textbox(label="识别结果", interactive=False)

    # --- 事件绑定 (不变) ---
    def select_example(evt: gr.SelectData):
        """当用户点击示例图片时调用"""
        current_examples = get_current_examples()
        path_to_return = None
        if current_examples and evt.index < len(current_examples):
            path_to_return = current_examples[evt.index][0]
        return gr.update(value=path_to_return)

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

# --- 启动应用 (不变) ---
if __name__ == "__main__":
    os.makedirs(EXAMPLE_IMAGES_DIR, exist_ok=True)
    os.makedirs(MODEL_IMAGES_DIR, exist_ok=True)
    
    # 启动目录监控
    monitor_manager.add_directory(EXAMPLE_IMAGES_DIR)
    monitor_manager.add_directory(MODEL_IMAGES_DIR)
    monitor_manager.start_all()
    
    try:
        iface.launch(
            server_name="0.0.0.0",
            server_port=1869,
            share=False,
            debug=True,
            show_error=True
        )
    finally:
        monitor_manager.stop_all(join_threads=True)