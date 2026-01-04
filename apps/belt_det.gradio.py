import gradio as gr
import paddlex as pdx
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import logging
import subprocess
from utils.app_utils import AppUtils as util
from utils.app_utils import MultiDirectoryMonitor

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 全局检测实例 ---
detector_instance = None

# --- 目录配置 ---
BASE_DIR = Path(__file__).parent.parent
MODEL_BASE_DIR = BASE_DIR / "model" / "belt_det" / "model"
RESTART_SIGNAL_FILENAME = ".restart_signal_belt_det"
EXAMPLE_DIR = BASE_DIR / "dataset" / "belt_det"
OUTPUT_DIR = BASE_DIR / "model" / "belt_det" / "output"

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型选项
model_options = util.generate_paddlex_model_options(MODEL_BASE_DIR)


class VideoDetector:
    """视频/图片目标检测器类"""
    
    def __init__(self, model_dir, threshold=0.3):
        """初始化检测器"""
        self.model_dir = model_dir
        self.threshold = threshold
        self.predictor = None
        self.class_names = []
        self._load_model()
    
    def _load_model(self):
        """加载推理模型"""
        try:
            logging.info(f"Loading model from {self.model_dir}...")
            model_path = Path(self.model_dir)
            model_name = model_path.name
            self.predictor = pdx.create_model(
                model_name=model_name, 
                model_dir=self.model_dir
            )
            
            if hasattr(self.predictor, 'labels'):
                self.class_names = self.predictor.labels
                logging.info(f"Model classes: {self.class_names}")
            
            logging.info("Model loaded successfully!")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def detect_image(self, image_path):
        """检测单张图片"""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image from {image_path}")
        
        vis_frame, num_detections, detection_info = self._detect_frame(frame)
        return vis_frame, num_detections, detection_info
    
    def detect_video(self, video_path, progress=gr.Progress()):
        """检测视频"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Video: {width}x{height} @ {fps}FPS, {total_frames} frames")
        
        temp_output = OUTPUT_DIR / f"temp_{Path(video_path).stem}.mp4"
        final_output = OUTPUT_DIR / f"detected_{Path(video_path).stem}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        writer = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            cap.release()
            raise ValueError("Cannot initialize video writer")
        
        frame_count = 0
        total_detections = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1

                vis_frame, num_detections, _ = self._detect_frame(frame)
                total_detections += num_detections
                
                cv2.putText(vis_frame, f"Frame: {frame_count}/{total_frames}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Detections: {num_detections}", 
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                writer.write(vis_frame)
                
                if total_frames > 0:
                    progress_value = frame_count / total_frames
                    progress_text = f"处理中 {frame_count}/{total_frames}"
                    progress(progress_value, desc=progress_text)
        
        finally:
            cap.release()
            writer.release()
        
        # 转换为H.264格式
        if temp_output.exists():
            try:
                cmd = [
                    'ffmpeg', '-y', '-i', str(temp_output),
                    '-c:v', 'libx264', '-preset', 'medium',
                    '-crf', '23', '-pix_fmt', 'yuv420p',
                    str(final_output)
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                temp_output.unlink()
                logging.info(f"Video saved: {final_output}")
            except subprocess.CalledProcessError as e:
                error_message = e.stderr.decode() if e.stderr else str(e)
                logging.error(f"FFmpeg 转换失败，详细原因: \n{error_message}")
                final_output = temp_output
            except Exception as e:
                logging.error(f"视频处理未知错误: {e}")
                final_output = temp_output
        
        avg_det = total_detections / frame_count if frame_count > 0 else 0
        status = f"处理完成！总帧数: {frame_count}, 总检测数: {total_detections}, 平均检测/帧: {avg_det:.2f}"
        
        return str(final_output), status
    
    def _detect_frame(self, frame):
        """对单帧进行检测"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            result_generator = self.predictor.predict(img_rgb)
            result = next(result_generator)
            vis_frame, num_detections, detection_info = self._draw_detections(frame, result)
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return frame, 0, []
        
        return vis_frame, num_detections, detection_info
    
    def _draw_detections(self, frame, result):
        """绘制检测结果"""
        vis_frame = frame.copy()
        
        if isinstance(result, dict):
            boxes = result.get('boxes', [])
        elif hasattr(result, 'boxes'):
            boxes = result.boxes
        else:
            return vis_frame, 0, []
        
        if not boxes or len(boxes) == 0:
            return vis_frame, 0, []
        
        num_detections = 0
        detection_info = []
        
        # 转换为PIL格式以支持中文 - 放在最开始
        pil_img = Image.fromarray(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 加载中文字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 24)
        except:
            font = ImageFont.load_default()
        
        # 绘制检测框
        for box_dict in boxes:
            try:
                score = float(box_dict['score'])
                label = box_dict['label']
                cls_id = int(box_dict['cls_id'])
                coordinate = box_dict['coordinate']
                
                if score < self.threshold:
                    continue
                
                num_detections += 1
                x1, y1, x2, y2 = map(int, coordinate)
                
                # 确保坐标在范围内
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                color = self._get_color(cls_id)
                
                # 使用 ImageDraw 绘制矩形框（而不是 cv2.rectangle）
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
                
                # 准备文本
                text = f"{label}: {score:.2f}"
                detection_info.append(f"{label} ({score:.2%})")
                
                # 获取文本尺寸
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                # 绘制文本背景
                draw.rectangle([(x1, y1 - text_h - 10), (x1 + text_w + 10, y1)], 
                            fill=color)
                
                # 绘制文本
                draw.text((x1 + 5, y1 - text_h - 5), text, 
                        fill=(255, 255, 255), font=font)
                
            except Exception as e:
                logging.error(f"Error drawing box: {e}")
                continue
        
        # 转换回OpenCV格式
        vis_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return vis_frame, num_detections, detection_info
    
    def _get_color(self, label):
        """根据标签获取颜色"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
        ]
        return colors[int(label) % len(colors)]


def initialize_detector(model_choice):
    """初始化检测器"""
    global detector_instance
    try:
        models_config = model_options[model_choice]
        detector_instance = VideoDetector(
            model_dir=models_config,
            threshold=0.3
        )
        return "✓ 模型初始化成功"
    except Exception as e:
        logging.error(f"初始化模型失败: {e}")
        return f"✗ 初始化失败: {str(e)}"


def detect_image(image_path, model_choice, threshold):
    """检测图片"""
    global detector_instance
    
    if image_path is None:
        return None, "请先上传或选择图片"
    
    if detector_instance is None:
        initialize_detector(model_choice)
    
    # 更新阈值
    detector_instance.threshold = threshold
    
    try:
        vis_frame, num_detections, detection_info = detector_instance.detect_image(image_path)
        
        # 转换为RGB格式
        result_image = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        
        status = f"检测到 {num_detections} 个目标"
        if detection_info:
            status += f"\n目标详情: {', '.join(detection_info)}"
        
        return result_image, status
        
    except Exception as e:
        logging.error(f"检测出错: {e}")
        return None, f"检测出错: {str(e)}"


def detect_video(video_path, model_choice, threshold, progress=gr.Progress()):
    """检测视频"""
    global detector_instance
    
    if video_path is None:
        return None, "请先上传视频"
    
    if detector_instance is None:
        initialize_detector(model_choice)
    
    # 更新阈值
    detector_instance.threshold = threshold
    
    try:
        output_path, status = detector_instance.detect_video(video_path, progress)
        return output_path, status
        
    except Exception as e:
        logging.error(f"视频检测出错: {e}")
        return None, f"视频检测出错: {str(e)}"


def change_model(model_choice):
    """切换模型"""
    global detector_instance
    detector_instance = None
    return f"已选择模型: {model_choice}，下次检测时将自动加载"


def clear_outputs():
    """清空输出"""
    return None, None, "", None, ""


def create_gradio_interface():
    """创建Gradio界面"""
    health_check_js = '''
    () => {
        let isConnected = true;
        setInterval(async () => {
            try {
                await fetch('/');
                if (!isConnected) {
                    console.log("重新连接成功，刷新页面...");
                    location.reload();
                }
                isConnected = true;
            } catch (e) {
                if (isConnected) {
                    console.log("连接断开，等待重连...");
                }
                isConnected = false;
            }
        }, 2000);
    }
    '''
    
    with gr.Blocks(title="传送带目标检测", theme=gr.themes.Default(), js=health_check_js) as iface:
        gr.Markdown("""
        # 🚚 传送带目标检测系统
        **功能特点：** 基于PaddleX PP-YOLOE+模型、支持图片和视频检测、实时可视化标注、可调节检测阈值
        """)
        
        with gr.Row():
            # 左侧：示例和模型选择
            with gr.Column(scale=1):
                gr.Markdown("### 🖼️ 示例图片")
                example_gallery = gr.Gallery(
                    value=util.get_current_examples(EXAMPLE_DIR),
                    label="点击选择示例",
                    show_label=False,
                    columns=4,
                    rows=1,
                    height=200,
                    allow_preview=False
                )
                
                gr.Markdown("### ⚙️ 模型配置")
                model_selector = gr.Dropdown(
                    choices=list(model_options.keys()),
                    value=list(model_options.keys())[0] if model_options else None,
                    label="选择检测模型",
                    info="支持PaddleX预训练和自定义模型"
                )
                
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.3,
                    step=0.05,
                    label="检测阈值",
                    info="降低阈值可提高召回率"
                )
                
                with gr.Accordion("📖 使用说明", open=True):
                    gr.Markdown("""
                    **操作步骤：**
                    1. 选择检测模型和阈值
                    2. 上传图片或选择示例
                    3. 点击对应的检测按钮
                    4. 查看检测结果和统计信息
                    
                    **图片格式：** JPG, PNG, JPEG
                    **视频格式：** MP4, AVI, MOV
                    
                    **模型：** PP-YOLOE+ (PaddleX)
                    """)
            
            # 右侧：上传和结果
            with gr.Column(scale=1):
                with gr.Tabs():
                    # 图片检测标签页
                    with gr.Tab("📷 图片检测"):
                        gr.Markdown("### 📤 上传图片")
                        input_image = gr.Image(
                            type="filepath",
                            label="上传图片",
                            height=200,
                            sources=['upload']
                        )
                        
                        with gr.Row():
                            image_detect_btn = gr.Button("🔍 检测图片", variant="primary", size="lg")
                            image_clear_btn = gr.Button("🗑️ 清空", variant="secondary")
                        
                        gr.Markdown("### 📋 检测结果")
                        output_image = gr.Image(label="检测结果", height=500)
                        image_status = gr.Textbox(label="检测信息", lines=3, interactive=False)
                    
                    # 视频检测标签页
                    with gr.Tab("🎬 视频检测",visible=False):
                        gr.Markdown("### 📤 上传视频")
                        input_video = gr.Video(
                            label="上传视频",
                            height=200
                        )
                        
                        with gr.Row():
                            video_detect_btn = gr.Button("🔍 检测视频", variant="primary", size="lg")
                            video_clear_btn = gr.Button("🗑️ 清空", variant="secondary")
                        
                        gr.Markdown("### 📋 检测结果")
                        output_video = gr.Video(label="检测结果视频", height=500)
                        video_status = gr.Textbox(label="处理信息", lines=3, interactive=False)
        
        # 事件绑定
        def select_example(evt: gr.SelectData):
            current_examples = util.get_current_examples(EXAMPLE_DIR)
            if current_examples and evt.index < len(current_examples):
                return current_examples[evt.index]
            return None
        
        example_gallery.select(select_example, None, outputs=[input_image])
        
        image_detect_btn.click(
            fn=detect_image,
            inputs=[input_image, model_selector, threshold_slider],
            outputs=[output_image, image_status]
        )
        
        image_clear_btn.click(
            fn=lambda: (None, None, ""),
            outputs=[input_image, output_image, image_status]
        )
        
        video_detect_btn.click(
            fn=detect_video,
            inputs=[input_video, model_selector, threshold_slider],
            outputs=[output_video, video_status]
        )
        
        video_clear_btn.click(
            fn=lambda: (None, None, ""),
            outputs=[input_video, output_video, video_status]
        )
        
        model_selector.change(
            fn=change_model,
            inputs=[model_selector]
        )
        
        return iface


def main():
    """主函数"""
    monitor_manager = MultiDirectoryMonitor(restart_signal_file_name=RESTART_SIGNAL_FILENAME)
    monitor_manager.add_directory(MODEL_BASE_DIR)
    monitor_manager.add_directory(EXAMPLE_DIR)
    
    if not monitor_manager.start_all():
        logging.error("❌ 启动目录监控失败")
        return
    
    port = 7862
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            if port < 1024 or port > 65535:
                logging.warning(f"端口号 {port} 不在有效范围，使用默认端口 7862")
                port = 7862
        except ValueError:
            logging.warning(f"无效端口号，使用默认端口 7862")
    
    iface = create_gradio_interface()
    
    try:
        iface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
        )
    finally:
        monitor_manager.stop_all(join_threads=True)


if __name__ == "__main__":
    main()