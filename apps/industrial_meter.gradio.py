from utils.app_utils import AppUtils as util
from utils.app_utils import MultiDirectoryMonitor


import os
import sys
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import numpy as np
import math
import cv2
import paddlex as pdx
import gradio as gr
from PIL import Image
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


METER_SHAPE = 512
CIRCLE_CENTER = [256, 256]
CIRCLE_RADIUS = 250
PI = 3.1415926536
LINE_HEIGHT = 120
LINE_WIDTH = 1570
TYPE_THRESHOLD = 40
METER_CONFIG = [{
    'scale_value': 25.0 / 50.0,
    'range': 25.0,
    'unit': "(MPa)"
}, {
    'scale_value': 1.6 / 32.0,
    'range': 1.6,
    'unit': "(MPa)"
}]

from pathlib import Path

# --- 模型目录配置 ---
BASE_DIR = Path(__file__).parent.parent
MODEL_BASE_DIR = BASE_DIR / "model" / "industrail_metric_det" / "model"
RESTART_SIGNAL_FILENAME = ".restart_signal_industrai_meter"
EXAMPLE_DIR = BASE_DIR / "dataset" / "industrail_metric_det"

class MeterReader:
    def __init__(self, detector_dir, segmenter_dir):
        if not osp.exists(detector_dir):
            raise Exception("Model path {} does not exist".format(
                detector_dir))
        if not osp.exists(segmenter_dir):
            raise Exception("Model path {} does not exist".format(
                segmenter_dir))
        
        self.detector = pdx.create_model(model_name='PP-YOLOE_plus-S', model_dir=detector_dir)
        self.segmenter = pdx.create_model(model_name='SegFormer-B1', model_dir=segmenter_dir)

    def predict(self,
                im_file,
                use_erode=True,
                erode_kernel=4,
                score_threshold=0.5,
                seg_batch_size=2):
        if isinstance(im_file, str):
            im = cv2.imread(im_file).astype('float32')
        else:
            im = im_file.copy().astype('float32')
        
        # Get detection results - PaddleX 3.2 预测接口
        det_results = self.detector.predict(im_file if isinstance(im_file, str) else im)
        
        # 适配新的结果格式
        det_result = list(det_results)[0]
        if det_result.get('boxes', None):
            # 新版本可能返回不同的数据结构
            filtered_results = list()
            boxes = det_result.get('boxes', None)
            
            for box in boxes:
                score = box.get('score', None)
                if score and score > score_threshold:
                    # 提取坐标
                    xmin = box['coordinate'][0]
                    ymin = box['coordinate'][1]
                    xmax = box['coordinate'][2]
                    ymax = box['coordinate'][3]
                    result_dict = {
                        'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                        'score': score,
                        'category_id': getattr(box, 'cls_id', 0),
                        'category': getattr(box, 'label', 'meter')
                    }
                    filtered_results.append(result_dict)
        else:
            # 兼容旧格式
            filtered_results = list()
            for res in det_result:
                if res['score'] > score_threshold:
                    filtered_results.append(res)

        resized_meters = list()
        for res in filtered_results:
            # Crop the bbox area
            xmin, ymin, w, h = res['bbox']
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(im.shape[1], int(xmin + w - 1))
            ymax = min(im.shape[0], int(ymin + h - 1))
            sub_image = im[ymin:(ymax + 1), xmin:(xmax + 1), :]

            # Resize the image with shape (METER_SHAPE, METER_SHAPE)
            meter_shape = sub_image.shape
            scale_x = float(METER_SHAPE) / float(meter_shape[1])
            scale_y = float(METER_SHAPE) / float(meter_shape[0])
            meter_meter = cv2.resize(
                sub_image,
                None,
                None,
                fx=scale_x,
                fy=scale_y,
                interpolation=cv2.INTER_LINEAR)
            meter_meter = meter_meter.astype('float32')
            resized_meters.append(meter_meter)

        meter_num = len(resized_meters)
        seg_results = list()
        
        # 分割预测
        for i in range(0, meter_num, seg_batch_size):
            im_size = min(meter_num, i + seg_batch_size)
            for j in range(i, im_size):
                # 单独预测每个仪表图像
                seg_result_generator = self.segmenter.predict(resized_meters[j])
                seg_result = list(seg_result_generator)[0]
                # 处理分割结果
                if seg_result.get('label_map', None) is not None:
                    label_map = seg_result.get('label_map')
                elif seg_result.get('pred', None) is not None and np.any(seg_result.get('pred', None)):
                    label_map = seg_result['pred']
                else:
                    label_map = seg_result if isinstance(seg_result, np.ndarray) else seg_result[0]
                label_map_uint8 = label_map.astype(np.uint8)
                if use_erode:
                    kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
                    label_map = cv2.erode(label_map_uint8, kernel)
                
                seg_results.append({'label_map': label_map})

        results = list()
        for i, seg_result in enumerate(seg_results):
            result = self.read_process(seg_result['label_map'])
            results.append(result)

        meter_values = list()
        for i, result in enumerate(results):
            if result['scale_num'] > TYPE_THRESHOLD:
                value = result['scales'] * METER_CONFIG[0]['scale_value']
            else:
                value = result['scales'] * METER_CONFIG[1]['scale_value']
            meter_values.append(value)

        # 生成可视化结果图像
        result_image = self.visualize_results(im_file, filtered_results, meter_values)
        
        return result_image, meter_values, filtered_results

    def visualize_results(self, im_file, filtered_results, meter_values):
        """
        可视化检测和读数结果，返回PIL图像
        """
        if isinstance(im_file, str):
            im = cv2.imread(im_file)
        else:
            im = im_file.copy()
        
        # 在图像上绘制检测框和读数值
        for i, (res, value) in enumerate(zip(filtered_results, meter_values)):
            xmin, ymin, w, h = res['bbox']
            xmin, ymin = int(xmin), int(ymin)
            xmax, ymax = int(xmin + w), int(ymin + h)
            
            # 绘制检测框
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # 添加读数文本
            if value != -1:
                text = f"Meter {i+1}: {value:.2f} MPa"
            else:
                text = f"Meter {i+1}: Unable to read"
            cv2.putText(im, text, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
        
        # 转换为RGB格式并返回PIL图像
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return Image.fromarray(im_rgb)

    def read_process(self, label_maps):
        # Convert the circular meter into rectangular meter
        line_images = self.creat_line_image(label_maps)
        # Convert the 2d meter into 1d meter
        scale_data, pointer_data = self.convert_1d_data(line_images)
        # Fliter scale data whose value is lower than the mean value
        self.scale_mean_filtration(scale_data)
        # Get scale_num, scales and ratio of meters
        result = self.get_meter_reader(scale_data, pointer_data)
        return result

    def creat_line_image(self, meter_image):
        if len(meter_image.shape) == 3:
            meter_image = meter_image[0]  # 取第一个批次
        line_image = np.zeros((LINE_HEIGHT, LINE_WIDTH), dtype=np.uint8)
        for row in range(LINE_HEIGHT):
            for col in range(LINE_WIDTH):
                theta = PI * 2 / LINE_WIDTH * (col + 1)
                rho = CIRCLE_RADIUS - row - 1
                x = int(CIRCLE_CENTER[0] + rho * math.cos(theta) + 0.5)
                y = int(CIRCLE_CENTER[1] - rho * math.sin(theta) + 0.5)
                line_image[row, col] = meter_image[x, y]
        return line_image

    def convert_1d_data(self, meter_image):
        scale_data = np.zeros((LINE_WIDTH), dtype=np.uint8)
        pointer_data = np.zeros((LINE_WIDTH), dtype=np.uint8)
        for col in range(LINE_WIDTH):
            for row in range(LINE_HEIGHT):
                if meter_image[row, col] == 1:
                    pointer_data[col] += 1
                elif meter_image[row, col] == 2:
                    scale_data[col] += 1
        return scale_data, pointer_data

    def scale_mean_filtration(self, scale_data):
        mean_data = np.mean(scale_data)
        for col in range(LINE_WIDTH):
            if scale_data[col] < mean_data:
                scale_data[col] = 0

    def get_meter_reader(self, scale_data, pointer_data):
        scale_flag = False
        pointer_flag = False
        one_scale_start = 0
        one_scale_end = 0
        one_pointer_start = 0
        one_pointer_end = 0
        scale_location = list()
        pointer_location = 0
        for i in range(LINE_WIDTH - 1):
            if scale_data[i] > 0 and scale_data[i + 1] > 0:
                if scale_flag == False:
                    one_scale_start = i
                    scale_flag = True
            if scale_flag:
                if scale_data[i] == 0 and scale_data[i + 1] == 0:
                    one_scale_end = i - 1
                    one_scale_location = (one_scale_start + one_scale_end) / 2
                    scale_location.append(one_scale_location)
                    one_scale_start = 0
                    one_scale_end = 0
                    scale_flag = False
            if pointer_data[i] > 0 and pointer_data[i + 1] > 0:
                if pointer_flag == False:
                    one_pointer_start = i
                    pointer_flag = True
            if pointer_flag:
                if pointer_data[i] == 0 and pointer_data[i + 1] == 0:
                    one_pointer_end = i - 1
                    pointer_location = (
                        one_pointer_start + one_pointer_end) / 2
                    one_pointer_start = 0
                    one_pointer_end = 0
                    pointer_flag = False

        scale_num = len(scale_location)
        scales = -1
        ratio = -1
        if scale_num > 0:
            for i in range(scale_num - 1):
                if scale_location[
                        i] <= pointer_location and pointer_location < scale_location[
                            i + 1]:
                    scales = i + (pointer_location - scale_location[i]) / (
                        scale_location[i + 1] - scale_location[i] + 1e-05) + 1
            ratio = (pointer_location - scale_location[0]) / (
                scale_location[scale_num - 1] - scale_location[0] + 1e-05)
        result = {'scale_num': scale_num, 'scales': scales, 'ratio': ratio}
        return result


# 全局变量存储模型实例和选项
meter_reader = None
model_options = util.generate_paddlex_model_options(MODEL_BASE_DIR)

def initialize_model(detector_dir, segmenter_dir):
    """初始化模型"""
    global meter_reader
    try:
        meter_reader = MeterReader(detector_dir, segmenter_dir)
        return "✅ 模型加载成功！"
    except Exception as e:
        return f"❌ 模型加载失败: {str(e)}"

def predict_meter_reading(image, detector_model, segmenter_model, use_erode, erode_kernel, score_threshold):
    """预测仪表读数的主函数"""
    global meter_reader
    
    detector_dir = model_options[detector_model]
    segmenter_dir = model_options[segmenter_model]
    
    if image is None:
        return None, "❌ 请上传图片！", "", "⚠️ 请先上传图片"
    
    # 检查是否需要重新加载模型
    if meter_reader is None:
        status = initialize_model(detector_dir, segmenter_dir)
        if "失败" in status:
            return None, status, "", "❌ 模型加载失败"
    
    try:
        # 将PIL图像转换为numpy数组
        image_array = np.array(image)
        
        # 执行预测
        result_image, meter_values, filtered_results = meter_reader.predict(
            image_array,
            use_erode=use_erode,
            erode_kernel=erode_kernel,
            score_threshold=score_threshold
        )
        
        # 生成结果文本
        if len(meter_values) == 0:
            result_text = "❌ 未检测到任何仪表"
            summary_text = "未找到仪表"
            status_text = "⚠️ 未检测到仪表"
        else:
            result_lines = []
            valid_readings = []
            
            for i, value in enumerate(meter_values):
                if value != -1:
                    result_lines.append(f"仪表 {i+1}: {value:.3f} MPa")
                    valid_readings.append(value)
                else:
                    result_lines.append(f"仪表 {i+1}: 无法读取")
            
            result_text = "📊 检测结果:\n" + "\n".join(result_lines)
            
            if valid_readings:
                summary_text = f"共检测到 {len(meter_values)} 个仪表，成功读取 {len(valid_readings)} 个"
                status_text = f"✅ 成功读取 {len(valid_readings)}/{len(meter_values)} 个仪表"
            else:
                summary_text = f"共检测到 {len(meter_values)} 个仪表，但无法读取数值"
                status_text = "⚠️ 检测到仪表但无法读取"
        
        return result_image, result_text, summary_text, status_text
        
    except Exception as e:
        error_msg = f"❌ 预测过程中出现错误: {str(e)}"
        return None, error_msg, "", "❌ 预测失败"


# 创建Gradio界面
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
    with gr.Blocks(title="智能仪表读数系统", js=health_check_js) as iface:        
        gr.Markdown("""
        # 🎯 智能仪表读数系统
        **功能特点：** 基于深度学习的压力表自动读数工具
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                initial_examples = util.get_current_examples(EXAMPLE_DIR)
                example_gallery = gr.Gallery(
                    value=initial_examples,
                    label="点击选择示例",
                    show_label=False,
                    elem_id="example_gallery",
                    columns=4,
                    rows=1,
                    height=200,
                    allow_preview=False
                )
                # 模型配置区域
                gr.Markdown("### 🔧 模型配置")
                with gr.Group():
                    detector_dropdown = gr.Dropdown(
                        choices=list(model_options.keys()),
                        value=list(model_options.keys())[1] if model_options else None,
                        label="检测模型",
                        info="选择用于检测仪表的模型路径"
                    )
                    
                    segmenter_dropdown = gr.Dropdown(
                        choices=list(model_options.keys()),
                        value=list(model_options.keys())[0] if model_options else None,
                        label="分割模型",
                        info="选择用于分割指针和刻度的模型路径"
                    )
                
                # 参数配置区域
                gr.Markdown("### ⚙️ 检测参数")
                with gr.Group():
                    score_threshold = gr.Slider(
                        minimum=0.1, 
                        maximum=0.9, 
                        value=0.5, 
                        step=0.1, 
                        label="检测置信度阈值",
                        info="低于此值的检测结果将被过滤"
                    )
                    
                    with gr.Row():
                        use_erode = gr.Checkbox(
                            label="使用形态学腐蚀", 
                            value=True,
                            info="减少分割噪声"
                        )
                        erode_kernel = gr.Slider(
                            minimum=1, 
                            maximum=10, 
                            value=4, 
                            step=1, 
                            label="腐蚀核大小",
                            info="值越大腐蚀效果越强"
                        )
                
                # 预测按钮和状态显示
                predict_btn = gr.Button("🔍 开始读数", variant="primary", size="lg")
                        
                 # 使用说明
                with gr.Accordion("📋 使用说明", open=False):
                    gr.Markdown("""
                    **📋 使用步骤**
                    
                    1. **模型选择**: 从下拉列表中选择合适的检测器和分割器模型
                    2. **上传图片**: 上传包含压力表的图片（支持JPG, PNG, BMP格式）
                    3. **调整参数**: 根据需要调整检测置信度和形态学处理参数
                    4. **开始读数**: 点击"开始读数"按钮进行预测
                    5. **查看结果**: 在右侧查看带有标注的结果图片和具体读数值
                    
                    **⚙️ 参数说明**
                    
                    - **检测置信度阈值**: 控制检测的敏感度，值越高检测越严格
                    - **使用形态学腐蚀**: 对分割结果进行后处理，可以减少噪声
                    - **腐蚀核大小**: 腐蚀操作的强度，适当调整可以改善读数准确性
                    """)
            
            with gr.Column(scale=1):
                # 图片上传区域
                gr.Markdown("### 📷 上传图片")
                input_image = gr.Image(
                    label="选择包含仪表的图片", 
                    type="pil",
                    height=300,
                    sources=['upload']
                )
                
                gr.Markdown("### 📊 检测结果")
                
                # 结果图片显示
                output_image = gr.Image(
                    label="检测结果图片", 
                    height=400,
                    show_label=True
                )
                
                # 结果摘要
                summary_text = gr.Textbox(
                    label="检测摘要", 
                    interactive=False,
                    show_label=True
                )
                
                # 详细结果
                result_text = gr.Textbox(
                    label="详细结果", 
                    lines=8, 
                    interactive=False,
                    show_label=True
                )
        
        def select_example(evt: gr.SelectData):
            current_examples = util.get_current_examples(EXAMPLE_DIR)
            if current_examples and evt.index < len(current_examples):
                selected_path = current_examples[evt.index]
                return selected_path
            return None

        example_gallery.select(select_example, None, outputs=[input_image])
        
        # 事件绑定
        predict_btn.click(
            predict_meter_reading,
            inputs=[
                input_image, 
                detector_dropdown, 
                segmenter_dropdown,
                use_erode, 
                erode_kernel, 
                score_threshold
            ],
            outputs=[output_image, result_text, summary_text]
        )
    
    return iface


def main():
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
                logging.warning(f"警告：端口号 {port} 不在有效范围内(1024-65535)，将使用默认端口{port}")
                port = port
        except ValueError:
            logging.warning(f"警告：无效的端口号参数 '{sys.argv[1]}'，将使用默认端口{port}")
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


if __name__ == '__main__':
    main()