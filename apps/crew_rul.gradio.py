# -*- coding: utf-8 -*-
import gradio as gr
import pandas as pd
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras import losses, metrics
from scipy import signal, stats
import glob
import pickle
from utils.app_utils import MultiDirectoryMonitor
from utils.app_utils import AppUtils as util
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from io import BytesIO
from PIL import Image
import matplotlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.app_utils import AppUtils as util
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path
matplotlib.use('Agg')
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量
selected_data_path = None
selected_model_version = None
max_start_index = 0
file_count = 0
plt = util.auto_config_chinese_font()

# 模型缓存
model_cache = {}
scaler_cache = {}




class Config:

    WINDOW_SIZE = 25
    FS = 25600
    RUL_CAP = 150.0
    
    def __init__(self, model_version=4400):
        self.MODEL_VERSION = model_version
        self.BASE_DIR = Path(__file__).parent.parent
        self.MODEL_BASE_DIR = self.BASE_DIR / "model" / "crew_rul" / "model"
        self.RESTART_SIGNAL_FILENAME = ".restart_signal_crew_rul"
        self.EXAMPLE_DIR = self.BASE_DIR / "model" / "crew_rul" / "dataset"

    @property
    def MODEL_PATH(self):
        logging.info(self.MODEL_BASE_DIR)
        return self.MODEL_BASE_DIR / f"rul_model_v{self.MODEL_VERSION}.h5"

    @property
    def SCALER_PATH(self):
        return self.MODEL_BASE_DIR / f"scalers_v{self.MODEL_VERSION}.pkl"

class TemporalAttention(Layer):
    """时序注意力机制"""
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = inputs * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

model_options = util.generate_paddlex_model_options(Config().MODEL_BASE_DIR)
config=None

def load_model_cached(model_version):
    """加载模型并缓存"""
    if model_version not in model_cache:
        config = Config(model_version=model_version)
        if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.SCALER_PATH):
            logging.error(f"找不到模型v{model_version}或scaler文件")
            raise FileNotFoundError(f"找不到模型v{model_version}或scaler文件")
        
        logging.info(f"🔄 正在加载模型 v{model_version}...")
        
        # 使用 compile=False 避免某些序列化问题
        model = load_model(config.MODEL_PATH, custom_objects={
            'TemporalAttention': TemporalAttention(),
            'mse': losses.MeanSquaredError(),
            'mae': metrics.MeanAbsoluteError()
        }, compile=False)
        
        # 使用更安全的 pickle 加载方式
        try:
            with open(config.SCALER_PATH, 'rb') as f:
                saved_data = pickle.load(f)
                scaler_X = saved_data['scaler_X']
        except (AttributeError, ModuleNotFoundError) as e:
            # 如果直接加载失败，尝试使用兼容模式
            logging.warning(f"⚠️ 标准加载失败，尝试兼容模式: {e}")
            import sys
            import types
            
            # 创建临时模块以支持 pickle 加载
            if '__main__' not in sys.modules or not hasattr(sys.modules['__main__'], 'Config'):
                main_module = sys.modules.get('__main__')
                if main_module is None:
                    main_module = types.ModuleType('__main__')
                    sys.modules['__main__'] = main_module
                
                main_module.Config = Config
                main_module.TemporalAttention = TemporalAttention
            
            with open(config.SCALER_PATH, 'rb') as f:
                saved_data = pickle.load(f)
                scaler_X = saved_data['scaler_X']
        
        model_cache[model_version] = model
        scaler_cache[model_version] = scaler_X
        logging.info(f"✅ 模型 v{model_version} 加载完成并已缓存")
    
    return model_cache[model_version], scaler_cache[model_version]

def create_features_for_signal(sig, fs=25600):
    """为单通道信号提取时域和频域特征"""
    rms = np.sqrt(np.mean(sig**2))
    kurtosis = stats.kurtosis(sig)
    skewness = stats.skew(sig)
    peak_to_peak = np.max(sig) - np.min(sig)

    if rms > 1e-6:
        crest_factor = np.max(np.abs(sig)) / rms
    else:
        crest_factor = 0

    mean_sqrt = np.mean(np.sqrt(np.abs(sig)))
    if mean_sqrt > 1e-6:
        margin_factor = np.max(np.abs(sig)) / (mean_sqrt**2)
    else:
        margin_factor = 0

    mean_abs = np.mean(np.abs(sig))
    if mean_abs > 1e-6:
        impulse_factor = np.max(np.abs(sig)) / mean_abs
    else:
        impulse_factor = 0

    n = len(sig)
    freqs = np.fft.fftfreq(n, 1/fs)
    fft_vals = np.abs(np.fft.fft(sig))
    positive_freq_mask = freqs > 0
    freqs = freqs[positive_freq_mask]
    fft_vals = fft_vals[positive_freq_mask]

    total_power = np.sum(fft_vals)
    if total_power > 1e-6:
        freq_mean = np.average(freqs, weights=fft_vals)
        freq_std = np.sqrt(np.average((freqs - freq_mean)**2, weights=fft_vals))
    else:
        freq_mean = 0
        freq_std = 0

    return [rms, kurtosis, skewness, peak_to_peak, crest_factor, 
            margin_factor, impulse_factor, freq_mean, freq_std]

def load_and_process_csv_features(csv_path):
    """加载CSV，处理并提取特征向量"""
    try:
        data = pd.read_csv(csv_path)
        h = data.iloc[:, 0].values
        v = data.iloc[:, 1].values

        b, a = signal.butter(4, [20, 10000], btype='band', fs=25600)
        h = signal.filtfilt(b, a, h)
        v = signal.filtfilt(b, a, v)
        
        features_h = create_features_for_signal(h)
        features_v = create_features_for_signal(v)
        
        return np.array(features_h + features_v, dtype=np.float32)
    except Exception as e:
        print(f"警告: 处理文件 {csv_path} 时出错: {e}")
        return None

def create_visualization(predicted_rul, config):
    # 优化后的字体配置
    FONT_CONFIG = {
        'value_xlarge': 42,    # 主要数值
        'value_large': 32,     # 次要数值
        'title_large': 22,     # 主标题
        'title_medium': 18,    # 次标题
        'label_medium': 16,    # 标签
        'label_small': 14,     # 小标签
        'ticks': 11           # 刻度
    }

    # 优化图表尺寸和布局 - 调整为16:9比例
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.30, 
                         left=0.08, right=0.96, top=0.93, bottom=0.08)
    
    health_threshold = config.RUL_CAP * 0.5
    percentage = min(100, (predicted_rul / config.RUL_CAP) * 100)

    # 确定健康状态和颜色
    if predicted_rul > health_threshold:
        color = '#2ecc71'
        status = '健康'
        icon = '✓'
    elif predicted_rul > health_threshold * 0.5:
        color = '#f39c12'
        status = '轻微退化'
        icon = '⚠'
    else:
        color = '#e74c3c'
        status = '严重退化'
        icon = '✗'

    # 1. 半圆仪表盘 (左上，占2列)
    ax1 = fig.add_subplot(gs[0, :2], projection='polar')
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    # 背景半圆
    ax1.plot(theta, r, color='#ecf0f1', linewidth=22, solid_capstyle='round', zorder=1)
    
    # 填充半圆
    theta_fill = np.linspace(0, np.pi * (percentage / 100), 100)
    r_fill = np.ones_like(theta_fill)
    ax1.plot(theta_fill, r_fill, color=color, linewidth=22, solid_capstyle='round', zorder=2)
    
    ax1.set_ylim(0, 1.25)
    ax1.set_xlim(-0.2, np.pi + 0.2)
    ax1.axis('off')
    
    # 中心文字
    ax1.text(np.pi/2, 0.42, f'{predicted_rul:.1f}', 
            ha='center', va='center', 
            fontsize=FONT_CONFIG['value_xlarge'],
            fontweight='bold', color=color, family='monospace')
    ax1.text(np.pi/2, 0.15, '剩余寿命 (分钟)', 
            ha='center', va='center', 
            fontsize=FONT_CONFIG['label_medium'],
            color='#7f8c8d')
    ax1.text(np.pi/2, -0.15, f'{icon} {status}', 
            ha='center', va='center', 
            fontsize=FONT_CONFIG['title_large'],
            fontweight='bold', color=color)

    # 2. 健康度百分比卡片 (右上)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    # 绘制渐变背景矩形
    rect = FancyBboxPatch((0.08, 0.2), 0.84, 0.65, 
                         boxstyle="round,pad=0.05", 
                         facecolor=color, alpha=0.12, 
                         edgecolor=color, linewidth=3)
    ax2.add_patch(rect)
    
    ax2.text(0.5, 0.68, f'{percentage:.1f}%', 
            ha='center', va='center', fontsize=FONT_CONFIG['value_large'], 
            fontweight='bold', color=color, transform=ax2.transAxes)
    ax2.text(0.5, 0.40, '健康度', 
            ha='center', va='center', fontsize=FONT_CONFIG['label_medium'], 
            color='#7f8c8d', transform=ax2.transAxes)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    # 3. 进度条式对比 (左下，占2列)
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.axis('off')
    ax3.set_xlim(-5, config.RUL_CAP * 1.08)
    ax3.set_ylim(-0.4, 2.2)
    
    # 绘制健康阈值线
    bar_height = 0.65
    bar_y = 0.5
    
    # 背景条
    rect_bg = Rectangle((0, bar_y), config.RUL_CAP, bar_height, 
                       facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=2.5)
    ax3.add_patch(rect_bg)
    
    # 当前RUL条
    rect_rul = Rectangle((0, bar_y), predicted_rul, bar_height, 
                       facecolor=color, alpha=0.85, edgecolor=color, linewidth=2.5)
    ax3.add_patch(rect_rul)
    
    # 健康阈值标记线
    ax3.plot([health_threshold, health_threshold], [0.25, 1.45], 
            color='#3498db', linewidth=2.5, linestyle='--', alpha=0.75)
    ax3.text(health_threshold, 1.60, f'健康阈值\n{health_threshold:.1f}分钟', 
            ha='center', va='bottom', fontsize=FONT_CONFIG['label_small'], 
            color='#3498db', fontweight='bold')
    
    # 当前RUL标记
    label_x = max(predicted_rul + 3, 8)
    ax3.text(label_x, bar_y + bar_height/2, f'{predicted_rul:.1f}', 
            ha='left', va='center', fontsize=FONT_CONFIG['label_medium'], 
            fontweight='bold', color=color)
    
    # 标题
    ax3.text(-3, 1.95, '剩余寿命对比图', 
            ha='left', va='center', fontsize=FONT_CONFIG['title_large'], 
            fontweight='bold', color='#2c3e50')
    
    # X轴刻度
    tick_y = -0.08
    ax3.text(0, tick_y, '0', ha='center', va='top', 
            fontsize=FONT_CONFIG['ticks'], color='#7f8c8d')
    ax3.text(config.RUL_CAP/2, tick_y, f'{config.RUL_CAP/2:.0f}', 
            ha='center', va='top', fontsize=FONT_CONFIG['ticks'], color='#7f8c8d')
    ax3.text(config.RUL_CAP, tick_y, f'{config.RUL_CAP:.0f}', 
            ha='center', va='top', fontsize=FONT_CONFIG['ticks'], color='#7f8c8d')

    # 4. 状态指示卡片 (右下)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    # 状态卡片
    card_colors = {
        '健康': ('#d5f4e6', '#27ae60'),
        '轻微退化': ('#fef5e7', '#e67e22'),
        '严重退化': ('#fadbd8', '#c0392b')
    }
    bg_color, border_color = card_colors[status]
    
    rect_card = FancyBboxPatch((0.08, 0.15), 0.84, 0.70, 
                             boxstyle="round,pad=0.05", 
                             facecolor=bg_color, 
                             edgecolor=border_color, linewidth=3)
    ax4.add_patch(rect_card)
    
    # 状态图标和文字
    ax4.text(0.5, 0.58, status, ha='center', va='center', 
            fontsize=FONT_CONFIG['title_large'], fontweight='bold', 
            color=border_color, transform=ax4.transAxes)
    ax4.text(0.5, 0.35, '当前状态', ha='center', va='center', 
            fontsize=FONT_CONFIG['label_medium'], color='#7f8c8d', 
            transform=ax4.transAxes)

    # 转换为图像
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()

    return img

def get_file_count(data_path):
    """获取数据文件夹中CSV文件的数量"""
    if data_path and os.path.exists(data_path):
        files = glob.glob(os.path.join(data_path, '*.csv'))
        return len(files)
    return 0

def get_available_models(model_dir):
    """扫描模型目录，获取可用的模型版本"""
    models = []
    if os.path.exists(model_dir):
        model_files = glob.glob(os.path.join(model_dir, 'rul_model_v*.h5'))
        print(model_files)
        for model_file in sorted(model_files):
            basename = os.path.basename(model_file)
            version_str = basename.replace('rul_model_v', '').replace('.h5', '')
            display_name = basename.replace('.h5', '')
            try:
                scaler_file = os.path.join(model_dir, f'scalers_v{version_str}.pkl')
                if os.path.exists(scaler_file):
                    models.append((display_name, version_str))
            except ValueError:
                continue
    
    if not models:
        models = [("模型 v4400 (默认)", 4400)]
    
    return models

def predict_rul(data_path, start_index, model_version):
    """对给定的数据文件夹进行RUL预测"""
    config = Config(model_version=model_version)

    try:
        # 使用缓存的模型
        model, scaler_X = load_model_cached(model_version)
        
        key_func = lambda f: int(os.path.splitext(os.path.basename(f))[0])
        files = sorted(glob.glob(os.path.join(data_path, '*.csv')), key=key_func)
        
        if len(files) < config.WINDOW_SIZE:
            return None, f"❌ 错误: 数据点不足。需要至少 {config.WINDOW_SIZE} 个CSV文件，但只找到 {len(files)} 个。"
        
        if start_index + config.WINDOW_SIZE > len(files):
            return None, f"❌ 错误: 起始索引 {start_index} 超出范围。最大起始索引为 {len(files) - config.WINDOW_SIZE}。"
        
        latest_files = files[start_index:start_index + config.WINDOW_SIZE]
        features_list = []
        
        for f in latest_files:
            fv = load_and_process_csv_features(f)
            if fv is not None:
                features_list.append(fv)
        
        if len(features_list) < config.WINDOW_SIZE:
            return None, f"❌ 错误: 有效特征不足 {config.WINDOW_SIZE} 个"
        
        features = np.array(features_list)
        features_scaled = scaler_X.transform(features)
        input_data = np.expand_dims(features_scaled, axis=0)
        
        prediction_array = model.predict(input_data, verbose=0)
        log_rul_pred = prediction_array[0][0]
        predicted_rul = np.maximum(0, np.expm1(log_rul_pred))
        
        img = create_visualization(predicted_rul, config)
        
        health_threshold = config.RUL_CAP * 0.5
        if predicted_rul > health_threshold:
            status = "✅ 健康"
            recommendation = "轴承状态良好，继续正常运行。"
            emoji = "🟢"
        elif predicted_rul > health_threshold * 0.5:
            status = "⚠️ 轻微退化"
            recommendation = "建议安排定期检查，密切监控轴承状态。"
            emoji = "🟡"
        else:
            status = "🚨 严重退化"
            recommendation = "强烈建议尽快安排维护或更换轴承！"
            emoji = "🔴"
        
        result_text = f"""
        {emoji} 预测结果
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        🔢 剩余使用寿命 (RUL): {predicted_rul:.2f} 分钟
        📈 健康阈值: {health_threshold:.2f} 分钟
        📊 健康度: {min(100, (predicted_rul/config.RUL_CAP)*100):.1f}%

        🏥 健康状态: {status}

        💡 维护建议:
        {recommendation}

        📁 数据来源: {os.path.basename(data_path)}
        🤖 使用模型: v{model_version} {'(已缓存)' if model_version in model_cache else ''}
        📝 预测区间: 样本 {start_index} 到 {start_index + config.WINDOW_SIZE - 1}
        📊 使用数据点: {config.WINDOW_SIZE} 个样本
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        
        return img, result_text.strip()
        
    except Exception as e:
        return None, f"❌ 预测过程中出错: {str(e)}"

def select_data_folder(folder_path):
    """选择数据文件夹并更新滑块范围"""
    global selected_data_path
    selected_data_path = folder_path
    
    file_count = get_file_count(folder_path)
    config = Config()
    max_start_index = max(0, file_count - config.WINDOW_SIZE)
    
    return (
        f"✅ 已选择数据文件夹: {os.path.basename(folder_path)}\n📊 总文件数: {file_count}",
        gr.update(maximum=max_start_index, value=min(1500, max_start_index), interactive=True)
    )

def select_model(model_version):
    """选择模型版本"""
    global selected_model_version
    selected_model_version = model_version
    
    # 预加载模型到缓存
    try:
        load_model_cached(model_version)
        return f"✅ 已选择并加载模型: v{model_version}"
    except Exception as e:
        return f"⚠️ 模型加载失败: {str(e)}"

def run_prediction(start_index, model_version):
    """运行预测"""
    global selected_data_path
    if selected_data_path is None:
        return None, "⚠️ 请先选择数据文件夹！"
    
    return predict_rul(selected_data_path, start_index, model_version)

def create_interface():
    config = Config()
    data_base_dir = config.EXAMPLE_DIR
    model_dir = config.MODEL_BASE_DIR
    
    data_folders = []
    if os.path.exists(data_base_dir):
        for item in os.listdir(data_base_dir):
            item_path = os.path.join(data_base_dir, item)
            if os.path.isdir(item_path):
                data_folders.append((item, item_path))

    if not data_folders:
        data_folders = [("示例数据", data_base_dir)]
    
    available_models = get_available_models(model_dir)

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
    
    with gr.Blocks(title="🔧 轴承剩余寿命预测系统",  js=health_check_js) as iface:
        gr.Markdown("""
        # 🔧 轴承剩余寿命预测系统
        ### 基于深度学习的智能预测性维护解决方案
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📂 数据源选择")
                
                data_dropdown = gr.Dropdown(
                    choices=data_folders,
                    label="选择数据文件夹",
                    value=data_folders[0][1] if data_folders else "",
                    interactive=True
                )
                
                selection_status = gr.Textbox(
                    label="当前选择",
                    value="",
                    interactive=False,
                    lines=2
                )
                
                gr.Markdown("### 🤖 模型选择")
                
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    label="选择预测模型",
                    value=available_models[0][1] if available_models else 4400,
                    interactive=True
                )
                
                model_status = gr.Textbox(
                    label="模型状态",
                    value=f"✅ 已选择模型: v{available_models[0][1]}" if available_models else "",
                    interactive=False,
                    lines=1
                )
                
                gr.Markdown("### 🎚️ 预测区间选择")
                
                start_index_slider = gr.Slider(
                    minimum=0,
                    maximum=1500,
                    value=1500,
                    step=1,
                    label="起始索引（选择预测的起始位置）",
                    info="滑块范围会根据文件数量自动调整",
                    interactive=True
                )
                
                gr.Markdown("---")
                
                predict_btn = gr.Button(
                    "🚀 开始预测",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                ---
                ### ℹ️ 使用说明
                1. 从下拉菜单选择数据文件夹
                2. 选择要使用的模型版本（首次加载会缓存）
                3. 使用滑块选择预测的起始位置
                4. 点击"开始预测"按钮
                5. 查看预测结果和维护建议
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 预测结果可视化")
                
                result_plot = gr.Image(
                    label="健康状态仪表盘",
                    type="pil",
                    height=500
                )
                
                gr.Markdown("### 📋 详细报告")
                
                result_text = gr.Textbox(
                    label="分析结果",
                    lines=15,
                    max_lines=20,
                    interactive=False
                )
        
        # 绑定事件
        data_dropdown.change(
            fn=select_data_folder,
            inputs=[data_dropdown],
            outputs=[selection_status, start_index_slider]
        )
        
        model_dropdown.change(
            fn=select_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
        predict_btn.click(
            fn=run_prediction,
            inputs=[start_index_slider, model_dropdown],
            outputs=[result_plot, result_text]
        )
        
        # 页面加载时自动选择第一个选项并预加载模型
        def init_interface():
            global selected_data_path, selected_model_version
            
            data_path = data_folders[0][1] if data_folders else ""
            model_version = available_models[0][1] if available_models else 4400
            
            # 更新全局变量
            selected_data_path = data_path
            selected_model_version = model_version
            
            # 预加载模型
            try:
                load_model_cached(model_version)
                model_msg = f"✅ 已选择并加载模型: v{model_version}"
            except Exception as e:
                model_msg = f"⚠️ 模型加载失败: {str(e)}"
            
            file_count = get_file_count(data_path)
            config = Config()
            max_start_index = max(0, file_count - config.WINDOW_SIZE)
            
            data_msg = f"✅ 已选择数据文件夹: {os.path.basename(data_path)}\n📊 总文件数: {file_count}"
            
            return (
                data_msg,
                gr.update(maximum=max_start_index, value=min(1500, max_start_index), interactive=True),
                model_msg
            )
        
        iface.load(
            fn=init_interface,
            outputs=[selection_status, start_index_slider, model_status]
        )

    return iface

def main():
    config = Config()    
    monitor_manager = MultiDirectoryMonitor(restart_signal_file_name=config.RESTART_SIGNAL_FILENAME)
    monitor_manager.add_directory(config.MODEL_BASE_DIR)
    monitor_manager.add_directory(config.EXAMPLE_DIR)
    if not monitor_manager.start_all():
        print("❌ 启动目录监控失败")
        return
    port = 7863
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            if port < 1024 or port > 65535:
                logging.warning(f"警告：端口号 {port} 不在有效范围内(1024-65535)，将使用默认端口7863")
                port = 7863
        except ValueError:
            logging.warning(f"警告：无效的端口号参数 '{sys.argv[1]}'，将使用默认端口7863")

    iface = create_interface()
    try:
        iface.launch(server_name="0.0.0.0", server_port=port, share=False)
    finally:
        monitor_manager.stop_all(join_threads=True)

if __name__ == "__main__":
    main()
