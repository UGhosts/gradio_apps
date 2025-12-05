import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import gradio as gr
import sys
import logging
from pathlib import Path
import io
from PIL import Image
import os
import tensorflow as tf
from tensorflow import keras
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.app_utils import AppUtils as util
from utils.app_utils import MultiDirectoryMonitor

warnings.filterwarnings('ignore')
plt = util.auto_config_chinese_font()

# ============================================================
# 配置路径
# ============================================================
BASE_DIR = Path(__file__).parent.parent / "model" / "driver_belt_rul"
MODEL_BASE_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
EXAMPLE_DIR = BASE_DIR / "examples"
RESTART_SIGNAL_FILENAME = ".restart_signal_driver_belt_rul"

# 确保目录存在
for dir_path in [BASE_DIR, MODEL_BASE_DIR, DATA_DIR, EXAMPLE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# 配置参数
# ============================================================
CONFIG = {
    'WINDOW_SIZE': 15,
    'RAW_FEATURES': ['temperature', 'thickness', 'vibration_x', 'vibration_y', 'vibration_z'],
    'ADDED_FEATURES': [
        'thickness_loss_ratio', 'thickness_diff', 'temp_diff',
        'vib_total', 'vib_diff', 'thickness_ma', 'vib_ma',
        'temp_vib'
    ]
}

# ============================================================
# 配置日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================
# 特征工程类
# ============================================================
class FeatureEngineer:
    """特征工程增强器"""
    
    @staticmethod
    def add_degradation_features(df, window_size=10):
        df = df.copy()
        
        if 'dataset_id' not in df.columns:
            df['dataset_id'] = 'inference_device'
            
        groups = df.groupby('dataset_id')
        enhanced_dfs = []
        
        for device_id, group in groups:
            group = group.copy()
            
            # 厚度损失率
            initial_thickness = group['thickness'].iloc[0]
            group['thickness_loss_ratio'] = (initial_thickness - group['thickness']) / initial_thickness
            
            # 变化率
            group['thickness_diff'] = -group['thickness'].diff().fillna(0)
            group['temp_diff'] = group['temperature'].diff().fillna(0)
            
            # 振动总量
            group['vib_total'] = (group['vibration_x']**2 + 
                                 group['vibration_y']**2 + 
                                 group['vibration_z']**2) ** 0.5
            group['vib_diff'] = group['vib_total'].diff().fillna(0)
            
            # 移动平均
            group['thickness_ma'] = group['thickness'].rolling(window=window_size, min_periods=1).mean()
            group['vib_ma'] = group['vib_total'].rolling(window=window_size, min_periods=1).mean()
            
            # 温度×振动
            group['temp_vib'] = group['temperature'] * group['vib_total'] / 1000
            
            enhanced_dfs.append(group)
        
        if len(enhanced_dfs) > 1:
            enhanced_df = pd.concat(enhanced_dfs, ignore_index=True)
        else:
            enhanced_df = enhanced_dfs[0]
            
        return enhanced_df

# ============================================================
# RUL 预测器类
# ============================================================
class RULPredictor:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.window_size = CONFIG['WINDOW_SIZE']
        self.feature_cols = CONFIG['RAW_FEATURES'] + CONFIG['ADDED_FEATURES']
        self._load_artifacts()
        
    def _load_artifacts(self):
        """加载模型和归一化器"""
        logging.info(f"正在从 {self.model_dir} 加载模型组件...")
        
        # 加载 Scalers
        try:
            with open(self.model_dir / 'feature_scaler.pkl', 'rb') as f:
                self.feature_scaler = pickle.load(f)
            with open(self.model_dir / 'target_scaler.pkl', 'rb') as f:
                self.target_scaler = pickle.load(f)
            logging.info("✓ Scalers 加载成功")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"找不到Scaler文件: {e}")

        # 加载模型
        model_path = self.model_dir / 'final_model.h5'
        if not model_path.exists():
            model_path = self.model_dir / 'best_model.h5'
        
        if not model_path.exists():
            raise FileNotFoundError(f"找不到模型文件: {self.model_dir}")
        
        try:
            self.model = keras.models.load_model(model_path, compile=False)
            logging.info(f"✓ 模型加载成功: {model_path.name}")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")

    def preprocess(self, df):
        """数据预处理"""
        # 特征工程
        df_processed = FeatureEngineer.add_degradation_features(df, window_size=10)
        
        # 确保所有特征列都存在
        missing_cols = [c for c in self.feature_cols if c not in df_processed.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺失特征列: {missing_cols}")
            
        # 提取特征矩阵
        X_raw = df_processed[self.feature_cols].values
        
        # 归一化
        X_scaled = self.feature_scaler.transform(X_raw)
        
        # 生成滑动窗口
        X_windows = []
        valid_indices = []
        
        total_len = len(X_scaled)
        if total_len < self.window_size:
            logging.warning(f"数据长度 ({total_len}) 小于窗口大小 ({self.window_size})")
            return None, None, None
            
        for i in range(total_len - self.window_size + 1):
            window = X_scaled[i : i + self.window_size]
            X_windows.append(window)
            valid_indices.append(df_processed.index[i + self.window_size - 1])
            
        return np.array(X_windows), valid_indices, df_processed

    def predict(self, input_data):
        """执行预测"""
        if isinstance(input_data, str):
            logging.info(f"读取数据: {input_data}")
            df = pd.read_csv(input_data)
        else:
            df = input_data.copy()
            
        logging.info("执行预处理...")
        X_windows, valid_indices, df_engineered = self.preprocess(df)
        
        if X_windows is None or len(X_windows) == 0:
            return df, np.array([])
            
        logging.info(f"生成 {len(X_windows)} 个时间窗口，开始推理...")
        
        # 模型推理
        y_pred_scaled = self.model.predict(X_windows, verbose=0)
        
        # 反归一化
        y_pred_rul = self.target_scaler.inverse_transform(y_pred_scaled).flatten()
        
        # 将结果合并回DataFrame
        df_result = df_engineered.copy()
        df_result['Predicted_RUL'] = np.nan
        df_result.loc[valid_indices, 'Predicted_RUL'] = y_pred_rul
        
        # 平滑预测结果
        df_result['Predicted_RUL_Smooth'] = df_result['Predicted_RUL'].rolling(window=5, min_periods=1).mean()
        
        logging.info("✓ 推理完成")
        return df_result, y_pred_rul

# ============================================================
# 数据模拟器类
# ============================================================
class BeltDegradationSimulator:
    """驱动带退化过程模拟器"""
    
    def __init__(self, 
                 total_hours=1000,
                 sampling_rate=1,
                 initial_thickness=1.0,
                 ambient_temp=25.0,
                 expected_life_hours=2000,
                 warning_loss_ratio=0.15,
                 fault_loss_ratio=0.25):
        
        self.total_hours = total_hours
        self.sampling_rate = sampling_rate
        self.initial_thickness = initial_thickness
        self.ambient_temp = ambient_temp
        self.expected_life_hours = expected_life_hours
        self.warning_loss_ratio = warning_loss_ratio
        self.fault_loss_ratio = fault_loss_ratio
        
        # 动态计算阈值
        self.warning_thickness = initial_thickness * (1 - warning_loss_ratio)
        self.fault_thickness = initial_thickness * (1 - fault_loss_ratio)
        
        # 计算磨损速率
        total_wear = initial_thickness * fault_loss_ratio
        self.wear_rate = total_wear / expected_life_hours
        
        # 计算总采样点数
        samples_per_hour = 60
        self.n_samples = int(total_hours * samples_per_hour)
        self.sampling_rate = 1/60.0
        self.time_hours = np.linspace(0, total_hours, self.n_samples)
        
    def generate_temperature(self, noise_level=2.0):
        """生成温度数据"""
        base_temp = self.ambient_temp + 15 * (1 - np.exp(-self.time_hours / 500))
        daily_cycle = 5 * np.sin(2 * np.pi * self.time_hours / 24)
        load_variation = 3 * np.sin(2 * np.pi * self.time_hours / 2) * np.random.rand(self.n_samples)
        noise = np.random.normal(0, noise_level, self.n_samples)
        temperature = base_temp + daily_cycle + load_variation + noise
        return np.clip(temperature, self.ambient_temp, 80)
    
    def generate_thickness(self):
        """生成厚度数据"""
        linear_wear = self.wear_rate * self.time_hours
        
        accelerated_start_time = self.total_hours * 0.7
        time_in_accel_phase = np.maximum(0, self.time_hours - accelerated_start_time)
        accel_factor = (time_in_accel_phase / (self.total_hours - accelerated_start_time)) ** 2
        accelerated_wear = self.wear_rate * time_in_accel_phase * accel_factor * 2
        
        defect_start = int(self.n_samples * 0.5)
        defect_end = int(self.n_samples * 0.6)
        local_defect = np.zeros(self.n_samples)
        defect_amplitude = self.wear_rate * self.total_hours * 0.05
        local_defect[defect_start:defect_end] = defect_amplitude * np.sin(
            np.linspace(0, np.pi, defect_end - defect_start)
        )
        
        noise = np.random.normal(0, self.initial_thickness * 0.002, self.n_samples)
        thickness = self.initial_thickness - linear_wear - accelerated_wear - local_defect + noise
        min_thickness = self.fault_thickness * 0.6
        return np.clip(thickness, min_thickness, self.initial_thickness)
    
    def generate_vibration(self, thickness_data, temp_data):
        """生成三轴振动数据"""
        thickness_loss = 1 - thickness_data / self.initial_thickness
        temp_factor = 1 + 0.3 * (temp_data - self.ambient_temp) / 50
        
        # X轴振动
        base_vib_x = 4.0 + 0.5 * np.sin(2 * np.pi * self.time_hours / 12)
        degradation_x = 2.5 * thickness_loss * temp_factor
        high_freq_x = 0.3 * np.sin(2 * np.pi * self.time_hours * 50)
        noise_x = np.random.normal(0, 0.25, self.n_samples)
        vib_x = base_vib_x + degradation_x + high_freq_x + noise_x
        
        # Y轴振动
        base_vib_y = 4.5 + 1.5 * np.sin(2 * np.pi * self.time_hours / 8)
        degradation_y = 3.0 * thickness_loss * temp_factor
        high_freq_y = 0.4 * np.sin(2 * np.pi * self.time_hours * 35)
        noise_y = np.random.normal(0, 0.35, self.n_samples)
        vib_y = base_vib_y + degradation_y + high_freq_y + noise_y
        
        # Z轴振动
        base_vib_z = 6.0 + 1.0 * np.sin(2 * np.pi * self.time_hours / 10)
        degradation_z = 2.0 * thickness_loss * temp_factor
        high_freq_z = 0.35 * np.sin(2 * np.pi * self.time_hours * 25)
        noise_z = np.random.normal(0, 0.3, self.n_samples)
        vib_z = base_vib_z + degradation_z + high_freq_z + noise_z
        
        # 故障特征
        fault_indicator = self.time_hours > self.total_hours * 0.8
        vib_x += np.where(fault_indicator, 1.5 * np.sin(2 * np.pi * self.time_hours * 120), 0)
        vib_y += np.where(fault_indicator, 2.0 * np.sin(2 * np.pi * self.time_hours * 120), 0)
        vib_z += np.where(fault_indicator, 1.2 * np.sin(2 * np.pi * self.time_hours * 120), 0)
        
        vib_x = np.clip(vib_x, 2.5, 9.0)
        vib_y = np.clip(vib_y, 1.5, 12.0)
        vib_z = np.clip(vib_z, 3.5, 11.0)
        
        return vib_x, vib_y, vib_z
    
    def generate_health_label(self, thickness_data, vib_x, vib_y, vib_z):
        """生成健康相关标签"""
        current_wear = self.initial_thickness - thickness_data
        remaining_wear_capacity = self.initial_thickness * self.fault_loss_ratio - current_wear
        rul_hours = np.maximum(0, remaining_wear_capacity / self.wear_rate)
        
        vib_total = (vib_x + vib_y + vib_z) / 3
        vib_normal = (4.0 + 4.5 + 6.0) / 3
        vib_factor = np.clip(vib_total / vib_normal, 0.8, 2.0)
        rul_hours = rul_hours / vib_factor
        
        thickness_ratio = thickness_data / self.initial_thickness
        thickness_hi = thickness_ratio * 100
        vib_hi = 100 * (1 - (vib_total - vib_normal) / vib_normal / 2)
        vib_hi = np.clip(vib_hi, 0, 100)
        health_index = 0.6 * thickness_hi + 0.4 * vib_hi
        health_index = np.clip(health_index, 0, 100)
        
        health_state = np.zeros(self.n_samples, dtype=int)
        warning_condition = (rul_hours < self.expected_life_hours * 0.3) | (health_index < 85)
        fault_condition = (rul_hours < self.expected_life_hours * 0.1) | (health_index < 70)
        health_state[warning_condition] = 1
        health_state[fault_condition] = 2
        
        return rul_hours, health_index, health_state
    
    def generate_dataset(self):
        """生成完整数据集"""
        temperature = self.generate_temperature()
        thickness = self.generate_thickness()
        vib_x, vib_y, vib_z = self.generate_vibration(thickness, temperature)
        rul_hours, health_index, health_state = self.generate_health_label(
            thickness, vib_x, vib_y, vib_z
        )
        
        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=i/self.sampling_rate) 
                     for i in range(self.n_samples)]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'time_hours': self.time_hours,
            'temperature': temperature,
            'thickness': thickness,
            'vibration_x': vib_x,
            'vibration_y': vib_y,
            'vibration_z': vib_z,
            'RUL': rul_hours,
            'health_index': health_index,
            'health_state': health_state
        })
        
        return df

# ============================================================
# 全局变量
# ============================================================
predictor = None
model_options = {}
simulated_files = {}

# ============================================================
# 辅助函数
# ============================================================
def get_simulated_files():
    """获取所有模拟生成的文件"""
    global simulated_files
    simulated_files = {}
    
    if not DATA_DIR.exists():
        return simulated_files
    
    sim_files = list(DATA_DIR.glob("belt_data_*.csv"))
    
    for file_path in sorted(sim_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            filename = file_path.stem
            parts = filename.split('_')
            
            try:
                # 尝试解析时间戳
                if len(parts) >= 4:
                    timestamp = '_'.join(parts[2:])
                    try:
                        dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                        time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        time_str = timestamp
                    display_name = f"{filename.replace('_', ' ').title()} ({time_str})"
                else:
                    display_name = filename.replace('_', ' ').title()
            except:
                display_name = filename.replace('_', ' ').title()
            
            simulated_files[display_name] = str(file_path)
        except Exception as e:
            logging.warning(f"解析文件名失败: {file_path.name}, 错误: {e}")
            continue
    
    logging.info(f"找到 {len(simulated_files)} 个数据文件")
    return simulated_files

def initialize_models():
    """扫描并初始化模型选项"""
    global model_options
    model_options = {}
    
    logging.info(f"扫描模型目录: {MODEL_BASE_DIR}")
    
    if not MODEL_BASE_DIR.exists():
        logging.warning(f"模型目录不存在: {MODEL_BASE_DIR}")
        return
    
    # 扫描所有子目录
    for model_dir in MODEL_BASE_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        
        # 检查是否包含必要的模型文件
        has_model = (model_dir / 'final_model.h5').exists() or (model_dir / 'best_model.h5').exists()
        has_scaler = (model_dir / 'feature_scaler.pkl').exists() and (model_dir / 'target_scaler.pkl').exists()
        
        if has_model and has_scaler:
            model_options[model_dir.name] = str(model_dir)
            logging.info(f"  ✓ 发现模型: {model_dir.name}")
    
    logging.info(f"共发现 {len(model_options)} 个可用模型")

def load_model(model_name):
    """加载指定模型"""
    global predictor
    try:
        if model_name not in model_options:
            return f"❌ 模型不存在: {model_name}"
        
        model_dir = model_options[model_name]
        logging.info(f"加载模型: {model_name} ({model_dir})")
        
        predictor = RULPredictor(model_dir=model_dir)
        
        return f"✅ 模型加载成功\n\n模型名称: {model_name}\n模型路径: {model_dir}\n窗口大小: {CONFIG['WINDOW_SIZE']}\n特征数量: {len(CONFIG['RAW_FEATURES']) + len(CONFIG['ADDED_FEATURES'])}"
    except Exception as e:
        error_msg = f"❌ 模型加载失败: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return error_msg

def create_rul_visualization(df_result):
    """创建RUL预测可视化"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. RUL预测曲线
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_result['time_hours'], df_result['Predicted_RUL'], 
             label='预测RUL', color='blue', alpha=0.4, linewidth=1)
    ax1.plot(df_result['time_hours'], df_result['Predicted_RUL_Smooth'], 
             label='平滑RUL', color='red', linewidth=2)
    
    if 'RUL' in df_result.columns:
        ax1.plot(df_result['time_hours'], df_result['RUL'], 
                 label='真实RUL', color='green', linestyle='--', linewidth=2)
    
    ax1.set_xlabel('运行时间 (小时)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('剩余使用寿命 (小时)', fontsize=12, fontweight='bold')
    ax1.set_title('驱动带剩余使用寿命(RUL)预测', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. 温度曲线
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df_result['time_hours'], df_result['temperature'], 
             linewidth=1, color='orange', alpha=0.7)
    ax2.set_xlabel('运行时间 (小时)', fontsize=10)
    ax2.set_ylabel('温度 (℃)', fontsize=10, fontweight='bold')
    ax2.set_title('温度监测', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 厚度曲线
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df_result['time_hours'], df_result['thickness'], 
             linewidth=1, color='green', alpha=0.7)
    ax3.set_xlabel('运行时间 (小时)', fontsize=10)
    ax3.set_ylabel('厚度 (mm)', fontsize=10, fontweight='bold')
    ax3.set_title('厚度退化', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 三轴振动
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(df_result['time_hours'], df_result['vibration_x'], 
             linewidth=0.8, label='X轴', alpha=0.7)
    ax4.plot(df_result['time_hours'], df_result['vibration_y'], 
             linewidth=0.8, label='Y轴', alpha=0.7)
    ax4.plot(df_result['time_hours'], df_result['vibration_z'], 
             linewidth=0.8, label='Z轴', alpha=0.7)
    ax4.set_xlabel('运行时间 (小时)', fontsize=10)
    ax4.set_ylabel('振动 (m/s²)', fontsize=10, fontweight='bold')
    ax4.set_title('三轴振动监测', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 健康指数
    if 'health_index' in df_result.columns:
        ax5 = fig.add_subplot(gs[3, 0])
        ax5.plot(df_result['time_hours'], df_result['health_index'], 
                 linewidth=1.5, color='purple', alpha=0.8)
        ax5.axhline(y=85, color='orange', linestyle='--', linewidth=1, label='预警线')
        ax5.axhline(y=70, color='red', linestyle='--', linewidth=1, label='故障线')
        ax5.set_xlabel('运行时间 (小时)', fontsize=10)
        ax5.set_ylabel('健康指数', fontsize=10, fontweight='bold')
        ax5.set_title('健康指数', fontsize=11, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. RUL分布直方图
    ax6 = fig.add_subplot(gs[3, 1])
    rul_data = df_result['Predicted_RUL_Smooth'].dropna()
    if len(rul_data) > 0:
        ax6.hist(rul_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax6.axvline(rul_data.mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'均值: {rul_data.mean():.1f}h')
        ax6.set_xlabel('预测RUL (小时)', fontsize=10)
        ax6.set_ylabel('频数', fontsize=10, fontweight='bold')
        ax6.set_title('RUL分布', fontsize=11, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('驱动带退化监测与RUL预测分析', fontsize=16, fontweight='bold', y=0.995)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img

def generate_rul_report(df_result):
    """生成RUL分析报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("驱动带剩余使用寿命(RUL)预测报告")
    report_lines.append("=" * 80)
    report_lines.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"数据点数: {len(df_result)}")
    report_lines.append("")
    
    # RUL统计
    rul_data = df_result['Predicted_RUL_Smooth'].dropna()
    if len(rul_data) > 0:
        report_lines.append("【RUL预测统计】")
        report_lines.append("-" * 80)
        report_lines.append(f"  当前预测RUL: {rul_data.iloc[-1]:.1f} 小时")
        report_lines.append(f"  平均RUL: {rul_data.mean():.1f} 小时")
        report_lines.append(f"  最小RUL: {rul_data.min():.1f} 小时")
        report_lines.append(f"  最大RUL: {rul_data.max():.1f} 小时")
        report_lines.append("")
        
        # 健康状态评估
        current_rul = rul_data.iloc[-1]
        report_lines.append("【健康状态评估】")
        report_lines.append("-" * 80)
        if current_rul > 500:
            status = "健康"
            color = "🟢"
            suggestion = "设备运行正常，建议继续按常规周期进行维护检查。"
        elif current_rul > 200:
            status = "预警"
            color = "🟡"
            suggestion = "设备进入预警期，建议增加监测频率，准备备件，计划维护窗口。"
        else:
            status = "故障风险"
            color = "🔴"
            suggestion = "设备RUL较低，建议尽快安排维护或更换，避免意外停机。"
        
        report_lines.append(f"  状态: {color} {status}")
        report_lines.append(f"  建议: {suggestion}")
        report_lines.append("")
    
    # 退化趋势
    if 'thickness' in df_result.columns:
        report_lines.append("【退化趋势分析】")
        report_lines.append("-" * 80)
        initial_thickness = df_result['thickness'].iloc[0]
        current_thickness = df_result['thickness'].iloc[-1]
        wear_ratio = (initial_thickness - current_thickness) / initial_thickness * 100
        report_lines.append(f"  初始厚度: {initial_thickness:.3f} mm")
        report_lines.append(f"  当前厚度: {current_thickness:.3f} mm")
        report_lines.append(f"  磨损比例: {wear_ratio:.2f}%")
        report_lines.append("")
    
    # 振动状态
    if all(col in df_result.columns for col in ['vibration_x', 'vibration_y', 'vibration_z']):
        report_lines.append("【振动状态分析】")
        report_lines.append("-" * 80)
        vib_x_mean = df_result['vibration_x'].mean()
        vib_y_mean = df_result['vibration_y'].mean()
        vib_z_mean = df_result['vibration_z'].mean()
        report_lines.append(f"  X轴平均振动: {vib_x_mean:.2f} m/s² (正常范围: 3-5)")
        report_lines.append(f"  Y轴平均振动: {vib_y_mean:.2f} m/s² (正常范围: 2-7)")
        report_lines.append(f"  Z轴平均振动: {vib_z_mean:.2f} m/s² (正常范围: 4-8)")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("报告结束")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)

def predict_rul(csv_file, simulated_file_name, model_name):
    """RUL预测主函数"""
    global predictor
    
    # 检查模型选择
    if not model_options or model_name == "无可用模型" or model_name not in model_options:
        return None, "❌ 没有可用的模型！请先训练模型或检查模型目录。", None
    
    # 加载模型（如果需要）
    if predictor is None:
        status = load_model(model_name)
        if "失败" in status:
            return None, status, None
    
    # 确定数据源
    if simulated_file_name and simulated_file_name != "暂无数据文件" and simulated_file_name in simulated_files:
        file_path = simulated_files[simulated_file_name]
        data_source = f"模拟数据: {simulated_file_name}"
        logging.info(f"使用模拟文件: {file_path}")
    elif csv_file is not None:
        file_path = csv_file.name
        data_source = "上传文件"
        logging.info(f"使用上传文件: {file_path}")
    else:
        return None, "❌ 请上传CSV文件或选择数据文件！", None
    
    try:
        # 执行预测
        df_result, predictions = predictor.predict(file_path)
        
        if len(predictions) == 0:
            return None, "❌ 数据量不足，无法进行预测", None
        
        # 生成报告
        report_text = generate_rul_report(df_result)
        
        # 生成可视化
        viz_img = create_rul_visualization(df_result)
        
        # 保存结果
        output_path = DATA_DIR / f"rul_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_result.to_csv(output_path, index=False)
        
        # 保存报告
        report_path = DATA_DIR / f"rul_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"数据来源: {data_source}\n\n")
            f.write(report_text)
        
        result_summary = f"📊 RUL预测完成\n\n"
        result_summary += f"数据来源: {data_source}\n"
        result_summary += f"数据点数: {len(df_result)}\n"
        result_summary += f"有效预测: {len(predictions)} 个窗口\n"
        result_summary += f"预测文件: {output_path.name}\n"
        result_summary += f"分析报告: {report_path.name}\n"
        
        return result_summary, report_text, viz_img
        
    except Exception as e:
        error_msg = f"❌ 预测过程中出现错误: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return None, error_msg, None

def simulate_belt_data(total_hours, expected_life, initial_thickness, 
                      ambient_temp, warning_ratio, fault_ratio, show_viz):
    """模拟驱动带数据生成"""
    try:
        logging.info(f"开始生成模拟数据: 时长={total_hours}h, 寿命={expected_life}h")
        
        sim = BeltDegradationSimulator(
            total_hours=int(total_hours),
            initial_thickness=initial_thickness,
            ambient_temp=ambient_temp,
            expected_life_hours=int(expected_life),
            warning_loss_ratio=warning_ratio,
            fault_loss_ratio=fault_ratio
        )
        
        df = sim.generate_dataset()
        
        # 保存数据
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = DATA_DIR / f"belt_data_sim_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        
        result_text = f"✅ 数据生成成功！\n\n"
        result_text += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result_text += f"数据点数: {len(df)} 条\n"
        result_text += f"运行时长: {total_hours} 小时\n"
        result_text += f"预期寿命: {expected_life} 小时\n\n"
        result_text += f"【统计信息】\n"
        result_text += f"RUL范围: {df['RUL'].min():.1f} ~ {df['RUL'].max():.1f} 小时\n"
        result_text += f"平均健康指数: {df['health_index'].mean():.1f}\n"
        result_text += f"厚度范围: {df['thickness'].min():.3f} ~ {df['thickness'].max():.3f} mm\n\n"
        result_text += f"数据已保存至: {output_path.name}"
        
        viz_img = None
        if show_viz:
            fig, axes = plt.subplots(3, 2, figsize=(14, 10))
            fig.suptitle('驱动带退化模拟数据', fontsize=16, fontweight='bold')
            
            # 温度
            axes[0, 0].plot(df['time_hours'], df['temperature'], linewidth=0.8, alpha=0.7)
            axes[0, 0].set_title('温度')
            axes[0, 0].set_ylabel('℃')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 厚度
            axes[0, 1].plot(df['time_hours'], df['thickness'], linewidth=0.8, color='green')
            axes[0, 1].axhline(y=sim.warning_thickness, color='orange', linestyle='--', label='预警')
            axes[0, 1].axhline(y=sim.fault_thickness, color='red', linestyle='--', label='故障')
            axes[0, 1].set_title('厚度退化')
            axes[0, 1].set_ylabel('mm')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 振动
            axes[1, 0].plot(df['time_hours'], df['vibration_x'], linewidth=0.6, label='X', alpha=0.7)
            axes[1, 0].plot(df['time_hours'], df['vibration_y'], linewidth=0.6, label='Y', alpha=0.7)
            axes[1, 0].plot(df['time_hours'], df['vibration_z'], linewidth=0.6, label='Z', alpha=0.7)
            axes[1, 0].set_title('三轴振动')
            axes[1, 0].set_ylabel('m/s²')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # RUL
            axes[1, 1].plot(df['time_hours'], df['RUL'], linewidth=1.2, color='purple')
            axes[1, 1].fill_between(df['time_hours'], 0, df['RUL'], alpha=0.3, color='purple')
            axes[1, 1].set_title('剩余使用寿命(RUL)')
            axes[1, 1].set_ylabel('小时')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 健康指数
            axes[2, 0].plot(df['time_hours'], df['health_index'], linewidth=1.2, color='blue')
            axes[2, 0].axhline(y=85, color='orange', linestyle='--', label='预警')
            axes[2, 0].axhline(y=70, color='red', linestyle='--', label='故障')
            axes[2, 0].set_title('健康指数')
            axes[2, 0].set_xlabel('运行时间 (小时)')
            axes[2, 0].set_ylabel('指数')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # 健康状态分布
            health_counts = df['health_state'].value_counts().sort_index()
            colors = ['green', 'orange', 'red']
            labels = ['健康', '预警', '故障']
            axes[2, 1].bar(range(len(health_counts)), health_counts.values, 
                          color=colors[:len(health_counts)], alpha=0.7)
            axes[2, 1].set_xticks(range(len(health_counts)))
            axes[2, 1].set_xticklabels([labels[i] for i in health_counts.index])
            axes[2, 1].set_title('健康状态分布')
            axes[2, 1].set_ylabel('数据点数')
            axes[2, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            viz_img = Image.open(buf)
            plt.close()
        
        return str(output_path), result_text, viz_img
        
    except Exception as e:
        error_msg = f"❌ 数据生成失败: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return None, error_msg, None

def create_gradio_interface():
    """创建Gradio界面"""
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
    
    with gr.Blocks(title="驱动带RUL预测系统", js=health_check_js) as iface:
        gr.Markdown("""
        # 🔧 驱动带剩余使用寿命(RUL)预测系统
        **功能特点：** 基于深度学习的设备RUL预测与健康评估
        """)
        
        with gr.Tab("📊 RUL预测诊断"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔧 模型配置")
                    model_dropdown = gr.Dropdown(
                        choices=list(model_options.keys()) if model_options else ["无可用模型"],
                        value=list(model_options.keys())[0] if model_options else "无可用模型",
                        label="选择模型",
                        info="选择用于预测的RUL模型"
                    )
                    
                    model_status = gr.Textbox(
                        label="模型状态",
                        value="请选择模型...",
                        interactive=False,
                        lines=5,
                        visible=False
                    )
                    
                    gr.Markdown("### 📁 数据输入 (优先使用模拟数据)")
                    
                    simulated_dropdown = gr.Dropdown(
                        choices=list(simulated_files.keys()) if simulated_files else ["暂无数据文件"],
                        value=list(simulated_files.keys())[0] if simulated_files else "暂无数据文件",
                        label="选择数据文件",
                        info="选择模拟生成的数据文件（优先）"
                    )
                    
                    refresh_btn = gr.Button("🔄 刷新文件列表", size="sm")
                    
                    # gr.Markdown("或")
                    
                    csv_input = gr.File(
                        label="上传CSV文件 (备选)",
                        file_types=[".csv"],
                        type="filepath",
                        visible=False
                    )
                    
                    predict_btn = gr.Button("🔍 开始预测", variant="primary", size="lg")
                    
                    with gr.Accordion("📋 使用说明", open=False):
                        gr.Markdown("""
                        **使用步骤：**
                        1. 选择模型（自动加载）
                        2. 选择数据文件或上传CSV
                        3. 点击"开始预测"按钮
                        
                        **输入数据要求：**
                        - 必需列: temperature, thickness, vibration_x, vibration_y, vibration_z
                        - 可选列: time_hours, RUL (用于对比)
                        
                        **输出内容：**
                        - RUL预测曲线
                        - 退化趋势分析
                        - 健康状态评估
                        - 维护建议
                        """)
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 预测结果")
                    result_summary = gr.Textbox(label="预测概要", lines=6, interactive=False)
                    
                    with gr.Row():
                        visualization = gr.Image(label="可视化分析", height=600)
                    
                    rul_report = gr.Textbox(label="详细分析报告", lines=20, interactive=False)
        
        with gr.Tab("🎯 数据模拟生成"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 模拟参数配置")
                    
                    total_hours_slider = gr.Slider(
                        minimum=100,
                        maximum=3000,
                        value=1000,
                        step=100,
                        label="模拟运行时长 (小时)"
                    )
                    
                    expected_life_slider = gr.Slider(
                        minimum=500,
                        maximum=5000,
                        value=2000,
                        step=100,
                        label="预期设备寿命 (小时)"
                    )
                    
                    thickness_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="初始厚度 (mm)"
                    )
                    
                    temp_slider = gr.Slider(
                        minimum=10,
                        maximum=50,
                        value=25,
                        step=5,
                        label="环境温度 (℃)"
                    )
                    
                    warning_ratio_slider = gr.Slider(
                        minimum=0.10,
                        maximum=0.25,
                        value=0.15,
                        step=0.01,
                        label="预警磨损比例"
                    )
                    
                    fault_ratio_slider = gr.Slider(
                        minimum=0.20,
                        maximum=0.40,
                        value=0.25,
                        step=0.01,
                        label="故障磨损比例"
                    )
                    
                    show_viz_checkbox = gr.Checkbox(
                        label="显示可视化结果",
                        value=True
                    )
                    
                    simulate_btn = gr.Button("🎯 生成模拟数据", variant="primary", size="lg")
                    
                    with gr.Accordion("📋 使用说明", open=False):
                        gr.Markdown("""
                        **使用步骤：**
                        1. 调整模拟参数
                        2. 点击"生成模拟数据"按钮
                        3. 生成的数据会自动保存到data目录
                        4. 可在预测标签页使用生成的数据
                        
                        **参数说明：**
                        - 运行时长：模拟的设备运行总时长
                        - 预期寿命：设备设计寿命
                        - 预警/故障比例：厚度损失到该比例时触发对应状态
                        """)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 生成结果")
                    sim_result_file = gr.Textbox(label="数据文件路径", interactive=False)
                    sim_result_text = gr.Textbox(label="生成统计", lines=15, interactive=False)
                    
                    sim_viz_output = gr.Image(label="数据可视化", height=600)
        
        # 模型下拉菜单改变时自动加载
        model_dropdown.change(
            load_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
        # 刷新文件列表
        def refresh_files():
            get_simulated_files()
            choices = list(simulated_files.keys()) if simulated_files else ["暂无数据文件"]
            value = choices[0] if simulated_files else "暂无数据文件"
            return gr.update(choices=choices, value=value)
        
        refresh_btn.click(refresh_files, outputs=[simulated_dropdown])
        
        # 预测按钮
        predict_btn.click(
            predict_rul,
            inputs=[csv_input, simulated_dropdown, model_dropdown],
            outputs=[result_summary, rul_report, visualization]
        )
        
        # 模拟按钮
        simulate_btn.click(
            simulate_belt_data,
            inputs=[
                total_hours_slider, expected_life_slider, thickness_slider,
                temp_slider, warning_ratio_slider, fault_ratio_slider,
                show_viz_checkbox
            ],
            outputs=[sim_result_file, sim_result_text, sim_viz_output]
        )
    
    return iface

def main():
    """主函数"""
    print(f"\n{'='*80}")
    print("驱动带RUL预测系统 Gradio 应用")
    print(f"{'='*80}\n")
    
    # 配置GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"✓ GPU配置成功: {len(gpus)} 个GPU可用")
        except RuntimeError as e:
            logging.warning(f"GPU配置失败: {e}")
    
    # 初始化
    initialize_models()
    get_simulated_files()
    
    # 目录监控（可选）
    monitor_manager = None
    if MultiDirectoryMonitor is not None:
        monitor_manager = MultiDirectoryMonitor(restart_signal_file_name=RESTART_SIGNAL_FILENAME)
        monitor_manager.add_directory(MODEL_BASE_DIR)
        if EXAMPLE_DIR.exists():
            monitor_manager.add_directory(EXAMPLE_DIR)
        
        if not monitor_manager.start_all():
            logging.error("❌ 启动目录监控失败")
        else:
            logging.info("✅ 目录监控已启动")
    
    # 获取端口
    port = 7865
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            if port < 1024 or port > 65535:
                logging.warning(f"端口号 {port} 不在有效范围内，使用默认端口 7865")
                port = 7865
        except ValueError:
            logging.warning(f"无效的端口号参数，使用默认端口 7865")
    
    # 创建并启动界面
    iface = create_gradio_interface()
    
    try:
        iface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False
        )
    finally:
        if monitor_manager is not None:
            monitor_manager.stop_all(join_threads=True)
            logging.info("目录监控已停止")

if __name__ == '__main__':
    main()