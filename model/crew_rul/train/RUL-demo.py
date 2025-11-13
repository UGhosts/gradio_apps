from collections import defaultdict
from datetime import timedelta,datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import signal, stats
import os
import glob
import pickle
import time
import warnings
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, BatchNormalization, 
                                      Bidirectional, Concatenate, Conv1D, MaxPooling1D,
                                      GlobalAveragePooling1D, Layer)
from minio import Minio
import json
from tensorflow.keras import backend as K

warnings.filterwarnings('ignore')
from tensorflow.keras import losses,metrics
# GPU优化
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class Config:
    DATA_DIR = '/home/software/gradio_apps/dataset/crew_rul'
    SAVE_DIR = '/home/software/gradio_apps/model/crew_rul'
    WINDOW_SIZE = 25
    STRIDE = 3
    BATCH_SIZE = 32
    EPOCHS = 150
    LEARNING_RATE = 0.001
    FS = 25600  # 采样频率
    
    # RUL标签配置
    RUL_CAP = 150.0  # 健康期RUL上限（以文件数为单位）
    DEGRADATION_FALLBACK_RATIO = 0.6
    CHANGE_POINT_SMOOTH_WINDOW = 15
    CHANGE_POINT_HEALTHY_RATIO = 0.25
    
    # 时间转换配置
    SAMPLING_INTERVAL_SECONDS = 60
    
    # 模型路径
    # MODEL_VERSION = 4100
    MODEL_VERSION = 4400
    
    @property
    def MODEL_PATH(self):
        return f'{self.SAVE_DIR}/rul_model_v{self.MODEL_VERSION}.h5'
    
    @property
    def SCALER_PATH(self):
        return f'{self.SAVE_DIR}/scalers_v{self.MODEL_VERSION}.pkl'
    
    BEARINGS_INFO = {
        '35Hz12kN/Bearing1_1': {'files': 123, 'lifetime': 123},
        '35Hz12kN/Bearing1_3': {'files': 158, 'lifetime': 158},
        '35Hz12kN/Bearing1_4': {'files': 122, 'lifetime': 122},
        '35Hz12kN/Bearing1_5': {'files': 52, 'lifetime': 52},
        '37.5Hz11kN/Bearing2_1': {'files': 491, 'lifetime': 491},
        '37.5Hz11kN/Bearing2_2': {'files': 161, 'lifetime': 161},
        '37.5Hz11kN/Bearing2_3': {'files': 533, 'lifetime': 533},
        '37.5Hz11kN/Bearing2_5': {'files': 339, 'lifetime': 339},
        '40Hz10kN/Bearing3_1': {'files': 2538, 'lifetime': 2538},
        '40Hz10kN/Bearing3_3': {'files': 371, 'lifetime': 371},
        '40Hz10kN/Bearing3_4': {'files': 1515, 'lifetime': 1515},
        '40Hz10kN/Bearing3_5': {'files': 114, 'lifetime': 114},
    }

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
        # inputs shape: (batch, time_steps, features)
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = inputs * a
        return K.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class MinioConfig:
    """安全地配置MinIO连接参数"""
    ENDPOINT = os.getenv('MINIO_ENDPOINT', '192.168.37.160:9000')
    ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'n7nFX93gluYrX2j2bhKT')
    SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'wQhCHuChMQMU49FIedIvHSP3juyQ4cyFCpSBdyvb')
    SECURE = "https" in ENDPOINT.lower()

def create_features_for_signal(sig, fs=25600):
    """为单通道信号提取时域和频域特征"""
    # 时域特征
    rms = np.sqrt(np.mean(sig**2))
    kurtosis = stats.kurtosis(sig)
    skewness = stats.skew(sig)
    peak_to_peak = np.max(sig) - np.min(sig)
    
    # 避免除零
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
    
    # 频域特征
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
        
        # 带通滤波
        b, a = signal.butter(4, [20, 10000], btype='band', fs=25600)
        h = signal.filtfilt(b, a, h)
        v = signal.filtfilt(b, a, v)
        
        features_h = create_features_for_signal(h)
        features_v = create_features_for_signal(v)
        
        return np.array(features_h + features_v, dtype=np.float32)
    except Exception as e:
        print(f"    警告: 处理文件 {csv_path} 时出错: {e}")
        return None

def find_degradation_start_point(features_sequence):
    kurtosis_signal = np.maximum(features_sequence[:, 0], features_sequence[:, 9])
    
    # 计算整体平均值
    mean_value = np.mean(kurtosis_signal)
    
    # 寻找第一个超过平均值的点
    for i, value in enumerate(kurtosis_signal):
        if value > mean_value:
            print(f"检测到退化起始点: {i} (值: {value:.3f} > 平均值: {mean_value:.3f})")
            return i
    
    # 如果没有找到超过平均值的点，返回最后一个点
    fallback_point = len(kurtosis_signal) - 1
    print(f"未找到超过平均值的点，使用最后一点: {fallback_point}")
    return fallback_point

def compute_advanced_rul(n_timesteps, lifetime, degradation_start_idx, config):
    """生成带有上限和线性衰减的RUL标签"""
    rul = np.zeros(n_timesteps, dtype=np.float32)
    
    # 健康期: RUL上限
    rul[:degradation_start_idx] = config.RUL_CAP
    
    # 退化期: 线性衰减
    degradation_len = n_timesteps - degradation_start_idx
    if degradation_len > 0:
        # 从RUL_CAP线性衰减到0
        linear_decay = np.linspace(config.RUL_CAP, 0, num=degradation_len)
        rul[degradation_start_idx:] = np.maximum(0, linear_decay)
    
    # 对数变换
    return np.log(rul + 1)

def load_bearing_features(bearing_name, info, config):
    """加载轴承特征和RUL标签"""
    # 检查bearing_name是否已经是完整路径
    if os.path.isdir(bearing_name):
        bearing_path = bearing_name
    else:
        bearing_path = os.path.join(config.DATA_DIR, bearing_name)
    
    key_func = lambda f: int(os.path.splitext(os.path.basename(f))[0])
    csv_files = sorted(glob.glob(os.path.join(bearing_path, '*.csv')), key=key_func)
    
    if not csv_files:
        print(f"    警告: 未找到CSV文件")
        return None, None
    
    print(f"    提取 {len(csv_files)} 个文件...", end=" ", flush=True)
    features_list = []
    for file in csv_files:
        fv = load_and_process_csv_features(file)
        if fv is not None:
            features_list.append(fv)
    
    if len(features_list) < config.WINDOW_SIZE:
        print(f"有效文件不足 (需要 >= {config.WINDOW_SIZE})")
        return None, None
    
    features_sequence = np.array(features_list)
    print("完成!", end=" ")
    
    # 生成RUL标签
    degradation_start_idx = find_degradation_start_point(features_sequence)
    log_rul_labels = compute_advanced_rul(
        len(features_sequence), info['lifetime'], degradation_start_idx, config
    )
    
    print(f"特征形状: {features_sequence.shape}")
    return features_sequence, log_rul_labels

def create_windows(features, labels, window_size, stride):
    """创建滑动窗口"""
    X, y = [], []
    for i in range(0, len(features) - window_size + 1, stride):
        X.append(features[i:i+window_size])
        y.append(labels[i+window_size-1])
    return np.array(X), np.array(y)

def create_balanced_windows(features_dict, labels_dict, config, target_windows=300, oversample_minority=True):
    """
    为每个轴承创建滑动窗口，并进行重采样以平衡数据集。

    Args:
        features_dict (dict): 字典，键为轴承名，值为其特征序列。
        labels_dict (dict): 字典，键为轴承名，值为其RUL标签序列。
        config (Config): 配置类实例。
        target_windows (int): 每个轴承的目标窗口数。
        oversample_minority (bool): 是否对窗口数不足的轴承进行过采样。

    Returns:
        tuple: (平衡后的X, 平衡后的y, 平衡后的bearing_ids)
    """
    X_balanced, y_balanced, bearing_ids_balanced = [], [], []
    
    print(f"  正在进行数据集平衡，目标窗口数: {target_windows}...")
    
    for i, bearing_name in enumerate(features_dict.keys()):
        features = features_dict[bearing_name]
        labels = labels_dict[bearing_name]
        
        # 1. 为当前轴承生成所有可能的窗口
        X_bearing, y_bearing = create_windows(
            features, labels, config.WINDOW_SIZE, config.STRIDE
        )
        
        num_windows = len(X_bearing)
        if num_windows == 0:
            continue
            
        # 2. 根据目标窗口数进行重采样
        if num_windows > target_windows:
            # 欠采样: 随机选择 'target_windows' 个窗口
            print(f"    轴承 {bearing_name}: 欠采样 {num_windows} -> {target_windows}")
            indices = np.random.choice(num_windows, target_windows, replace=False)
            X_sampled, y_sampled = X_bearing[indices], y_bearing[indices]
            
        elif num_windows < target_windows and oversample_minority:
            # 过采样: 有放回地随机选择，直到达到 'target_windows'
            print(f"    轴承 {bearing_name}: 过采样 {num_windows} -> {target_windows}")
            indices = np.random.choice(num_windows, target_windows, replace=True)
            X_sampled, y_sampled = X_bearing[indices], y_bearing[indices]
        else:
            # 数量适中或不进行过采样
            X_sampled, y_sampled = X_bearing, y_bearing

        X_balanced.append(X_sampled)
        y_balanced.append(y_sampled)
        bearing_ids_balanced.extend([i] * len(X_sampled))

    return (
        np.concatenate(X_balanced, axis=0), 
        np.concatenate(y_balanced, axis=0), 
        np.array(bearing_ids_balanced)
    )

def create_lstm_model_on_features(window_size, num_features):
    """
    Inception风格的多尺度卷积 + 双向LSTM
    同时提取短期、中期、长期模式
    
    核心思想: 类似Inception模块，并行使用不同大小的卷积核
    """
    inputs = Input(shape=(window_size, num_features))
    x = BatchNormalization(name='input_bn')(inputs)
    
    # =========================================================================
    # Inception模块: 多尺度并行卷积
    # =========================================================================
    # 分支1: 小卷积核 - 捕获局部高频特征
    conv1x1 = Conv1D(32, kernel_size=1, activation='relu', padding='same', 
                     name='conv_1x1')(x)
    conv1x1 = BatchNormalization(name='bn_conv1x1')(conv1x1)
    
    # 分支2: 中等卷积核 - 捕获短期模式
    conv3 = Conv1D(32, kernel_size=3, activation='relu', padding='same',
                   name='conv_3')(x)
    conv3 = BatchNormalization(name='bn_conv3')(conv3)
    
    # 分支3: 大卷积核 - 捕获中期趋势
    conv5 = Conv1D(32, kernel_size=5, activation='relu', padding='same',
                   name='conv_5')(x)
    conv5 = BatchNormalization(name='bn_conv5')(conv5)
    
    # 分支4: 更大卷积核 - 捕获长期模式
    conv7 = Conv1D(32, kernel_size=7, activation='relu', padding='same',
                   name='conv_7')(x)
    conv7 = BatchNormalization(name='bn_conv7')(conv7)
    
    # 分支5: 最大池化 + 1x1卷积 - 保留显著特征
    maxpool = MaxPooling1D(pool_size=3, strides=1, padding='same', 
                           name='max_pool')(x)
    maxpool_conv = Conv1D(32, kernel_size=1, activation='relu', padding='same',
                          name='conv_maxpool')(maxpool)
    maxpool_conv = BatchNormalization(name='bn_maxpool')(maxpool_conv)
    
    # 拼接所有分支
    inception_out = Concatenate(name='inception_concat')([
        conv1x1, conv3, conv5, conv7, maxpool_conv
    ])
    
    # =========================================================================
    # 第二层多尺度卷积（可选，增强特征提取）
    # =========================================================================
    conv_shallow = Conv1D(64, kernel_size=3, activation='relu', padding='same',
                          name='conv_shallow')(inception_out)
    conv_shallow = BatchNormalization(name='bn_shallow')(conv_shallow)
    conv_shallow = Dropout(0.2, name='dropout_cnn')(conv_shallow)
    
    # =========================================================================
    # 双向LSTM层 - 捕获时序依赖
    # =========================================================================
    lstm1 = Bidirectional(
        LSTM(96, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name='bi_lstm_1'
    )(conv_shallow)
    lstm1 = BatchNormalization(name='bn_lstm1')(lstm1)
    
    lstm2 = Bidirectional(
        LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name='bi_lstm_2'
    )(lstm1)
    lstm2 = BatchNormalization(name='bn_lstm2')(lstm2)
    
    # =========================================================================
    # 注意力机制
    # =========================================================================
    attention_out = TemporalAttention(name='temporal_attention')(lstm2)
    gap = GlobalAveragePooling1D(name='global_avg_pool')(lstm2)
    
    # 特征融合
    combined = Concatenate(name='feature_fusion')([attention_out, gap])
    
    # =========================================================================
    # 全连接层
    # =========================================================================
    dense1 = Dense(128, activation='relu', name='dense_1')(combined)
    dense1 = Dropout(0.4, name='dropout_1')(dense1)
    dense1 = BatchNormalization(name='bn_dense1')(dense1)
    
    dense2 = Dense(64, activation='relu', name='dense_2')(dense1)
    dense2 = Dropout(0.3, name='dropout_2')(dense2)
    
    outputs = Dense(1, activation='linear', dtype='float32', name='output')(dense2)
    
    model = Model(inputs=inputs, outputs=outputs, name='Inception_CNN_LSTM_RUL')
    return model

def calculate_metrics(y_true, y_pred, config, set_name=""):
    """计算详细评估指标"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # 分段评估
    health_threshold = config.RUL_CAP
    high_rul_mask = y_true > health_threshold
    low_rul_mask = y_true <= health_threshold
    
    mape_high = 0
    mape_low = 0
    
    if np.sum(high_rul_mask) > 0:
        mape_high = np.mean(np.abs((y_true[high_rul_mask] - y_pred[high_rul_mask]) / 
                                    (y_true[high_rul_mask] + 1e-6))) * 100
    
    if np.sum(low_rul_mask) > 0:
        mape_low = np.mean(np.abs((y_true[low_rul_mask] - y_pred[low_rul_mask]) / 
                                   (y_true[low_rul_mask] + 1e-6))) * 100
    
    print(f"\n{'='*60}")
    print(f"{set_name}评估结果:")
    print(f"{'='*60}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE (健康期, RUL > {health_threshold:.0f}): {mape_high:.2f}%")
    print(f"  MAPE (退化期, RUL ≤ {health_threshold:.0f}): {mape_low:.2f}%")
    print(f"{'='*60}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape_high': mape_high, 'mape_low': mape_low}

def plot_predictions(y_true, y_pred, title, save_path=None):
    """绘制预测结果对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 真实值 vs 预测值散点图
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
    axes[0, 0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True RUL')
    axes[0, 0].set_ylabel('Predicted RUL')
    axes[0, 0].set_title('True vs Predicted RUL')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 误差分布
    errors = y_pred - y_true
    axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Error Distribution (Mean: {errors.mean():.2f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 时间序列对比
    sample_size = min(500, len(y_true))
    indices = np.random.choice(len(y_true), sample_size, replace=False)
    indices = np.sort(indices)
    
    axes[1, 0].plot(indices, y_true[indices], 'b-', label='True', alpha=0.7)
    axes[1, 0].plot(indices, y_pred[indices], 'r-', label='Predicted', alpha=0.7)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('RUL')
    axes[1, 0].set_title('RUL Time Series Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 残差图
    axes[1, 1].scatter(y_pred, errors, alpha=0.5, s=10)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted RUL')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residual Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图表已保存: {save_path}")
    
    plt.close()

def train_model():
    start_time = time.time()
    config = Config()
    
    print("="*70)
    print("RUL预测系统 v4.0 (改进版)")
    print("="*70)
    print("核心策略: [高级特征] + [数据驱动RUL标签] + [LSTM模型]")
    print("="*70)
    
    # 1. 加载数据
    print("\n[1/4] 提取特征并生成高级RUL标签...")
    # 修改：使用字典来存储每个轴承的数据
    all_features = {}
    all_log_ruls = {}
    
    for i, (name, info) in enumerate(config.BEARINGS_INFO.items()):
        print(f"\n处理 [{i+1}/{len(config.BEARINGS_INFO)}]: {name}")
        features, log_rul = load_bearing_features(name, info, config)
        
        if features is not None:
            # 存入字典
            all_features[name] = features
            all_log_ruls[name] = log_rul
    
    if not all_features:
        print("\n错误: 没有成功加载任何轴承数据!")
        return
    
    # 新增：调用平衡函数来创建数据集
    # target_windows 可以根据你的数据集进行调整，中位数通常是个不错的选择
    # 窗口数大约是 (文件数 - WINDOW_SIZE) / STRIDE
    # Bearing2_2: (161-25)/3 ≈ 45, Bearing2_5: (339-25)/3 ≈ 104, Bearing3_3: (371-25)/3 ≈ 115
    # 我们选择一个介于短寿命和中等寿命之间的值，比如 150 或 200
    X, y_log, bearing_ids = create_balanced_windows(
        all_features, all_log_ruls, config, target_windows=200 
    )

    print(f"\n平衡后的总窗口数: {len(X)}, 形状: {X.shape}")
    
    # 2. 数据预处理
    print("\n[2/4] 数据预处理...")
    # 步骤 2.1: 先划分训练集和验证集
    unique_bearings = np.unique(bearing_ids)
    np.random.seed(42)
    val_size = max(2, int(len(unique_bearings) * 0.2))
    val_bearing_indices = np.random.choice(unique_bearings, size=val_size, replace=False)
    
    train_mask = ~np.isin(bearing_ids, val_bearing_indices)
    val_mask = np.isin(bearing_ids, val_bearing_indices)
    
    X_train, y_train_log = X[train_mask], y_log[train_mask]
    X_val, y_val_log = X[val_mask], y_log[val_mask]
    
    # 步骤 2.2: 再进行标准化 (Scaler只在训练集上fit)
    scaler_X = StandardScaler()
    # Reshape to 2D for scaler
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler_X.fit(X_train_reshaped)
    
    # Transform train and val sets
    X_train_scaled_reshaped = scaler_X.transform(X_train_reshaped)
    X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
    X_val_scaled_reshaped = scaler_X.transform(X_val_reshaped)
    
    # Reshape back to 3D
    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)
    X_val_scaled = X_val_scaled_reshaped.reshape(X_val.shape)
    
    print(f"  训练集: {len(X_train_scaled)} | 验证集: {len(X_val_scaled)}")
    
    # 3. 训练模型
    print("\n[3/4] 训练模型...")
    model = create_lstm_model_on_features(config.WINDOW_SIZE, X_train_scaled.shape[-1])
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    print("\n模型结构:")
    model.summary()
    
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            f'{config.SAVE_DIR}/best_model_v{config.MODEL_VERSION}.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train_scaled, y_train_log,
        validation_data=(X_val_scaled, y_val_log),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 4. 模型评估
    print("\n[4/4] 模型评估...")
    
    # 预测
    log_y_train_pred = model.predict(X_train_scaled, batch_size=512, verbose=0).flatten()
    log_y_val_pred = model.predict(X_val_scaled, batch_size=512, verbose=0).flatten()
    
    # 反变换到原始RUL空间
    y_train_true_rul = np.expm1(y_train_log)
    y_val_true_rul = np.expm1(y_val_log)
    y_train_pred_rul = np.maximum(0, np.expm1(log_y_train_pred))
    y_val_pred_rul = np.maximum(0, np.expm1(log_y_val_pred))
    
    # 计算评估指标
    train_metrics = calculate_metrics(y_train_true_rul, y_train_pred_rul, config, "训练集")
    val_metrics = calculate_metrics(y_val_true_rul, y_val_pred_rul, config, "验证集")
    
    # 绘制预测结果
    plot_predictions(
        y_train_true_rul, y_train_pred_rul,
        "Training Set Predictions",
        f'{config.SAVE_DIR}/train_predictions.png'
    )
    plot_predictions(
        y_val_true_rul, y_val_pred_rul,
        "Validation Set Predictions",
        f'{config.SAVE_DIR}/val_predictions.png'
    )
    
    # 保存训练历史
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'{config.SAVE_DIR}/training_history.csv', index=False)
    
    # 保存模型和scaler
    print("\n保存模型...")
    model.save(f'{config.SAVE_DIR}/rul_model_v{config.MODEL_VERSION}.h5')
    with open(f'{config.SAVE_DIR}/scalers_v{config.MODEL_VERSION}.pkl', 'wb') as f:
        pickle.dump({'scaler_X': scaler_X, 'config': config}, f)
    
    # 保存评估结果
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'training_time_minutes': (time.time() - start_time) / 60
    }
    with open(f'{config.SAVE_DIR}/evaluation_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*70}")
    print(f"✓ 训练完成！总用时: {results['training_time_minutes']:.1f} 分钟")
    print(f"{'='*70}")

def predict_rul(data_path):
    """对给定的数据文件夹进行RUL预测"""
    config = Config()
    
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.SCALER_PATH):
        print("错误: 找不到模型或scaler文件。请先运行 'train' 命令。")
        return None
    
    print("正在加载模型和scaler...")
    model = load_model(config.MODEL_PATH,custom_objects={'TemporalAttention': TemporalAttention(),
                                            'mse': losses.MeanSquaredError()
                                            , 'mae': metrics.MeanAbsoluteError()})    
    
    with open(config.SCALER_PATH, 'rb') as f:
        saved_data = pickle.load(f)
        scaler_X = saved_data['scaler_X']
        
    key_func = lambda f: int(os.path.splitext(os.path.basename(f))[0])

    files = sorted(glob.glob(os.path.join(data_path, '*.csv')), key=key_func)
    
    if len(files) < config.WINDOW_SIZE:
        print(f"错误: 数据点不足。需要至少 {config.WINDOW_SIZE} 个CSV文件，")
        print(f"但只找到 {len(files)} 个。")
        return None
    
    print(f"正在从 '{data_path}' 中提取最新 {config.WINDOW_SIZE} 个数据点的特征...")
    
    # 只处理最新的 WINDOW_SIZE 个文件
    start_index = 1500
    # latest_files = files[start_index:start_index + config.WINDOW_SIZE]
    latest_files = files[-config.WINDOW_SIZE:]
    features_list = []
    
    for f in latest_files:
        fv = load_and_process_csv_features(f)
        if fv is not None:
            features_list.append(fv)
    
    if len(features_list) < config.WINDOW_SIZE:
        print(f"错误: 有效特征不足 {config.WINDOW_SIZE} 个")
        return None
    
    features = np.array(features_list)
    
    # 数据归一化并重塑
    features_scaled = scaler_X.transform(features)
    input_data = np.expand_dims(features_scaled, axis=0)
    
    print("正在预测RUL...")
    prediction_array = model.predict(input_data, verbose=0)
    
    # 从预测数组中提取标量值
    log_rul_pred = prediction_array[0][0]
    predicted_rul = np.maximum(0, np.expm1(log_rul_pred))
    
    print("\n" + "="*60)
    print("RUL 预测结果")
    print("="*60)
    print(predicted_rul)

    print("="*60)
    
    # 健康状态评估
    health_threshold = config.RUL_CAP * 0.5
    if predicted_rul > health_threshold:
        status = "健康"
        color = "✓"
    elif predicted_rul > health_threshold * 0.5:
        status = "轻微退化"
        color = "⚠"
    else:
        status = "严重退化，建议维护"
        color = "✗"
    
    print(f"\n{color} 轴承健康状态: {status}")
    print("="*60 + "\n")
    
    return {
        'rul_cycles': predicted_rul,
        'status': status
    }

def process_and_extract_features_from_dataframe(df, fs=25600):
    """从内存中的DataFrame提取特征，替代从文件路径读取"""
    try:
        h = df.iloc[:, 0].values
        v = df.iloc[:, 1].values
        
        # 带通滤波
        b, a = signal.butter(4, [20, 10000], btype='band', fs=fs)
        h = signal.filtfilt(b, a, h)
        v = signal.filtfilt(b, a, v)
        
        # (如果您使用了强化后的特征提取，请确保这里调用的是强化后的函数)
        # create_enhanced_features_for_signal 或 create_features_for_signal
        features_h = create_features_for_signal(h, fs=fs) 
        features_v = create_features_for_signal(v, fs=fs)
        
        return np.array(features_h + features_v, dtype=np.float32)
    except Exception as e:
        print(f"    警告: 处理DataFrame时出错: {e}")
        return None

def predict_rul_from_minio(bucket_name, prefix):
    """
    从MinIO路径加载数据并进行RUL预测。
    此版本会自动向前追溯几天的数据，以寻找一个完整、连续的时间窗口。
    """
    config = Config()
    minio_config = MinioConfig()

    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.SCALER_PATH):
        print("错误: 找不到模型或scaler文件。请先运行 'train' 命令。")
        return None
        
    print("正在加载模型和scaler...")
    model = load_model(config.MODEL_PATH, custom_objects={'TemporalAttention': TemporalAttention(),
                                            'mse': losses.MeanSquaredError(), 'mae': metrics.MeanAbsoluteError()})
    with open(config.SCALER_PATH, 'rb') as f:
        saved_data = pickle.load(f)
        scaler_X = saved_data['scaler_X']

    try:
        print(f"正在连接到 MinIO 服务器: {minio_config.ENDPOINT}...")
        client = Minio(
            minio_config.ENDPOINT,
            access_key=minio_config.ACCESS_KEY,
            secret_key=minio_config.SECRET_KEY,
            secure=minio_config.SECURE
        )
        print("✓ 连接成功!")
    except Exception as e:
        print(f"✗ 错误: 无法连接到 MinIO. 请检查您的配置和网络。错误信息: {e}")
        return None

    # --- **新的、跨天的数据获取逻辑** ---
    print("\n" + "="*60)
    print("开始执行跨天数据搜索...")
    SEARCH_DAYS_LIMIT = 2  # 定义最多向前搜索几天
    all_csv_objects = []

    try:
        # 1. 从输入prefix解析出基础路径和开始日期
        prefix_parts = prefix.strip('/').split('/')
        year, month, day = map(int, prefix_parts[-3:])
        start_date = datetime(year, month, day).date()
        base_prefix = "/".join(prefix_parts[:-3])
        print(f"起始搜索日期: {start_date}, 基础路径: '{base_prefix}'")

        # 2. 循环向前搜索指定天数
        for day_offset in range(SEARCH_DAYS_LIMIT):
            current_date = start_date - timedelta(days=day_offset)
            current_prefix = f"{base_prefix}/{current_date.strftime('%Y/%m/%d')}"
            
            print(f"  -> 正在搜索路径: '{bucket_name}/{current_prefix}'...")
            
            # 从当前路径获取对象并添加到总列表
            daily_objects = client.list_objects(bucket_name, prefix=current_prefix, recursive=True)
            daily_csv_objects = [obj for obj in daily_objects if obj.object_name.lower().endswith('.json')]
            
            if daily_csv_objects:
                print(f"     ✓ 找到 {len(daily_csv_objects)} 个json文件。")
                all_csv_objects.extend(daily_csv_objects)
                if len(daily_csv_objects) >= config.WINDOW_SIZE * 2:
                    break
            else:
                print(f"     - 未找到任何json文件。")

        if not all_csv_objects:
            print(f"\n✗ 错误: 在过去 {SEARCH_DAYS_LIMIT} 天的路径下未找到任何 .json 文件。")
            return None

        print("\n对所有找到的文件按时间进行排序...")
        key_func = lambda obj: os.path.splitext(os.path.basename(obj.object_name))[0].split('_')[3]
        files_sorted = sorted(all_csv_objects, key=key_func, reverse=True)
        print(f"总共找到并排序 {len(files_sorted)} 个文件。")

    except Exception as e:
        print(f"✗ 错误: 访问 MinIO 或解析路径失败。错误信息: {e}")
        return None
    print("="*60 + "\n")

    print(f"正在寻找一个包含 {config.WINDOW_SIZE} 分钟的、连续且完整的数据窗口...")
    print(f"(每分钟需要有2个文件才算完整，总计 {config.WINDOW_SIZE * 2} 个文件)")

    # 1. 按分钟对所有文件进行分组
    files_by_minute = defaultdict(list)
    for obj in files_sorted:
        try:
            minute_key = os.path.splitext(os.path.basename(obj.object_name))[0].split('_')[3][:12]
            files_by_minute[minute_key].append(obj)
        except IndexError:
            print(f"警告: 文件名 '{obj.object_name}' 格式不正确，已跳过。")
            continue

    # 2. 获取所有存在数据的分钟，并按时间倒序排列
    sorted_minutes = sorted(files_by_minute.keys(), reverse=True)

    # 3. 检查是否有足够的数据分钟来形成一个窗口
    if len(sorted_minutes) < config.WINDOW_SIZE:
        print(f"✗ 错误: 数据分钟总数不足。")
        print(f"  需要至少 {config.WINDOW_SIZE} 个不同的分钟数据，但只找到了 {len(sorted_minutes)} 个。")
        return None

    # 4. 从最新的分钟开始，滑动检查窗口的有效性
    valid_window_minutes = None
    for i in range(len(sorted_minutes) - config.WINDOW_SIZE + 1):
        candidate_window = sorted_minutes[i : i + config.WINDOW_SIZE]
        is_window_valid = all(len(files_by_minute[minute_key]) == 2 for minute_key in candidate_window)
        
        if is_window_valid:
            valid_window_minutes = candidate_window
            print(f"\n✓ 成功找到一个有效的连续数据窗口。")
            print(f"  窗口时间范围: 从 {valid_window_minutes[-1]} 到 {valid_window_minutes[0]}")
            break

    # 5. 如果循环结束仍未找到有效窗口
    if not valid_window_minutes:
        print(f"\n✗ 错误: 在搜索了 {len(all_csv_objects)} 个文件后，")
        print(f"  仍未能找到任何一个包含 {config.WINDOW_SIZE} 分钟的、数据完整的连续时间窗口。")
        return None

    # 6. 按正确的时间顺序构建最终的文件列表
    latest_files = []
    for minute_key in reversed(valid_window_minutes):
        files_in_minute = sorted(files_by_minute[minute_key], key=lambda obj: obj.object_name)
        latest_files.extend(files_in_minute)


    features_list = []
    
    file_iterator = iter(latest_files)

    for obj1, obj2 in zip(file_iterator, file_iterator):
        response1, response2 = None, None
        try:
            minute_ts = os.path.splitext(os.path.basename(obj1.object_name))[0].split('_')[3]
            response1 = client.get_object(bucket_name, obj1.object_name)
            content1 = response1.read()

            if not content1:
                continue

            json_data1 = json.loads(content1)
            df1 = pd.DataFrame({
                'x': json_data1['dataX'],
                'z': json_data1['dataZ']
            })

            df1 = df1.head(16384)
            response2 = client.get_object(bucket_name, obj2.object_name)
            content2 = response2.read()

            if not content2:
                continue
            json_data2 = json.loads(content2)

            df2 = pd.DataFrame({
                'x': json_data2['dataX'],
                'z': json_data2['dataZ']
            })
            df2 = df2.head(16384)

            # ignore_index=True 会重置合并后DataFrame的索引
            merged_df = pd.concat([df1, df2], ignore_index=True)


            fv = process_and_extract_features_from_dataframe(merged_df, fs=config.FS)
            
            if fv is not None:
                features_list.append(fv)
            else:
                 print("   ✗ 特征提取失败或返回空值。")

        except Exception as e:
            print(f"  ✗ 错误: 处理文件对 '{obj1.object_name}' 和 '{obj2.object_name}' 时发生错误，已跳过。")
            print(f"    错误信息: {e}")
            continue

        finally:
            # 确保两个连接都被安全关闭
            if response1:
                response1.close()
                response1.release_conn()
            if response2:
                response2.close()
                response2.release_conn()

    if len(features_list) < config.WINDOW_SIZE:
        print(f"\n错误: 有效特征组数量 ({len(features_list)}) 不足所需的 {config.WINDOW_SIZE} 个。")
        print("可能是因为部分文件对处理失败。无法进行预测。")
        return None
        
    features = np.array(features_list)

    if len(features_list) < config.WINDOW_SIZE:
        print(f"错误: 有效特征不足 {config.WINDOW_SIZE} 个")
        return None
        
    features = np.array(features_list)
    features_scaled = scaler_X.transform(features)
    input_data = np.expand_dims(features_scaled, axis=0)
    
    print("正在预测RUL...")
    prediction_array = model.predict(input_data, verbose=0)
    
    log_rul_pred = prediction_array[0][0]
    predicted_rul = np.maximum(0, np.expm1(log_rul_pred))
    
    print("\n" + "="*60)
    print("RUL 预测结果 (来自 MinIO)")
    print("="*60)
    print(f"预测剩余寿命 (周期数): {predicted_rul:.2f}")
    print("="*60)
    
    health_threshold = config.RUL_CAP * 0.5
    if predicted_rul > health_threshold:
        status = "健康"
        color = "✓"
    elif predicted_rul > health_threshold * 0.5:
        status = "轻微退化"
        color = "⚠"
    else:
        status = "严重退化，建议维护"
        color = "✗"
    
    print(f"\n{color} 轴承健康状态: {status}")
    print("="*60 + "\n")
    
    return {
        'rul_cycles': predicted_rul,
        'status': status
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'train':
            train_model()
            
        elif command == 'predict':
            if len(sys.argv) > 2:
                data_folder = sys.argv[2]
                if os.path.isdir(data_folder):
                    predict_rul(data_folder)
                else:
                    print(f"错误: 提供的路径 '{data_folder}' 不是一个有效的文件夹。")
            else:
                print("使用方法: python your_script_name.py predict <path_to_data_folder>")
                
        # 新增的命令处理
        elif command == 'predict_minio':
            if len(sys.argv) == 4:
                bucket_name = sys.argv[2]
                prefix = sys.argv[3]
                predict_rul_from_minio(bucket_name, prefix)
            else:
                print("使用方法: python your_script_name.py predict_minio <bucket_name> <path/to/data/prefix>")
                print("示例: python your_script_name.py predict_minio iiot-test VtSensorData/v_s_d_1960892330983149570/2025/10/14")
        
        else:
            print(f"未知命令: '{command}'")
            print("可用命令: train, predict, predict_minio")
    else:
        # 更新帮助信息
        print("="*70)
        print("轴承RUL预测系统 v4.1 (已集成MinIO)")
        print("="*70)
        print("\n使用方法:")
        print("  训练模型:             python your_script_name.py train")
        print("  本地预测:             python your_script_name.py predict <local_folder_path>")
        print("  MinIO预测:            python your_script_name.py predict_minio <bucket> <prefix>")
        print("\nMinIO 预测示例:")
        print("  python your_script_name.py predict_minio iiot-test VtSensorData/v_s_d_1960892330983149570/2025/10/14")
        print("\n重要: 运行MinIO预测前，请先设置环境变量:")
        print("  export MINIO_ENDPOINT='your_server:port'")
        print("  export MINIO_ACCESS_KEY='your_access_key'")
        print("  export MINIO_SECRET_KEY='your_secret_key'")
        print("="*70)