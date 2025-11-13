import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import warnings
import gradio as gr
import sys
import logging
from pathlib import Path
import io
from PIL import Image
import os
from taosrest import connect
from sklearn.preprocessing import StandardScaler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.app_utils import AppUtils as util
from utils.app_utils import MultiDirectoryMonitor
warnings.filterwarnings('ignore')
plt = util.auto_config_chinese_font()

# ============================================================
# 配置路径
# ============================================================
BASE_DIR = Path(__file__).parent.parent / "model" / "air_compressor"
MODEL_BASE_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
EXAMPLE_DIR = BASE_DIR / "examples"
RESTART_SIGNAL_FILENAME = ".restart_signal_air_compressor"

# ============================================================
# 数据库配置
# ============================================================
DB_CONFIG = {
    "url": "http://192.168.37.160:6041",
    "user": "iot_admin",
    "password": "qihang123.",
    "database": "iot_admin_test"
}

# ============================================================
# 故障定义
# ============================================================
fault_code_map = {
    1: '主机过载',
    2: '主机不平衡',
    3: '风机过载',
    4: '排气温度高',
    5: '供气压力高',
    6: '电压过低',
    7: '电压过高',
    8: '电机过载',
    9: '风机过载',
}

# 故障点描述和建议
fault_point_analysis = {
    # ----------------------- 1: 主机过载 (Diagnosis) -----------------------
    1: {
        'name': '主机过载',
        'fault_points': [
            '主电机输出电流异常升高',
            '主电机输出功率超负荷运行',
            '主电机总功耗差值增大',
            '排气温度上升',
            '主电机转速略有下降'
        ],
        'root_causes': [
            '负载超过设备额定功率',
            '压缩机内部磨损',
            '润滑系统异常',
            '进气过滤器堵塞'
        ],
        'suggestions': [
            '检查当前负载是否超过设备额定值',
            '检查压缩机内部部件磨损情况',
            '检查润滑油位和油质',
            '清洁或更换进气过滤器'
        ]
    },
    # ----------------------- 101: 主机过载 (Warning) -----------------------
    101: {
        'name': '主机过载 - 预警',
        'fault_points': [
            '主电机输出电流持续升高',
            '主电机输出功率开始增加',
            '主电机总功耗差值开始增大',
            '排气温度略微上升',
            '主电机转速轻微波动'
        ],
        'root_causes': [
            '负载超过设备额定功率',
            '压缩机内部磨损',
            '润滑系统异常',
            '进气过滤器堵塞'
        ],
        'suggestions': [
            '检查当前负载是否超过设备额定值',
            '检查压缩机内部部件磨损情况',
            '检查润滑油位和油质',
            '清洁或更换进气过滤器'
        ]
    },
    # ----------------------- 2: 主机不平衡 (Diagnosis) -----------------------
    2: {
        'name': '主机不平衡',
        'fault_points': [
            '三相电流平衡度异常',
            '主电机输出电流波动',
            '主电机输出电压不稳定',
            '功耗差值异常',
            '排气温度轻微升高'
        ],
        'root_causes': [
            '三相电源不平衡',
            '电机绕组故障',
            '电机轴承磨损',
            '负载分布不均'
        ],
        'suggestions': [
            '检查三相电源电压平衡度',
            '检查电机绕组绝缘和阻值',
            '检查轴承状态和润滑',
            '调整负载分布'
        ]
    },
    # ----------------------- 102: 主机不平衡 (Warning) -----------------------
    102: {
        'name': '主机不平衡 - 预警',
        'fault_points': [
            '三相电流平衡度轻微异常',
            '主电机输出电流开始波动',
            '主电机输出电压轻微波动',
            '功耗差值开始变化',
            '排气温度保持基准或轻微波动'
        ],
        'root_causes': [
            '三相电源不平衡',
            '电机绕组故障',
            '电机轴承磨损',
            '负载分布不均'
        ],
        'suggestions': [
            '检查三相电源电压平衡度',
            '检查电机绕组绝缘和阻值',
            '检查轴承状态和润滑',
            '调整负载分布'
        ]
    },
    # ----------------------- 3: 风机过载 (Diagnosis) -----------------------
    3: {
        'name': '风机过载',
        'fault_points': [
            '风机电机输出电流显著升高',
            '风机电机输出功率超标',
            '风机总功耗差值增大',
            '风机转速降低',
            '排气温度明显升高'
        ],
        'root_causes': [
            '散热风扇叶片损坏或积灰',
            '风机轴承磨损',
            '风道堵塞',
            '环境温度过高'
        ],
        'suggestions': [
            '清洁风扇叶片和散热器',
            '检查风机轴承状态',
            '清理风道障碍物',
            '改善环境通风条件'
        ]
    },
    # ----------------------- 103: 风机过载 (Warning) -----------------------
    103: {
        'name': '风机过载 - 预警',
        'fault_points': [
            '风机电机输出电流升高',
            '风机电机输出功率增加',
            '风机总功耗差值开始增大',
            '风机转速略有降低',
            '排气温度开始上升'
        ],
        'root_causes': [
            '散热风扇叶片损坏或积灰',
            '风机轴承磨损',
            '风道堵塞',
            '环境温度过高'
        ],
        'suggestions': [
            '清洁风扇叶片和散热器',
            '检查风机轴承状态',
            '清理风道障碍物',
            '改善环境通风条件'
        ]
    },
    # ----------------------- 4: 排气温度高 (Diagnosis) -----------------------
    4: {
        'name': '排气温度高',
        'fault_points': [
            '排气温度持续升高',
            '主电机输出电流波动',
            '主电机功率增加',
            '风机转速显著升高',
            '供气压力变化'
        ],
        'root_causes': [
            '冷却系统效率下降',
            '环境温度过高',
            '压缩比过大',
            '润滑油冷却效果差'
        ],
        'suggestions': [
            '检查冷却器清洁度和效率',
            '改善环境通风和温度',
            '调整压缩机工作压力',
            '检查润滑油温度和流量'
        ]
    },
    # ----------------------- 104: 排气温度高 (Warning) -----------------------
    104: {
        'name': '排气温度高 - 预警',
        'fault_points': [
            '排气温度开始上升',
            '主电机输出电流轻微波动',
            '主电机功率轻微增加',
            '风机转速升高',
            '供气压力轻微波动'
        ],
        'root_causes': [
            '冷却系统效率下降',
            '环境温度过高',
            '压缩比过大',
            '润滑油冷却效果差'
        ],
        'suggestions': [
            '检查冷却器清洁度和效率',
            '改善环境通风和温度',
            '调整压缩机工作压力',
            '检查润滑油温度和流量'
        ]
    },
    # ----------------------- 5: 供气压力高 (Diagnosis) -----------------------
    5: {
        'name': '供气压力高',
        'fault_points': [
            '供气压力超出正常范围',
            '主电机输出功率增加',
            '主电机输出电流升高',
            '排气温度升高'
        ],
        'root_causes': [
            '用气量减少导致压力上升',
            '压力控制器设定值过高',
            '卸载阀故障',
            '管路阻力增大'
        ],
        'suggestions': [
            '检查用气量和压力控制器设定',
            '校准压力传感器',
            '检查卸载阀动作是否正常',
            '检查管路是否有堵塞'
        ]
    },
    # ----------------------- 105: 供气压力高 (Warning) -----------------------
    105: {
        'name': '供气压力高 - 预警',
        'fault_points': [
            '供气压力略微上升',
            '主电机输出功率轻微增加',
            '主电机输出电流轻微升高',
            '排气温度轻微上升'
        ],
        'root_causes': [
            '用气量减少导致压力上升',
            '压力控制器设定值过高',
            '卸载阀故障',
            '管路阻力增大'
        ],
        'suggestions': [
            '检查用气量和压力控制器设定',
            '校准压力传感器',
            '检查卸载阀动作是否正常',
            '检查管路是否有堵塞'
        ]
    },
    # ----------------------- 6: 电压过低 (Diagnosis) -----------------------
    6: {
        'name': '电压过低',
        'fault_points': [
            '输入电压低于正常范围',
            '主电机输出电压降低',
            '风机输出电压降低',
            '主电机电流升高',
            '主电机转速下降'
        ],
        'root_causes': [
            '电网电压波动',
            '供电线路压降过大',
            '变压器容量不足',
            '电缆截面积过小'
        ],
        'suggestions': [
            '检查电网供电质量',
            '检查供电线路和接头',
            '评估变压器容量是否足够',
            '检查电缆规格是否匹配'
        ]
    },
    # ----------------------- 106: 电压过低 (Warning) -----------------------
    106: {
        'name': '电压过低 - 预警',
        'fault_points': [
            '输入电压略微下降',
            '主电机输出电压轻微降低',
            '风机输出电压轻微降低',
            '主电机电流略微升高',
            '主电机转速轻微下降'
        ],
        'root_causes': [
            '电网电压波动',
            '供电线路压降过大',
            '变压器容量不足',
            '电缆截面积过小'
        ],
        'suggestions': [
            '检查电网供电质量',
            '检查供电线路和接头',
            '评估变压器容量是否足够',
            '检查电缆规格是否匹配'
        ]
    },
    # ----------------------- 7: 电压过高 (Diagnosis) -----------------------
    7: {
        'name': '电压过高',
        'fault_points': [
            '输入电压高于正常范围',
            '主电机输出电压升高',
            '风机输出电压升高',
            '主电机电流降低',
            '主电机转速略微升高'
        ],
        'root_causes': [
            '电网电压调节不当',
            '变压器分接开关位置不对',
            '无功补偿过度',
            '轻载时电压上升'
        ],
        'suggestions': [
            '联系供电部门调整电压',
            '调整变压器分接开关',
            '检查无功补偿装置',
            '安装稳压装置'
        ]
    },
    # ----------------------- 107: 电压过高 (Warning) -----------------------
    107: {
        'name': '电压过高 - 预警',
        'fault_points': [
            '输入电压略微升高',
            '主电机输出电压轻微升高',
            '风机输出电压轻微升高',
            '主电机电流略微降低',
            '主电机转速轻微波动'
        ],
        'root_causes': [
            '电网电压调节不当',
            '变压器分接开关位置不对',
            '无功补偿过度',
            '轻载时电压上升'
        ],
        'suggestions': [
            '联系供电部门调整电压',
            '调整变压器分接开关',
            '检查无功补偿装置',
            '安装稳压装置'
        ]
    }
}

fault_definitions = {
    1: {
        'name': '主机过载',
        'duration_range': (30, 120),
        'affected_params': {
            'main_motor_output_current': {'baseline_shift': 1.8, 'volatility': 0.15, 'trend': 0.02},
            'main_motor_output_power': {'baseline_shift': 1.75, 'volatility': 0.12, 'trend': 0.015},
            'main_motor_total_power_consumption_diff': {'baseline_shift': 1.9, 'volatility': 0.2, 'trend': 0.03},
            'exhaust_temperature': {'baseline_shift': 1.25, 'volatility': 0.1, 'trend': 0.01},
            'main_motor_speed': {'baseline_shift': 0.98, 'volatility': 0.05, 'trend': -0.005}
        }
    },
    2: {
        'name': '主机不平衡',
        'duration_range': (20, 90),
        'affected_params': {
            'current_balance_degree': {'baseline_shift': 1.7, 'volatility': 0.3, 'trend': -0.05},
            'main_motor_output_current': {'baseline_shift': 0.7, 'volatility': 0.25, 'trend': 0.0, 'oscillate': 0.2},
            'main_motor_output_voltage': {'baseline_shift': 0.9, 'volatility': 0.2, 'trend': 0.0, 'oscillate': 0.1},
            'main_motor_total_power_consumption_diff': {'baseline_shift': 0.85, 'volatility': 0.2, 'trend': 0.01},
            'exhaust_temperature': {'baseline_shift': 0.8, 'volatility': 0.15, 'trend': 0.005},
        }
    },
    3: {
        'name': '风机过载',
        'duration_range': (25, 100),
        'affected_params': {
            'fan_motor_output_current': {'baseline_shift': 1.7, 'volatility': 0.2, 'trend': 0.015},
            'fan_motor_output_power': {'baseline_shift': 1.6, 'volatility': 0.18, 'trend': 0.015},
            'fan_motor_total_power_consumption_diff': {'baseline_shift': 1.8, 'volatility': 0.2, 'trend': 0.02},
            'fan_motor_speed': {'baseline_shift': 0.6, 'volatility': 0.08, 'trend': -0.01},
            'exhaust_temperature': {'baseline_shift': 1.35, 'volatility': 0.15, 'trend': 0.012},
        }
    },
    4: {
        'name': '排气温度高',
        'duration_range': (40, 150),
        'affected_params': {
            'main_motor_output_current': {'baseline_shift': 1.2, 'volatility': 0.25, 'trend': 0.0, 'oscillate': 0.2},
            'main_motor_output_power': {'baseline_shift': 1.2, 'volatility': 0.08, 'trend': 0.001},
            'supply_pressure': {'baseline_shift': 1.0, 'volatility': 0.2, 'trend': 0},
            'fan_motor_speed': {'baseline_shift': 1.6, 'volatility': 0.1, 'trend': 0.002},
            'exhaust_temperature': {'baseline_shift': 1.4, 'volatility': 0.12, 'trend': 0.005},
            'main_motor_total_power_consumption_diff': {'baseline_shift': 1.54, 'volatility': 0.2, 'trend': 0.03},
        }
    },
    5: {
        'name': '供气压力高',
        'duration_range': (30, 120),
        'affected_params': {
            'supply_pressure': {'baseline_shift': 1.3, 'volatility': 0.1, 'trend': 0.002},
            'main_motor_output_power': {'baseline_shift': 1.25, 'volatility': 0.1, 'trend': 0.0015},
            'main_motor_output_current': {'baseline_shift': 1.2, 'volatility': 0.1, 'trend': 0.0015},
            'exhaust_temperature': {'baseline_shift': 1.15, 'volatility': 0.08, 'trend': 0.001}
        }
    },
    6: {
        'name': '电压过低',
        'duration_range': (15, 80),
        'affected_params': {
            'voltage': {'baseline_shift': 0.85, 'volatility': 0.06, 'trend': -0.0005},
            'main_motor_output_voltage': {'baseline_shift': 0.87, 'volatility': 0.06, 'trend': -0.0005},
            'fan_motor_output_voltage': {'baseline_shift': 0.88, 'volatility': 0.06, 'trend': -0.0005},
            'main_motor_output_current': {'baseline_shift': 1.2, 'volatility': 0.1, 'trend': 0.001},
            'fan_motor_output_current': {'baseline_shift': 1.18, 'volatility': 0.1, 'trend': 0.001},
            'main_motor_speed': {'baseline_shift': 0.97, 'volatility': 0.05, 'trend': -0.001},
            'main_motor_output_power': {'baseline_shift': 0.95, 'volatility': 0.1, 'trend': -0.001},
        }
    },
    7: {
        'name': '电压过高',
        'duration_range': (15, 80),
        'affected_params': {
            'voltage': {'baseline_shift': 1.12, 'volatility': 0.06, 'trend': 0.0005},
            'main_motor_output_voltage': {'baseline_shift': 1.1, 'volatility': 0.06, 'trend': 0.0005},
            'fan_motor_output_voltage': {'baseline_shift': 1.1, 'volatility': 0.06, 'trend': 0.0005},
            'main_motor_output_current': {'baseline_shift': 0.9, 'volatility': 0.08, 'trend': -0.0005},
            'fan_motor_output_current': {'baseline_shift': 0.92, 'volatility': 0.08, 'trend': -0.0005},
            'main_motor_speed': {'baseline_shift': 1.02, 'volatility': 0.04, 'trend': 0.0001},
            'exhaust_temperature': {'baseline_shift': 1.03, 'volatility': 0.1, 'trend': 0.0002}
        }
    }
}

def fetch_normal_data_from_db(start_time, end_time):
    """从数据库获取正常数据"""
    try:
        logging.info(f"开始连接数据库...")
        conn = connect(
            url=DB_CONFIG["url"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"]
        )
        logging.info("✓ 数据库连接成功！")
        
        query_sql = f"""
        SELECT
        `_ts` as `timestamp`,
        `d01` as `supply_pressure`,
        `d02` as `exhaust_temperature`,
        `d05` as `main_motor_current_a`,
        `d06` as `main_motor_current_b`,
        `d07` as `main_motor_current_c`,
        `d25` as `voltage`,
        `d33` as `main_motor_output_voltage`,
        `d34` as `main_motor_output_current`,
        `d35` as `main_motor_output_frequency`,
        `d36` as `main_motor_output_power`,
        `d37` as `fan_motor_output_voltage`,
        `d38` as `fan_motor_output_current`,
        `d39` as `fan_motor_output_frequency`,
        `d40` as `fan_motor_output_power`,
        `d41` as `main_motor_speed`,
        `d42` as `fan_motor_speed`,
        `d78` as `main_motor_total_power_consumption_h`,
        `d79` as `main_motor_total_power_consumption_l`,
        `d80` as `fan_motor_total_power_consumption_h`,
        `d81` as `fan_motor_total_power_consumption_l`
        FROM
        `iot_admin_test`.`iot_device_product_property_kongyaji_yxcs` 
        WHERE `_ts` >= '{start_time}' AND `_ts` < '{end_time}'
        """
        
        logging.info(f"执行查询: {start_time} 到 {end_time}")
        result = conn.query(query_sql)
        
        if result.field_count > 0:
            columns = [field['name'] for field in result.fields]
            raw_data = []
            for row in result:
                raw_data.append(row)
            
            df_raw = pd.DataFrame(raw_data, columns=columns)
            logging.info(f"✓ 查询成功，原始数据: {df_raw.shape[0]} 行")
            
            df_raw['main_motor_total_power_consumption'] = (
                df_raw['main_motor_total_power_consumption_h'] * 65536 / 100 + 
                df_raw['main_motor_total_power_consumption_l'] / 100
            )
            df_raw['fan_motor_total_power_consumption'] = (
                df_raw['fan_motor_total_power_consumption_h'] * 65536 / 100 + 
                df_raw['fan_motor_total_power_consumption_l'] / 100
            )
            
            df_raw = df_raw.drop(columns=[
                'main_motor_total_power_consumption_h',
                'main_motor_total_power_consumption_l',
                'fan_motor_total_power_consumption_h',
                'fan_motor_total_power_consumption_l'
            ])
            
            current_cols = ['main_motor_current_a', 'main_motor_current_b', 'main_motor_current_c']
            df_raw['current_max'] = df_raw[current_cols].max(axis=1)
            df_raw['current_min'] = df_raw[current_cols].min(axis=1)
            df_raw['current_balance_degree'] = (
                (df_raw['current_max'] / df_raw['current_min']) / (1 + (14/10))
            )
            
            df_raw = df_raw.drop(columns=[
                'current_max', 'current_min',
                'main_motor_current_a', 'main_motor_current_b', 'main_motor_current_c'
            ])
            
            df_clean = df_raw.dropna()
            
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
            df_clean = df_clean.set_index('timestamp')
            df_clean = df_clean.sort_index()
            df_clean = df_clean.resample('5min').mean()
            
            df_clean['main_motor_total_power_consumption_diff'] = df_clean['main_motor_total_power_consumption'].diff()
            df_clean['fan_motor_total_power_consumption_diff'] = df_clean['fan_motor_total_power_consumption'].diff()
            
            df_clean = df_clean.drop(columns=[
                'main_motor_total_power_consumption',
                'fan_motor_total_power_consumption'
            ])
            
            df_clean = df_clean.dropna()
            df_clean = df_clean.reset_index()
            
            conn.close()
            logging.info(f"✓ 数据处理完成，最终数据: {df_clean.shape[0]} 行")
            
            return df_clean, None
            
        else:
            conn.close()
            return None, "查询返回空结果"
            
    except Exception as e:
        error_msg = f"数据库查询失败: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return None, error_msg

def generate_fault_data(normal_df, fault_code, fault_definitions, 
                       num_pre_warning_samples=None, num_fault_samples=None,
                       pre_warning_severity=0.5, generate_pre_warning=True):
    """生成故障数据"""
    if fault_code not in fault_definitions:
        raise ValueError(f"未知的故障代码: {fault_code}")
    
    seed = int(datetime.now().timestamp())
    fault_info = fault_definitions[fault_code]
    
    total_samples = len(normal_df)
    if num_pre_warning_samples is None:
        num_pre_warning_samples = int(total_samples * 0.3)
    if num_fault_samples is None:
        num_fault_samples = total_samples - num_pre_warning_samples
    
    all_data_dfs = []
    
    if generate_pre_warning and num_pre_warning_samples > 0:
        df_pre_warning = normal_df.sample(n=num_pre_warning_samples, random_state=seed + fault_code).copy()
        
        for param, effects in fault_info['affected_params'].items():
            if param not in df_pre_warning.columns:
                continue
            
            s = pre_warning_severity
            pre_warning_effects = {
                'baseline_shift': 1 + (effects.get('baseline_shift', 1) - 1) * s,
                'volatility': effects.get('volatility', 0) * s,
                'trend': effects.get('trend', 0) * s,
                'oscillate': effects.get('oscillate', 0) * s,
            }
            
            original_series = df_pre_warning[param].copy()
            modified_series = original_series.copy()
            
            modified_series *= pre_warning_effects['baseline_shift']
            
            if pre_warning_effects['volatility'] > 0:
                noise = np.random.normal(0, original_series.std() * pre_warning_effects['volatility'], 
                                        size=len(modified_series))
                modified_series += noise
            
            if pre_warning_effects['trend'] != 0:
                trend_effect = np.linspace(0, pre_warning_effects['trend'], 
                                          len(modified_series)) * original_series.mean()
                modified_series += trend_effect
            
            if pre_warning_effects['oscillate'] > 0:
                oscillation = np.sin(np.linspace(0, 10*np.pi, len(modified_series))) * \
                             original_series.mean() * pre_warning_effects['oscillate']
                modified_series += oscillation
            
            df_pre_warning[param] = modified_series
        
        df_pre_warning['fault_code'] = 100 + fault_code
        df_pre_warning['fault_name'] = f"{fault_info['name']}-预警"
        all_data_dfs.append(df_pre_warning)
    
    if num_fault_samples > 0:
        df_fault = normal_df.sample(n=num_fault_samples, random_state=seed + fault_code + 1000).copy()
        
        for param, effects in fault_info['affected_params'].items():
            if param not in df_fault.columns:
                continue
            
            original_series = df_fault[param].copy()
            modified_series = original_series.copy()
            
            modified_series *= effects.get('baseline_shift', 1)
            
            if 'volatility' in effects:
                noise = np.random.normal(0, original_series.std() * effects['volatility'], 
                                        size=len(modified_series))
                modified_series += noise
            
            if 'trend' in effects:
                trend_effect = np.linspace(0, effects['trend'], 
                                          len(modified_series)) * original_series.mean()
                modified_series += trend_effect
            
            if 'oscillate' in effects:
                oscillation = np.sin(np.linspace(0, 10*np.pi, len(modified_series))) * \
                             original_series.mean() * effects['oscillate']
                modified_series += oscillation
            
            df_fault[param] = modified_series
        
        df_fault['fault_code'] = fault_code
        df_fault['fault_name'] = fault_info['name']
        all_data_dfs.append(df_fault)
    
    final_df = pd.concat(all_data_dfs, ignore_index=True)
    return final_df

class FaultClassifierPipeline:
    """内置标准化的分类器Pipeline"""
    
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler if scaler else StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y, **kwargs):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, **kwargs)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        return None

class FaultClassifierInference:
    """故障分类模型推理类"""
    
    def __init__(self, model_path, metadata_path):
        self.model = joblib.load(model_path)
        print(f"✓ 模型加载成功: {model_path}")
        
        metadata = joblib.load(metadata_path)
        self.label_mapping = metadata['label_mapping']
        self.reverse_mapping = metadata['reverse_mapping']
        self.feature_names = metadata['feature_names']
        self.model_name = metadata['model_name']
        self.test_f1 = metadata.get('test_f1', 'N/A')
    
    def prepare_features(self, df):
        """准备特征"""
        exclude_cols = ['fault_code', 'fault_name', 'timestamp']
        sensor_cols = [col for col in df.columns 
                      if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        print(f"  使用原始传感器特征: {len(sensor_cols)} 个")
        
        missing_features = set(self.feature_names) - set(sensor_cols)
        if missing_features:
            print(f"  警告: 缺失 {len(missing_features)} 个特征，用0填充")
            for feat in missing_features:
                df[feat] = 0
        
        X = df[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X
    
    def predict_batch(self, samples, return_proba=False):
        """批量预测"""
        if isinstance(samples, pd.DataFrame):
            X = self.prepare_features(samples)
        else:
            X = np.array(samples)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        y_pred = self.model.predict(X)
        fault_codes = [self.reverse_mapping[pred] for pred in y_pred]
        
        results = pd.DataFrame({
            'fault_code': fault_codes,
            'encoded_label': y_pred
        })
        
        if return_proba:
            try:
                proba = self.model.predict_proba(X)
                results['confidence'] = proba.max(axis=1)
                
                num_model_classes = proba.shape[1]
                for i in range(num_model_classes):
                    label = self.reverse_mapping.get(i, f'未知类别_{i}')
                    results[f'prob_{label}'] = proba[:, i]
            except Exception as e:
                print(f"  警告：无法获取概率。错误: {e}")
                results['confidence'] = None
        
        return results

def generate_fault_report(df, predictions):
    """生成故障分析报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("故障诊断分析报告")
    report_lines.append("=" * 80)
    report_lines.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"样本总数: {len(df)}")
    report_lines.append("")
    
    # 统计各故障类型
    fault_counts = predictions['fault_code'].value_counts().sort_index()
    
    report_lines.append("【故障分布统计】")
    report_lines.append("-" * 80)
    for fault_code, count in fault_counts.items():
        percentage = count / len(predictions) * 100
        fault_name = fault_point_analysis.get(fault_code, {}).get('name', f'未知故障{fault_code}')
        report_lines.append(f"  故障代码 {fault_code} - {fault_name}: {count}次 ({percentage:.2f}%)")
    report_lines.append("")
    
    # 详细故障点分析
    for fault_code in fault_counts.index:
        if fault_code not in fault_point_analysis:
            continue
            
        analysis = fault_point_analysis[fault_code]
        count = fault_counts[fault_code]
        percentage = count / len(predictions) * 100
        
        report_lines.append("=" * 80)
        report_lines.append(f"【故障类型 {fault_code}】{analysis['name']}")
        report_lines.append("=" * 80)
        report_lines.append(f"检出次数: {count}次 ({percentage:.2f}%)")
        report_lines.append("")
        
        report_lines.append("► 关键故障点:")
        for i, point in enumerate(analysis['fault_points'], 1):
            report_lines.append(f"  {i}. {point}")
        report_lines.append("")
        
        report_lines.append("► 可能原因:")
        for i, cause in enumerate(analysis['root_causes'], 1):
            report_lines.append(f"  {i}. {cause}")
        report_lines.append("")
        
        report_lines.append("► 处理建议:")
        for i, suggestion in enumerate(analysis['suggestions'], 1):
            report_lines.append(f"  {i}. {suggestion}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("报告结束")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)

def create_fault_distribution_chart(predictions):
    """创建故障分布图表"""
    fault_counts = predictions['fault_code'].value_counts().sort_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 柱状图
    fault_names = [fault_point_analysis.get(code, {}).get('name', f'故障{code}') 
                   for code in fault_counts.index]
    colors = plt.cm.Set3(range(len(fault_counts)))
    
    bars = ax1.bar(range(len(fault_counts)), fault_counts.values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(fault_counts)))
    ax1.set_xticklabels([f'{name}\n(代码{code})' for name, code in zip(fault_names, fault_counts.index)], 
                        rotation=45, ha='right')
    ax1.set_ylabel('检出次数', fontsize=12, fontweight='bold')
    ax1.set_title('故障类型检出次数分布', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, count in zip(bars, fault_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 饼图
    wedges, texts, autotexts = ax2.pie(fault_counts.values, 
                                        labels=[f'{name}\n({count}次)' for name, count in zip(fault_names, fault_counts.values)],
                                        autopct='%1.1f%%',
                                        colors=colors,
                                        explode=[0.05] * len(fault_counts),
                                        shadow=True,
                                        startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    for text in texts:
        text.set_fontsize(9)
    
    ax2.set_title('故障类型占比分布', fontsize=14, fontweight='bold')
    
    plt.suptitle('故障诊断结果分布分析', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img

classifier = None
model_options = {}
simulated_files = {}

def get_simulated_files():
    """获取所有模拟生成的文件"""
    global simulated_files
    simulated_files = {}
    
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return simulated_files
    
    sim_files = list(DATA_DIR.glob("simulated_fault_*.csv"))
    
    for file_path in sorted(sim_files, key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            filename = file_path.stem
            parts = filename.split('_')
            if len(parts) >= 4:
                fault_code = int(parts[2])
                timestamp = '_'.join(parts[3:])
                
                fault_name = fault_definitions.get(fault_code, {}).get('name', f'未知故障{fault_code}')
                
                try:
                    dt = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                    time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    time_str = timestamp
                
                display_name = f"{fault_name} ({time_str})"
                simulated_files[display_name] = str(file_path)
        except Exception as e:
            logging.warning(f"解析文件名失败: {file_path.name}, 错误: {e}")
            continue
    
    logging.info(f"找到 {len(simulated_files)} 个模拟文件")
    return simulated_files

def initialize_models():
    """初始化模型选项"""
    global model_options
    logging.info(f"发现模型目录: {MODEL_BASE_DIR}")
    if MODEL_BASE_DIR.exists():
        model_files = list(MODEL_BASE_DIR.glob("fault_model_*.pkl"))
        model_options = {f.stem: str(f) for f in model_files}
    logging.info(f"发现 {len(model_options)} 个模型文件")

def load_model(model_name):
    """加载指定模型"""
    global classifier
    try:
        model_path = model_options[model_name]
        metadata_path = model_path.replace('fault_model_', 'model_metadata_')
        
        classifier = FaultClassifierInference(model_path, metadata_path)
        return f"✅ 模型加载成功: {model_name}\n模型类型: {classifier.model_name}\nF1分数: {classifier.test_f1}"
    except Exception as e:
        return f"❌ 模型加载失败: {str(e)}"

def predict_from_csv(csv_file, model_name, simulated_file_name):
    """从CSV文件预测 - 优先使用模拟数据"""
    global classifier
    
    # 优先使用模拟数据
    if simulated_file_name and simulated_file_name != "暂无模拟数据" and simulated_file_name in simulated_files:
        file_path = simulated_files[simulated_file_name]
        data_source = f"模拟数据: {simulated_file_name}"
        logging.info(f"使用模拟文件: {file_path}")
    elif csv_file is not None:
        file_path = csv_file.name
        data_source = "上传文件"
        logging.info(f"使用上传文件: {file_path}")
    else:
        return None, "❌ 请上传CSV文件或选择模拟数据！", None
    
    # 检查模型
    if not model_options or model_name == "无可用模型" or model_name not in model_options:
        return None, "❌ 没有可用的模型！请先训练模型或检查模型文件路径。", None
    
    if classifier is None or model_name not in model_options:
        status = load_model(model_name)
        if "失败" in status:
            return None, status, None
    
    try:
        df = pd.read_csv(file_path)
        logging.info(f"读取数据: {len(df)} 条样本")
        
        predictions = classifier.predict_batch(df, return_proba=True)
        output_df = pd.concat([df, predictions], axis=1)
        
        # 生成故障报告
        report_text = generate_fault_report(df, predictions)
        
        # 生成分布图表
        distribution_chart = create_fault_distribution_chart(predictions)
        
        # 保存预测结果
        output_path = DATA_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        
        # 保存报告
        report_path = DATA_DIR / f"fault_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"数据来源: {data_source}\n\n")
            f.write(report_text)
        
        result_summary = f"📊 故障诊断完成\n\n"
        result_summary += f"数据来源: {data_source}\n"
        result_summary += f"样本总数: {len(df)}\n"
        result_summary += f"预测文件: {output_path}\n"
        result_summary += f"分析报告: {report_path}\n"
        
        return result_summary, report_text, distribution_chart
        
    except Exception as e:
        error_msg = f"❌ 预测过程中出现错误: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return None, error_msg, None

def simulate_fault_from_db(fault_code, num_samples, pre_warning_severity, generate_pre_warning,
                          start_time, end_time, show_visualization):
    """从数据库获取数据并模拟故障"""
    try:
        logging.info(f"从数据库获取数据: {start_time} 到 {end_time}")
        normal_df, error = fetch_normal_data_from_db(start_time, end_time)
        
        if error:
            return None, f"❌ 数据获取失败: {error}", None
        
        if normal_df is None or len(normal_df) == 0:
            return None, "❌ 未获取到有效数据，请检查时间范围", None
        
        logging.info(f"获取到 {len(normal_df)} 条正常数据")
        
        available_samples = len(normal_df)
        if num_samples > available_samples:
            warning_msg = f"⚠️ 警告：请求生成 {num_samples} 条样本，但只有 {available_samples} 条正常数据可用。\n"
            warning_msg += f"将自动调整为生成 {available_samples} 条样本。\n\n"
            num_samples = available_samples
            logging.warning(f"样本数量已调整为 {num_samples}")
        else:
            warning_msg = ""
        
        if generate_pre_warning:
            num_pre_warning = int(num_samples * 0.3)
            num_fault = num_samples - num_pre_warning
            
            total_needed = num_pre_warning + num_fault
            if total_needed > available_samples:
                num_pre_warning = int(available_samples * 0.3)
                num_fault = available_samples - num_pre_warning
                logging.info(f"重新分配样本: 预警={num_pre_warning}, 故障={num_fault}")
        else:
            num_pre_warning = 0
            num_fault = num_samples
        
        fault_df = generate_fault_data(
            normal_df=normal_df,
            fault_code=fault_code,
            fault_definitions=fault_definitions,
            num_pre_warning_samples=num_pre_warning,
            num_fault_samples=num_fault,
            pre_warning_severity=pre_warning_severity,
            generate_pre_warning=generate_pre_warning
        )
        
        result_text = warning_msg + f"📊 故障模拟完成！\n"
        result_text += f"数据来源: 数据库 ({start_time} ~ {end_time})\n"
        result_text += f"原始正常数据: {len(normal_df)} 条\n"
        result_text += f"生成故障样本: {len(fault_df)} 条\n\n"
        result_text += "生成数据分布:\n"
        for code, count in fault_df['fault_code'].value_counts().items():
            fault_name = fault_df[fault_df['fault_code'] == code]['fault_name'].iloc[0]
            percentage = count / len(fault_df) * 100
            result_text += f"  {fault_name} (代码 {code}): {count} 条 ({percentage:.2f}%)\n"
        
        viz_img = None
        if show_visualization:
            exclude_cols = ['fault_code', 'fault_name', 'timestamp']
            feature_names = [col for col in fault_df.columns 
                           if col not in exclude_cols and fault_df[col].dtype in ['int64', 'float64']]
            feature_names = feature_names[:6]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'故障模拟可视化 - {fault_definitions[fault_code]["name"]}', 
                        fontsize=16, fontweight='bold')
            
            axes_flat = axes.flatten()
            
            for idx, col in enumerate(feature_names):
                if idx >= 6:
                    break
                ax = axes_flat[idx]
                
                for code in fault_df['fault_code'].unique():
                    mask = fault_df['fault_code'] == code
                    label = fault_df[mask]['fault_name'].iloc[0] if mask.any() else f'代码{code}'
                    ax.hist(fault_df.loc[mask, col], alpha=0.6, label=label, bins=20)
                
                ax.set_title(f'{col}', fontsize=10, fontweight='bold')
                ax.set_xlabel('值')
                ax.set_ylabel('频数')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            for idx in range(len(feature_names), 6):
                axes_flat[idx].set_visible(False)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            viz_img = Image.open(buf)
            plt.close()
        
        output_path = DATA_DIR / f"simulated_fault_{fault_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fault_df.to_csv(output_path, index=False)
        
        logging.info(f"✓ 故障数据已保存: {output_path}")
        
        return str(output_path), result_text, viz_img
        
    except Exception as e:
        error_msg = f"❌ 故障模拟过程中出现错误: {str(e)}"
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
    
    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    default_start = today_start.strftime("%Y-%m-%d %H:%M:%S")
    default_end = now.strftime("%Y-%m-%d %H:%M:%S")
    
    with gr.Blocks(title="空压机智能故障分类诊断系统", js=health_check_js) as iface:
        gr.Markdown("""
        # 🔧 空压机智能故障分类诊断系统
        **功能特点：** 基于机器学习的设备故障分类与智能分析
        """)
        
        with gr.Tab("📊 故障诊断"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔧 模型配置")
                    model_dropdown = gr.Dropdown(
                        choices=list(model_options.keys()) if model_options else ["无可用模型"],
                        value=list(model_options.keys())[0] if model_options and "无可用模型" not in model_options else "无可用模型",
                        label="选择模型",
                        info="选择用于预测的模型"
                    )
                    
                    gr.Markdown("### 📁 数据输入 (优先使用模拟数据)")
                    
                    simulated_dropdown = gr.Dropdown(
                        choices=list(simulated_files.keys()) if simulated_files else ["暂无模拟数据"],
                        value=list(simulated_files.keys())[0] if simulated_files else "暂无模拟数据",
                        label="选择模拟数据",
                        info="选择之前生成的模拟数据（优先）"
                    )
                    
                    refresh_btn = gr.Button("🔄 刷新模拟数据列表", size="sm")
                    
                    gr.Markdown("或")
                    
                    csv_input = gr.File(
                        label="上传CSV文件 (备选)",
                        file_types=[".csv"],
                        type="filepath"
                    )
                    
                    predict_btn = gr.Button("🔍 开始诊断", variant="primary", size="lg")
                    
                    with gr.Accordion("📋 使用说明", open=False):
                        gr.Markdown("""
                        **使用步骤：**
                        1. 选择已训练的模型
                        2. 优先选择模拟数据（可点击刷新按钮更新列表）
                        3. 或上传CSV文件作为备选
                        4. 点击"开始诊断"按钮
                        
                        **输出内容：**
                        - 故障分类结果
                        - 详细故障点分析报告
                        - 可能原因和处理建议
                        - 故障分布可视化
                        """)
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 诊断结果")
                    result_summary = gr.Textbox(label="诊断概要", lines=6, interactive=False)
                    
                    with gr.Row():
                        distribution_chart = gr.Image(label="故障分布分析", height=400)
                    
                    fault_report = gr.Textbox(label="详细故障分析报告", lines=25, interactive=False)
        
        with gr.Tab("🎯 故障模拟"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📅 时间范围")
                    with gr.Row():
                        start_time_input = gr.Textbox(
                            label="开始时间",
                            value=default_start,
                            placeholder="2025-11-12 00:00:00",
                            info="格式: YYYY-MM-DD HH:MM:SS"
                        )
                        end_time_input = gr.Textbox(
                            label="结束时间",
                            value=default_end,
                            placeholder="2025-11-12 23:59:59",
                            info="格式: YYYY-MM-DD HH:MM:SS"
                        )
                    
                    with gr.Row():
                        today_btn = gr.Button("📅 今天", size="sm")
                        yesterday_btn = gr.Button("📅 昨天", size="sm")
                        last_week_btn = gr.Button("📅 最近7天", size="sm")
                    
                    gr.Markdown("### ⚙️ 故障参数")
                    fault_code_input = gr.Dropdown(
                        choices=[(f"{k}: {v['name']}", k) for k, v in fault_definitions.items()],
                        value=list(fault_definitions.keys())[0] if fault_definitions else None,
                        label="故障类型",
                        info="选择要模拟的故障类型"
                    )
                    
                    num_samples_slider = gr.Slider(
                        minimum=2,
                        maximum=500,
                        value=200,
                        step=50,
                        label="生成样本数"
                    )
                    
                    severity_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="预警严重程度",
                        info="0.1=轻微, 1.0=严重"
                    )
                    
                    gen_warning_checkbox = gr.Checkbox(
                        label="生成预警数据",
                        value=True
                    )
                    
                    show_sim_viz_checkbox = gr.Checkbox(
                        label="显示可视化结果",
                        value=True
                    )
                    
                    simulate_btn = gr.Button("🎯 开始模拟", variant="primary", size="lg")
                    
                    with gr.Accordion("📋 使用说明", open=False):
                        gr.Markdown("""
                        **使用步骤：**
                        1. 设置时间范围（默认为今天零点到当前时间）
                        2. 选择故障类型和参数
                        3. 点击"开始模拟"按钮
                        
                        **数据来源：**
                        - 从TDengine数据库实时获取正常运行数据
                        - 基于真实数据生成模拟故障样本
                        - 生成的数据可用于故障诊断测试
                        """)
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 模拟结果")
                    sim_result_file = gr.Textbox(label="结果文件路径", interactive=False)
                    sim_result_text = gr.Textbox(label="模拟统计", lines=10, interactive=False)
                    
                    sim_viz_output = gr.Image(label="特征分布可视化", height=500)
        
        # 刷新模拟数据列表
        def refresh_simulated_files():
            get_simulated_files()
            choices = list(simulated_files.keys()) if simulated_files else ["暂无模拟数据"]
            value = choices[0] if simulated_files else "暂无模拟数据"
            return gr.update(choices=choices, value=value)
        
        refresh_btn.click(
            refresh_simulated_files,
            outputs=[simulated_dropdown]
        )
        
        # 快捷时间按钮
        def set_today():
            now = datetime.now()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return today_start.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")
        
        def set_yesterday():
            now = datetime.now()
            yesterday = now - timedelta(days=1)
            yesterday_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_end = yesterday.replace(hour=23, minute=59, second=59, microsecond=0)
            return yesterday_start.strftime("%Y-%m-%d %H:%M:%S"), yesterday_end.strftime("%Y-%m-%d %H:%M:%S")
        
        def set_last_week():
            now = datetime.now()
            week_ago = now - timedelta(days=7)
            return week_ago.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")
        
        today_btn.click(set_today, outputs=[start_time_input, end_time_input])
        yesterday_btn.click(set_yesterday, outputs=[start_time_input, end_time_input])
        last_week_btn.click(set_last_week, outputs=[start_time_input, end_time_input])
        
        # 预测按钮事件
        predict_btn.click(
            predict_from_csv,
            inputs=[csv_input, model_dropdown, simulated_dropdown],
            outputs=[result_summary, fault_report, distribution_chart]
        )
        
        # 模拟按钮事件
        simulate_btn.click(
            simulate_fault_from_db,
            inputs=[
                fault_code_input, num_samples_slider, severity_slider,
                gen_warning_checkbox, start_time_input, end_time_input,
                show_sim_viz_checkbox
            ],
            outputs=[sim_result_file, sim_result_text, sim_viz_output]
        )
    
    return iface

def main():
    """主函数"""
    print(f"\n{'='*80}")
    print("故障分类推理系统 Gradio 应用 (优化版)")
    print(f"{'='*80}\n")
    
    initialize_models()
    get_simulated_files()
    
    monitor_manager = None
    if MultiDirectoryMonitor is not None:
        monitor_manager = MultiDirectoryMonitor(restart_signal_file_name=RESTART_SIGNAL_FILENAME)
        monitor_manager.add_directory(MODEL_BASE_DIR)
        # monitor_manager.add_directory(DATA_DIR)
        if EXAMPLE_DIR.exists():
            monitor_manager.add_directory(EXAMPLE_DIR)
        
        if not monitor_manager.start_all():
            logging.error("❌ 启动目录监控失败")
        else:
            logging.info("✅ 目录监控已启动")
    
    port = 7864
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            if port < 1024 or port > 65535:
                logging.warning(f"端口号 {port} 不在有效范围内，使用默认端口 7864")
                port = 7864
        except ValueError:
            logging.warning(f"无效的端口号参数，使用默认端口 7864")
    
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