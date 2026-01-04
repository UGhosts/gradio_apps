import random
from datetime import datetime

import gradio as gr
import time
import os
import sys
import csv
selected_preset = None
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
from utils.app_utils import AppUtils as util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plt = util.auto_config_chinese_font()
from sklearn.utils import shuffle
import warnings
import joblib
import os

os.makedirs(f'{BASE_DIR}/output/duanmo_prd/', exist_ok=True)

def generate_membrane_report(excel_path, save_image_path=None):
    # -------------------------- 初始化配置 --------------------------
    # 图片路径处理
    if save_image_path is None:
        # 默认保存路径：当前目录 + 时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_image_path = f'断膜与异常概率趋势图_{timestamp}.png'
    else:
        # 确保保存目录存在
        save_dir = os.path.dirname(save_image_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # -------------------------- 1. 读取并验证数据 --------------------------
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        raise ValueError(f"读取Excel文件失败：{str(e)}")

    # 验证必要列
    required_columns = ['SAVETIME', '延伸后  处理上/R速度现在监视器', '异常概率']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Excel文件缺少必要列：{col}")

    # 检查偏离模式列是否存在（报告需要）
    if '偏离模式' not in df.columns:
        df['偏离模式'] = '无数据'  # 填充默认值

    # -------------------------- 2. 数据预处理 --------------------------
    # 提取需要的列并删除缺失值
    df_clean = df[['SAVETIME', '延伸后  处理上/R速度现在监视器', '异常概率', '偏离模式']].copy()
    df_clean = df_clean.dropna(subset=['SAVETIME', '延伸后  处理上/R速度现在监视器', '异常概率'])

    # 处理“是否断膜”列：速度为0→y=1，非0→y=0
    df_clean['是否断膜'] = df_clean['延伸后  处理上/R速度现在监视器'].apply(lambda x: 1 if x == 0 else 0)

    # 时间排序
    df_clean = df_clean.sort_values(by='SAVETIME').reset_index(drop=True)

    # 筛选异常概率>0.5的点（用于标注和报告）
    high_prob_points = df_clean[df_clean['异常概率'] > 0.5].copy()

    # -------------------------- 3. 核心函数定义 --------------------------
    def plot_continuous_color_line(ax, x, y, threshold=0.5, color_below='#00A86B', color_above='#E63946', linewidth=2):
        """绘制连续的双色线条"""
        intersect_points = []
        x_arr = np.array(x)
        y_arr = np.array(y)

        for i in range(len(x_arr) - 1):
            y1, y2 = y_arr[i], y_arr[i + 1]
            x1, x2 = x_arr[i], x_arr[i + 1]

            if y1 <= threshold and y2 <= threshold:
                ax.plot([x1, x2], [y1, y2], color=color_below, linewidth=linewidth, alpha=0.9)
            elif y1 > threshold and y2 > threshold:
                ax.plot([x1, x2], [y1, y2], color=color_above, linewidth=linewidth, alpha=0.9)
            else:
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                if slope == 0:
                    x_intersect = x1
                else:
                    x_intersect = x1 + (threshold - y1) / slope
                y_intersect = threshold
                intersect_points.append((x_intersect, y_intersect))

                if y1 <= threshold:
                    ax.plot([x1, x_intersect], [y1, y_intersect], color=color_below, linewidth=linewidth, alpha=0.9)
                    ax.plot([x_intersect, x2], [y_intersect, y2], color=color_above, linewidth=linewidth, alpha=0.9)
                else:
                    ax.plot([x1, x_intersect], [y1, y_intersect], color=color_above, linewidth=linewidth, alpha=0.9)
                    ax.plot([x_intersect, x2], [y_intersect, y2], color=color_below, linewidth=linewidth, alpha=0.9)
        return intersect_points

    def annotate_high_prob_points(ax, df_high, x_base, offset_y=0.05):
        """标注高概率值"""
        for idx, row in df_high.iterrows():
            x_pos = idx
            y_pos = row['异常概率'] + offset_y
            prob_value = round(row['异常概率'], 4)

            ax.annotate(
                f'{prob_value}',
                xy=(x_pos, row['异常概率']),
                xytext=(x_pos, y_pos),
                fontsize=8,
                color='#E63946',
                fontweight='bold',
                ha='center',
                va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    facecolor='white',
                    edgecolor='#E63946',
                    alpha=0.8
                ),
                arrowprops=dict(
                    arrowstyle='->',
                    color='#E63946',
                    alpha=0.6,
                    lw=0.8
                )
            )

    # -------------------------- 4. 绘制图表 --------------------------
    fig, ax1 = plt.subplots(figsize=(18, 10))

    # 颜色方案
    color_membrane = '#2E86AB'
    color_prob_green = '#00A86B'
    color_prob_red = '#E63946'
    color_threshold = '#C73E1D'

    # 左轴：是否断膜
    ax1.set_xlabel('时间（SAVETIME）', fontsize=12, fontweight='bold')
    ax1.set_ylabel('是否断膜', color=color_membrane, fontsize=12, fontweight='bold')
    ax1.step(
        range(len(df_clean)),
        df_clean['是否断膜'],
        color=color_membrane,
        linewidth=2.5,
        alpha=0.8,
        where='mid'
    )
    ax1.tick_params(axis='y', labelcolor=color_membrane, labelsize=10)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['正常（非0）', '断膜（0）'], fontsize=10)

    # 右轴：异常概率
    ax2 = ax1.twinx()
    ax2.set_ylabel('异常概率', fontsize=12, fontweight='bold')

    # 绘制双色连续线
    x_data = range(len(df_clean))
    y_data = df_clean['异常概率'].values
    plot_continuous_color_line(ax2, x_data, y_data, threshold=0.5)

    # 绘制阈值线
    ax2.axhline(
        y=0.5,
        color=color_threshold,
        linestyle='--',
        linewidth=2,
        alpha=0.7,
        label='阈值=0.5'
    )

    # 标注高概率点
    if len(high_prob_points) > 0:
        annotate_high_prob_points(ax2, high_prob_points, x_data)

    # 右轴样式
    ax2.tick_params(axis='y', labelsize=10)
    ax2.set_ylim(0, 1.15)
    ax2.set_yticks(np.arange(0, 1.1, 0.1))

    # 图例
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color=color_prob_green, lw=2, label='异常概率≤0.5'),
        Line2D([0], [0], color=color_prob_red, lw=2, label='异常概率>0.5（标注数值）'),
        Line2D([0], [0], color=color_threshold, lw=2, linestyle='--', label='阈值=0.5')
    ]
    ax2.legend(handles=custom_lines, loc='upper right', fontsize=10, frameon=True, shadow=True)

    # 横坐标优化
    step = max(1, len(df_clean) // 25)
    x_ticks = range(0, len(df_clean), step)
    x_tick_labels = [t.strftime('%H:%M:%S') for t in df_clean['SAVETIME'].iloc[x_ticks]]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=9)

    # 标题和网格
    plt.title(
        '断膜状态与异常概率趋势图（标注>0.5概率值）',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    ax1.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(
        save_image_path,
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()  # 关闭画布释放资源

    # -------------------------- 5. 生成报告内容 --------------------------
    # 基础信息
    analysis_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total_data_points = len(df_clean)

    # 构建报告
    report_lines = []
    report_lines.append('=' * 80)
    report_lines.append('断膜预测报告')
    report_lines.append('=' * 80)
    report_lines.append(f'分析时间: {analysis_time}')
    report_lines.append(f'数据点数: {total_data_points}')
    report_lines.append('')
    report_lines.append('【预测统计】 （只统计异常概率大于0.5的）')
    report_lines.append('-' * 80)
    report_lines.append(f'{"时间":<40} {"预测概率":<20} {"TOP5影响因子"}')
    #report_lines.append(f'{"(SAVETIME列)":<40} {"（异常概率列）":<20} {"(偏离模式列)"}')
    report_lines.append('')

    # 添加高概率数据行
    if len(high_prob_points) > 0:
        for _, row in high_prob_points.iterrows():
            time_str = row['SAVETIME'].strftime('%Y-%m-%d %H:%M:%S')
            prob_str = f"{row['异常概率']:.4f}"
            factor_str = str(row['偏离模式']) if pd.notna(row['偏离模式']) else '无'
            report_lines.append(f'{time_str:<40} {prob_str:<20} {factor_str}')
    else:
        report_lines.append('无异常概率大于0.5的数据记录')

    report_lines.append('')
    report_lines.append('=' * 80)
    report_lines.append('报告结束')
    report_lines.append('=' * 80)

    # 拼接报告字符串
    report_content = '\n'.join(report_lines)

    # -------------------------- 6. 返回结果 --------------------------
    return save_image_path, report_content



def clean_numeric_column(series):
    """清理数值列：去除非数字字符，转换为数值型，处理无法转换的值"""
    series = series.astype(str).replace(
        r'[^\d\.\-]', '', regex=True  # 保留数字、小数点、负号
    )
    series = pd.to_numeric(series, errors='coerce')
    return series


def calculate_feature_importance(clf, X, feature_names, n_repeats=3, random_state=42):
    """计算特征重要性（排列重要性），用于确定偏离因子排名"""
    print("\n=== 计算特征重要性（排列重要性）===")
    # 1. 原始基准分数
    original_scores = clf.decision_function(X)
    original_mean_score = np.mean(original_scores)

    # 2. 逐个特征打乱计算重要性
    importance_scores = []
    avg_score_change_list = []  # 新增：单独存储得分列表用于求和
    for i, feature in enumerate(feature_names):
        if i % 10 == 0:
            print(f"处理第 {i + 1}/{len(feature_names)} 个特征...")

        score_changes = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[:, i] = shuffle(X_shuffled[:, i], random_state=random_state + _)
            shuffled_scores = clf.decision_function(X_shuffled)
            score_change = abs(original_mean_score - np.mean(shuffled_scores))
            score_changes.append(score_change)

        avg_score_change = np.mean(score_changes)
        avg_score_change_list.append(avg_score_change)  # 收集得分
        importance_scores.append({
            '特征名': feature,
            '重要性得分': avg_score_change,
            '重要性归一化得分': 0.0  # 先初始化，后续统一计算
        })

    # 3. 统一计算归一化得分（修复核心错误）
    total_score = sum(avg_score_change_list) if sum(avg_score_change_list) > 0 else 1e-8
    for idx, item in enumerate(importance_scores):
        item['重要性归一化得分'] = avg_score_change_list[idx] / total_score

    # 4. 排序并保存
    importance_df = pd.DataFrame(importance_scores)
    importance_df = importance_df.sort_values('重要性得分', ascending=False).reset_index(drop=True)
    importance_df['重要性排名'] = range(1, len(importance_df) + 1)

    return importance_df


def get_sample_top5_deviation(sample, feature_importance, scaler, numeric_cols):
    """
    计算单个样本的Top5偏离因子
    :param sample: 单条样本数据（Series）
    :param feature_importance: 特征重要性DataFrame
    :param scaler: 标准化器
    :param numeric_cols: 数值特征列表
    :return: Top5偏离因子列表，格式为[['排名', '特征名', '得分'], ...]
    """
    # 1. 计算每个特征的偏离度（样本值与训练均值的标准差倍数）
    deviation_scores = []
    for feat in numeric_cols:
        if feat in sample.index:
            try:
                # 训练数据的均值和标准差
                feat_idx = numeric_cols.index(feat)
                train_mean = scaler.mean_[feat_idx]
                train_std = scaler.scale_[feat_idx] if scaler.scale_[feat_idx] != 0 else 1e-8

                # 偏离度 = (样本值 - 均值) / 标准差（绝对值）
                feat_value = sample[feat] if not pd.isna(sample[feat]) else 0
                deviation = abs((feat_value - train_mean) / train_std)

                # 结合特征重要性的加权得分 = 偏离度 * 特征重要性
                feat_importance_row = feature_importance[feature_importance['特征名'] == feat]
                if not feat_importance_row.empty:
                    feat_importance = feat_importance_row['重要性得分'].values[0]
                    weighted_score = deviation * feat_importance

                    deviation_scores.append({
                        '特征名': feat,
                        '偏离度': deviation,
                        '重要性得分': feat_importance,
                        '加权得分': weighted_score,
                        '重要性排名': feat_importance_row['重要性排名'].values[0]
                    })
            except Exception as e:
                # 跳过计算出错的特征
                continue

    # 2. 按加权得分排序，取Top5
    deviation_df = pd.DataFrame(deviation_scores)
    if not deviation_df.empty:
        deviation_df = deviation_df.sort_values('加权得分', ascending=False).head(5).reset_index(drop=True)
    else:
        return [['无', '无', '0.0000']]

    # 3. 格式化结果
    top5_list = []
    for idx, row in deviation_df.iterrows():
        top5_list.append([
            #str(int(row['重要性排名'])),
            row['特征名'],
            f"{row['加权得分']:.4f}"
        ])

    # 补全不足5个的情况
    while len(top5_list) < 5:
        top5_list.append(['无', '无', '0.0000'])

    return top5_list


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 新增：非线性映射函数，增强概率区分度
def enhance_probability_discrimination(prob):
    enhanced = np.where(
        prob <= 0.5,
        # 小于等于0.5：压缩到0附近（三次函数）
        # (prob / 0.5) ** 6 * 0.5,
        # # 大于0.5：拉伸到1附近（三次函数）
        # 1 - ((1 - prob) / 0.5) ** 6 * 0.5
        sigmoid((prob / 0.5 - 1) * 10),
        # 大于0.5：拉伸到1附近（Sigmoid右半部分）
        # 映射逻辑：prob(0.5→1) → x(0→6) → sigmoid输出(0.5→0.998)
        sigmoid((1 - prob) / 0.5 * 2)
    )
    # 兜底限制在0-1范围内
    return np.clip(enhanced, 0, 1)


def predict_new_data(
        new_file_path,SCALER_PATH,PCA_PATH,MODEL_PATH,COLUMNS_PATH,IMPORTANCE_PATH,
        sheet_name=0,
        output_file="new_data_predictions.xlsx",
        time_column="SAVETIME"
):
    """预测新数据：新增孤立样本的Top5偏离因子和模式列"""
    required_files = [MODEL_PATH, SCALER_PATH, COLUMNS_PATH, IMPORTANCE_PATH]
    if os.path.exists(PCA_PATH):
        required_files.append(PCA_PATH)

    if not all(os.path.exists(f) for f in required_files):
        print("错误：未找到模型/特征重要性文件，请先训练！")
        return None

    try:
        # 1. 加载模型和配置
        scaler = joblib.load(SCALER_PATH)
        clf = joblib.load(MODEL_PATH)
        pca = joblib.load(PCA_PATH) if os.path.exists(PCA_PATH) else None
        feature_importance = pd.read_excel(IMPORTANCE_PATH)

        with open(COLUMNS_PATH, 'r', encoding='utf-8') as f:
            numeric_cols = [line.strip() for line in f.readlines()]

        # 2. 读取新数据
        df_new = pd.read_excel(new_file_path, sheet_name=sheet_name)
        time_series = df_new.iloc[:, 0].copy()
        df_data = df_new.iloc[:, 1:].copy()
        time_series = time_series.iloc[1:].reset_index(drop=True)
        df_data = df_data.iloc[1:].reset_index(drop=True)

        # 3. 预处理新数据
        df_data_cleaned = df_data.copy()
        for col in numeric_cols:
            if col in df_data_cleaned.columns:
                df_data_cleaned[col] = clean_numeric_column(df_data_cleaned[col])
            else:
                df_data_cleaned[col] = 0

        X_new = df_data_cleaned[numeric_cols].copy()
        for col in X_new.columns:
            if X_new[col].notna().sum() > 0:
                #X_new[col].fillna(X_new[col].median(), inplace=True)
                X_new[col] = X_new[col].fillna(X_new[col].median())
            else:
                X_new[col].fillna(0, inplace=True)
        X_new = X_new.fillna(0)

        # 标准化
        X_new_scaled = scaler.transform(X_new)
        X_new_scaled = np.nan_to_num(X_new_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # PCA降维
        if pca is not None:
            X_new_pca = pca.transform(X_new_scaled)
        else:
            X_new_pca = X_new_scaled

        # 4. 预测
        anomaly_labels = clf.predict(X_new_pca)
        isolation_score_new = clf.decision_function(X_new_pca)

        # 计算异常概率（核心修改：增强区分度）
        normalized_score_new = (isolation_score_new - (-1)) / (1 - (-1))
        raw_prob_new = 1 - normalized_score_new
        # 增强概率区分度：<0.5靠近0，>0.5靠近1
        anomaly_prob = enhance_probability_discrimination(raw_prob_new)
        anomaly_prob = anomaly_prob.clip(0, 1)

        # 5. 构建结果
        df_result = pd.DataFrame({
            time_column: time_series,
            **df_data_cleaned.to_dict('series'),
            '异常标签': anomaly_labels,
            '孤立程度分数': isolation_score_new,
            '异常概率': anomaly_prob,
            '是否孤立': np.where(anomaly_prob > 0.5, "是", "否")
        })

        # 6. 为孤立样本计算Top5偏离因子并生成模式列
        df_result['偏离模式'] = ""
        outlier_mask = df_result['是否孤立'] == "是"

        # 遍历每个孤立样本
        top5_summary = []
        for idx, row in df_result[outlier_mask].iterrows():
            # 计算该样本的Top5偏离因子
            top5_list = get_sample_top5_deviation(
                sample=row,
                feature_importance=feature_importance,
                scaler=scaler,
                numeric_cols=numeric_cols
            )
            # 生成模式列（格式：{['重要性排名', '特征名', '得分']}）
            df_result.loc[idx, '偏离模式'] = str(top5_list)
            # 汇总Top5信息用于打印
            top5_summary.append({
                '时间': row[time_column],
                '异常概率': row['异常概率'],
                'Top5偏离因子': top5_list
            })

        # 8. 按时间升序排列
        try:
            df_result[time_column] = pd.to_datetime(df_result[time_column])
            df_result_sorted = df_result.sort_values(by=time_column, ascending=True).reset_index(drop=True)
        except Exception as e:
            df_result_sorted = df_result.sort_values(by=time_column, ascending=True).reset_index(drop=True)

        # 9. 添加排名
        df_result_sorted['异常程度排名'] = range(1, len(df_result_sorted) + 1)

        # 10. 保存结果
        df_result_sorted.to_excel(output_file, index=False)


        return df_result_sorted

    except Exception as e:
        print(f"预测新数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_input(selected_model_dir):
    """处理全局选中的测试文件，返回图表和结果"""
    time.sleep(1)
    preset_info = f"测试文件: {selected_preset}" if selected_preset else "未选择测试文件"
    model_info = f"模型目录: {selected_model_dir}"

    SCALER_PATH = os.path.join(selected_model_dir, "scaler.pkl")
    PCA_PATH = os.path.join(selected_model_dir, "pca.pkl")
    MODEL_PATH = os.path.join(selected_model_dir, "isolation_forest_model.pkl")
    COLUMNS_PATH = os.path.join(selected_model_dir, "numeric_columns.txt")
    IMPORTANCE_PATH = os.path.join(selected_model_dir, "feature_importance.xlsx")
    new_file_path = selected_preset
    file_name =f"{BASE_DIR}/output/duanmo_prd/"+str(random.randint(1,2000000))
    output_file= file_name+'.xlsx'
    predict_new_data(
            new_file_path,SCALER_PATH,PCA_PATH,MODEL_PATH,COLUMNS_PATH,IMPORTANCE_PATH,
            sheet_name=0,
            output_file=output_file,
            time_column="SAVETIME",
    )
    # 获取报告和作图内容
    save_pic_name,result = generate_membrane_report(output_file,file_name +'.jpg')
    return save_pic_name, f"处理完成!\n{result}\n"


def set_selected(file_path, buttons, file_paths):
    """更新选中状态，修改按钮样式并更新全局变量"""
    global selected_preset
    selected_preset = file_path

    # 返回所有按钮的样式更新列表
    # 对于每个按钮，如果它对应的文件路径与选中的文件路径相同，则设置为primary（高亮），否则设置为secondary（默认）
    return [gr.update(variant="primary" if fp == file_path else "secondary") for fp, btn in zip(file_paths, buttons)]


def create_interface():
    # 从dataset/cwru_cls_test目录动态读取CSV文件
    cwru_dir = os.path.join(os.path.dirname(__file__), "dataset", "duanmo_prd")
    preset_files = {}

    # 确保使用绝对路径或者正确的相对路径
    if not os.path.exists(cwru_dir):
        # 尝试使用其他可能的路径
        alt_paths = [
            f"{BASE_DIR}/dataset/duanmo_prd",
            "./dataset/duanmo_prd",
            "dataset/duanmo_prd",
        ]
        for path in alt_paths:
            if os.path.exists(path):
                cwru_dir = path
                break

    # 获取目录下所有CSV文件
    if os.path.exists(cwru_dir):
        for file_name in os.listdir(cwru_dir):
            if file_name.endswith('.xlsx'):
                file_path = os.path.join(cwru_dir, file_name)
                preset_files[file_path] = f"📄 {file_name}"

    model_dir = os.path.join(os.path.dirname(__file__), "model", "duanmo_prd")
    model_options = []  # 将使用元组列表: [(子目录名称, 完整路径)]

    if not os.path.exists(model_dir):
        # 尝试使用其他可能的路径
        alt_model_paths = [
            f"{BASE_DIR}/model/duanmo_prd",
            "./model/duanmo_prd",
            "model/duanmo_prd",
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
    # if not model_options:
    #     default_model_name = "DLinear"
    #     default_model_dir = os.path.join(model_dir, default_model_name)
    #     model_options.append((default_model_name, default_model_dir))

    with gr.Blocks(title="断膜检测与诊断综合应用") as demo:
        gr.Markdown("# 🚀 断膜检测与诊断综合应用")

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
                gr.Markdown("### 预测曲线图")
                plot_output = gr.Image(label="数据曲线", type="pil")

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
    demo.launch(allowed_paths=[f'{BASE_DIR}/output'],server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()