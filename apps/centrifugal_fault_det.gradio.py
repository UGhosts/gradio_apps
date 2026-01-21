import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import gradio as gr
import sys
import logging
from pathlib import Path
import io
from PIL import Image
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from utils.app_utils import AppUtils as util

warnings.filterwarnings('ignore')
plt = util.auto_config_chinese_font()

# ============================================================
# 配置路径 (请确保与训练脚本一致)
# ============================================================
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR  / "model" / "centrifugal_fault_det"/"model" / "final_model.pdparams" 
PARAMS_PATH = BASE_DIR  / "model" / "centrifugal_fault_det"/"model" / "X_train_params.npz"
DATA_DIR = BASE_DIR / "model" / "centrifugal_fault_det"/ "model" / "centrifugal_fault_det"
EXAMPLE_DIR = BASE_DIR /"dataset"/"centrifugal_fault_det"

# ============================================================
# 故障定义
# ============================================================
fault_code_map = {
    0: '正常状态',
    1: '流量输送阀卡涩',
    2: '离心泵入口堵塞',
    3: '离心泵入口温度升高汽蚀',
    4: '离心泵气缚',
    5: '离心泵吸入罐压力控制入口阀卡涩'
}

fault_analysis = {
    0: {
        'name': '正常状态',
        'description': '设备运行正常，所有参数在正常范围内',
        'key_indicators': [
            'P101出料流量: ~20000 Kg/h',
            'V101压力: ~0.5 MPaG',
            'P101A/B入口压力: ~0.475-0.5 MPaG',
            'P101A/B出口压力: ~1.5 MPaG',
            'V101液位: ~50%'
        ],
        'suggestions': ['继续保持正常运行', '定期维护保养']
    },
    1: {
        'name': '流量输送阀卡涩',
        'description': '流量控制阀门开度异常，影响流体输送',
        'key_indicators': [
            'FV101.OP（流量控制阀开度）波动异常',
            'FT101.PV（出料流量）下降',
            'PI102/PI104（泵出口压力）波动',
            '阀门响应迟缓或不响应'
        ],
        'root_causes': [
            '阀芯污垢积聚',
            '阀杆密封圈老化',
            '执行机构故障',
            '阀体内部腐蚀'
        ],
        'suggestions': [
            '检查阀门执行机构，确认气源或电源供应正常',
            '拆检阀门，清理阀芯和阀座上的污垢',
            '检查阀杆填料和密封圈，必要时更换',
            '润滑阀杆，确保动作灵活',
            '校验阀门定位器，确保反馈信号准确'
        ]
    },
    2: {
        'name': '离心泵入口堵塞',
        'description': '泵入口管道或过滤器堵塞，导致吸入不畅',
        'key_indicators': [
            'PI101/PI103（泵入口压力）明显降低',
            'FT101.PV（出料流量）下降',
            'PI102/PI104（泵出口压力）下降',
            '泵振动增加，出现异常噪音'
        ],
        'root_causes': [
            '过滤器堵塞',
            '管道内杂质积聚',
            '吸入口滤网堵塞',
            '管道结垢'
        ],
        'suggestions': [
            '立即检查泵入口过滤器，清理或更换滤芯',
            '检查入口管道，排除异物堵塞',
            '清洗吸入口滤网',
            '检查V101罐底是否有沉淀物，必要时进行清罐',
            '增加过滤器清洗频次，防止再次堵塞'
        ]
    },
    3: {
        'name': '离心泵入口温度升高汽蚀',
        'description': '入口温度过高导致液体汽化，产生汽蚀现象',
        'key_indicators': [
            'TI101.PV（V101进料温度）升高',
            'PI101/PI103（泵入口压力）接近或低于饱和蒸汽压',
            'FT101.PV（出料流量）波动或下降',
            '泵振动加剧，发出气泡破裂的噼啪声',
            '泵效率显著下降'
        ],
        'root_causes': [
            '冷却系统失效',
            '上游工艺温度控制不当',
            'V101罐体保温失效',
            '环境温度过高',
            '泵入口压力过低'
        ],
        'suggestions': [
            '立即降低V101进料温度，检查冷却系统',
            '提高V101罐压力（PT101.PV），增加泵入口NPSH余量',
            '检查上游工艺，确保进料温度符合设计要求',
            '检查泵入口管道保温情况，减少热量损失',
            '考虑增加冷却水流量或降低冷却水温度',
            '必要时降低泵运行转速或流量，减轻汽蚀程度'
        ]
    },
    4: {
        'name': '离心泵气缚',
        'description': '泵内积聚气体，导致泵无法正常输送液体',
        'key_indicators': [
            'FT101.PV（出料流量）骤降至接近零',
            'PI102/PI104（泵出口压力）显著下降',
            'PI101/PI103（泵入口压力）正常或略高',
            '泵电流下降',
            '泵运转声音异常，类似空转'
        ],
        'root_causes': [
            'V101液位过低，吸入口暴露',
            '泵启动前未排气',
            '入口管道漏气',
            '液体中溶解气体过多',
            '泵密封失效导致空气吸入'
        ],
        'suggestions': [
            '立即停泵，检查V101液位（LT101.PV），确保液位正常',
            '打开泵顶部排气阀，充分排出气体',
            '检查泵入口管道及法兰连接，排除漏气点',
            '检查泵机械密封，确保密封完好',
            '重新启动前确保泵体充满液体',
            '若液位正常但仍气缚，检查液体是否含气过多，必要时增加脱气措施'
        ]
    },
    5: {
        'name': '离心泵吸入罐压力控制入口阀卡涩',
        'description': 'V101压力控制入口阀（PV101A）卡涩，影响罐压调节',
        'key_indicators': [
            'PV101A.OP（压力控制入口阀开度）异常或不变化',
            'PT101.PV（V101压力）波动或偏离设定值',
            '压力控制响应迟缓',
            'LT101.PV（V101液位）可能波动'
        ],
        'root_causes': [
            '阀芯卡死',
            '执行机构故障',
            '阀门定位器失效',
            '控制信号异常',
            '阀体内部结垢或腐蚀'
        ],
        'suggestions': [
            '切换至手动控制，手动调节PV101A阀门开度',
            '检查阀门执行机构和定位器，确认动作是否正常',
            '检查控制系统信号，确保PID控制器输出正常',
            '必要时拆检阀门，清理阀芯和阀座',
            '润滑阀杆，确保阀门动作灵活',
            '校验压力变送器（PT101），确保测量准确',
            '检查压力控制逻辑，优化PID参数'
        ]
    }
}

# 特征中文名称映射
feature_names_cn = {
    'FT101': 'P101出料流量',
    'PI101': 'P101A入口压力',
    'PI102': 'P101A出口压力',
    'PT101': 'V101 压力变送器',
    'TT101': 'V101 进料温度',
    'LV101': 'V101液位控制阀开度',
    'PV101A': 'V101压力控制进口阀开度',
    'PV101B': 'V101压力控制出口阀开度',
    'FV101': 'P101出口流量控制阀开度'
}


# ============================================================
# 模型定义 - 必须与训练时的 FaultNet 完全一致
# ============================================================
class FaultNet(nn.Layer):
    def __init__(self, num_classes):
        super(FaultNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1D(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1D(64),
            nn.ReLU(),
            nn.MaxPool1D(2),
            
            nn.Conv1D(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1D(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1D(1) 
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = paddle.flatten(x, 1)
        x = self.fc(x)
        return x

# ============================================================
# 推理类优化
# ============================================================
class FaultClassifier:
    def __init__(self, model_path, params_path):
        # 加载归一化参数
        self.params = np.load(params_path)
        self.mean = self.params['mean'].astype('float32')
        self.std = self.params['std'].astype('float32')
        
        num_classes = 6
        
        # 初始化模型
        self.model = FaultNet(num_classes)
        
        # 加载权重
        state_dict = paddle.load(str(model_path))
        self.model.set_state_dict(state_dict)
        self.model.eval()
        
        logging.info(f"✓ 模型加载成功，模型路径: {model_path}")
    
    def normalize_data(self, X):
        """完全对齐训练时的归一化逻辑"""
        normalized = ((X - self.mean) / (self.std + 1e-8)).astype('float32')
        return normalized
    
    def predict(self, df):
        """批量预测"""
        # 1. 确保特征顺序
        expected_cols = list(feature_names_cn.keys())
        X_raw = df[expected_cols].values.astype('float32')
        
        # 2. 归一化
        X_normalized = self.normalize_data(X_raw)
        
        # 3. 调整形状 [Batch, Channels, Features]
        X_input = X_normalized.reshape(-1, 1, X_normalized.shape[1])
        X_tensor = paddle.to_tensor(X_input, dtype='float32')
        
        # 4. 推理
        with paddle.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, axis=1)
            preds = paddle.argmax(logits, axis=1)
        
        # 5. 组装结果
        results = pd.DataFrame({
            'fault_code': preds.numpy(),
            'fault_name': [fault_code_map[c] for c in preds.numpy()],
            'confidence': probs.numpy().max(axis=1)
        })
        
        # 添加详细概率列
        for i in range(6):
            results[f'prob_class_{i}'] = probs.numpy()[:, i]
        
        return results

# ============================================================
# 样例数据管理
# ============================================================
def get_example_files():
    """获取所有样例文件"""
    if not EXAMPLE_DIR.exists():
        EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        logging.warning(f"样例目录不存在，已创建: {EXAMPLE_DIR}")
        return {}
    
    example_files = {}
    for file in EXAMPLE_DIR.glob("*.csv"):
        # 使用文件名（不含扩展名）作为显示名称
        display_name = file.stem
        example_files[display_name] = str(file)
    
    return example_files

# ============================================================
# 诊断逻辑优化
# ============================================================
def diagnose_from_data(csv_file, example_choice):
    """
    从上传文件或样例数据进行诊断
    优先使用上传的文件，如果没有上传则使用选择的样例
    """
    global classifier
    
    # 确定使用哪个数据源
    data_source = None
    source_name = ""
    
    if csv_file is not None:
        # 优先使用上传的文件
        data_source = csv_file.name
        source_name = Path(csv_file.name).name
    elif example_choice and example_choice != "请选择样例数据":
        # 使用选择的样例
        example_files = get_example_files()
        if example_choice in example_files:
            data_source = example_files[example_choice]
            source_name = f"样例: {example_choice}"
        else:
            return None, "❌ 选择的样例文件不存在", None, None
    else:
        return None, "❌ 请上传CSV文件或选择样例数据！", None, None
    
    # 加载分类器
    if classifier is None:
        status = load_classifier()
        if "失败" in status:
            return None, status, None, None
    
    try:
        # 读取数据
        try:
            df = pd.read_csv(data_source, encoding='utf-8')
        except:
            df = pd.read_csv(data_source, encoding='gbk')
            
        # 检查数据是否足够
        if len(df) < 1:
            return None, f"❌ 数据文件 [{source_name}] 数据量不足", None, None

        # 检查特征列
        expected_cols = list(feature_names_cn.keys())
        df.columns = df.columns.str.strip()
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            return None, f"❌ 数据文件 [{source_name}] 缺少特征列: {missing_cols}", None, None
        
        # 执行预测
        predictions = classifier.predict(df)
        
        # 合并结果
        output_df = pd.concat([df.reset_index(drop=True), predictions], axis=1)
        
        # 生成可视化和报告
        report_text = generate_fault_report(predictions, source_name)
        distribution_chart = create_distribution_chart(predictions)
        confidence_chart = create_confidence_chart(predictions)
        
        # 保存诊断记录
        output_path = DATA_DIR / f"diagnosis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        summary = (f"📊 诊断完成！\n"
                   f"数据源: {source_name}\n"
                   f"样本总数: {len(df)}\n"
                   f"主要结论: {predictions['fault_name'].mode()[0]}\n"
                   f"结果已存至: {output_path.name}")
        
        return summary, report_text, distribution_chart, confidence_chart
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 处理数据 [{source_name}] 时出错: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg, None, None

def generate_fault_report(predictions, source_name=""):
    """生成详细故障报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("离心泵故障诊断分析报告")
    report_lines.append("=" * 80)
    report_lines.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if source_name:
        report_lines.append(f"数据源: {source_name}")
    report_lines.append(f"样本总数: {len(predictions)}")
    report_lines.append("")
    
    fault_counts = predictions['fault_code'].value_counts().sort_index()
    
    report_lines.append("【故障分布统计】")
    report_lines.append("-" * 80)
    for fault_code, count in fault_counts.items():
        percentage = count / len(predictions) * 100
        fault_name = fault_code_map[fault_code]
        avg_conf = predictions[predictions['fault_code'] == fault_code]['confidence'].mean()
        report_lines.append(
            f"  [{fault_code}] {fault_name}: {count}次 ({percentage:.2f}%) "
            f"- 平均置信度: {avg_conf:.2%}"
        )
    report_lines.append("")
    
    # 详细分析
    for fault_code in fault_counts.index:
        if fault_code == 0:  # 跳过正常状态的详细分析
            continue
            
        analysis = fault_analysis[fault_code]
        count = fault_counts[fault_code]
        percentage = count / len(predictions) * 100
        
        report_lines.append("=" * 80)
        report_lines.append(f"【故障类型 {fault_code}】{analysis['name']}")
        report_lines.append("=" * 80)
        report_lines.append(f"检出次数: {count}次 ({percentage:.2f}%)")
        report_lines.append(f"\n故障描述: {analysis['description']}\n")
        
        report_lines.append("► 关键指标:")
        for indicator in analysis['key_indicators']:
            report_lines.append(f"  • {indicator}")
        report_lines.append("")
        
        if 'root_causes' in analysis:
            report_lines.append("► 可能原因:")
            for cause in analysis['root_causes']:
                report_lines.append(f"  • {cause}")
            report_lines.append("")
        
        if 'suggestions' in analysis:
            report_lines.append("► 处理建议:")
            for i, suggestion in enumerate(analysis['suggestions'], 1):
                report_lines.append(f"  {i}. {suggestion}")
            report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("报告结束")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)

def create_distribution_chart(predictions):
    """创建故障分布图表"""
    fault_counts = predictions['fault_code'].value_counts().sort_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 柱状图
    fault_names = [fault_code_map[code] for code in fault_counts.index]
    colors = plt.cm.Set3(range(len(fault_counts)))
    
    bars = ax1.bar(range(len(fault_counts)), fault_counts.values, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(fault_counts)))
    ax1.set_xticklabels([f'{name}\n(代码{code})' 
                         for name, code in zip(fault_names, fault_counts.index)], 
                        rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('检出次数', fontsize=12, fontweight='bold')
    ax1.set_title('故障类型检出次数分布', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, count in zip(bars, fault_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 饼图
    wedges, texts, autotexts = ax2.pie(
        fault_counts.values, 
        labels=[f'{name}' for name in fault_names],
        autopct='%1.1f%%',
        colors=colors,
        explode=[0.05] * len(fault_counts),
        shadow=True,
        startangle=90
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    for text in texts:
        text.set_fontsize(10)
    
    ax2.set_title('故障类型占比分布', fontsize=14, fontweight='bold')
    
    plt.suptitle('离心泵故障诊断结果分布', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img

def create_confidence_chart(predictions):
    """创建置信度分析图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('各故障类型预测置信度分布', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i in range(6):
        ax = axes[i]
        fault_name = fault_code_map[i]
        prob_col = f'prob_class_{i}'
        
        data = predictions[prob_col]
        
        ax.hist(data, bins=50, color=plt.cm.Set3(i), alpha=0.7, edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'均值: {data.mean():.3f}')
        ax.set_title(f'[{i}] {fault_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('预测概率', fontsize=10)
        ax.set_ylabel('样本数量', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img


def load_classifier():
    global classifier
    try:
        real_model_path = MODEL_PATH
        if not real_model_path.exists():
            alt_path = MODEL_PATH.with_suffix('')
            if alt_path.exists():
                real_model_path = alt_path
            else:
                return f"❌ 找不到模型文件: {MODEL_PATH}\n请确认训练后已生成 .pdparams 文件"
        
        if not PARAMS_PATH.exists():
            return f"❌ 找不到均值参数文件: {PARAMS_PATH}"
        
        classifier = FaultClassifier(str(real_model_path), str(PARAMS_PATH))
        return "✅ 模型与参数加载成功！"
    except Exception as e:
        return f"❌ 加载失败: {str(e)}"

def create_gradio_interface():
    """创建Gradio界面"""
    # 获取样例文件列表
    example_files = get_example_files()
    example_choices = ["请选择样例数据"] + list(example_files.keys())
    
    with gr.Blocks(title="离心泵智能故障诊断系统") as iface:
        with gr.Tab("📊 故障诊断"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 数据输入")
                    
                    # 样例数据选择
                    gr.Markdown("#### 方式一：选择样例数据")
                    example_dropdown = gr.Dropdown(
                        choices=example_choices,
                        value="请选择样例数据",
                        label="样例数据（选择预置的样例数据进行分析）"
                    )
                    
                    gr.Markdown("#### 方式二：上传自定义数据")
                    csv_input = gr.File(
                        label="上传CSV文件（上传的文件将优先于样例数据）",
                        file_types=[".csv"],
                        type="filepath"
                    )
                    
                    gr.Markdown("---")
                    diagnose_btn = gr.Button("🔍 开始诊断", variant="primary", size="lg")
                    
                    # 提示信息
                    with gr.Accordion("💡 使用说明", open=False):
                        gr.Markdown("""
                        1. **使用样例**: 从下拉菜单选择样例数据后，点击"开始诊断"
                        2. **上传文件**: 上传CSV文件后，点击"开始诊断"（优先级高于样例）
                        3. **数据格式**: CSV需包含以下特征列：
                           - FT101, PI101, PI102, PT101, TT101
                           - LV101, PV101A, PV101B, FV101
                        """)
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 诊断结果")
                    result_summary = gr.Textbox(label="诊断概要", lines=10, interactive=False)
                    
                    with gr.Row():
                        distribution_chart = gr.Image(label="故障分布分析", height=350,buttons=['fullscreen'])
                        confidence_chart = gr.Image(label="置信度分析", height=350,buttons=['fullscreen'])
                    
                    fault_report = gr.Textbox(label="详细故障分析报告", lines=30, interactive=False)
        
        # 事件绑定
        diagnose_btn.click(
            diagnose_from_data,
            inputs=[csv_input, example_dropdown],
            outputs=[result_summary, fault_report, distribution_chart, confidence_chart]
        )
    
    return iface

# ============================================================
# 主函数
# ============================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print(f"\n{'='*80}")
    print("离心泵智能故障诊断系统")
    print(f"{'='*80}\n")
    
    # 初始化模型
    status = load_classifier()
    print(status)
    
    # 创建必要目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 检查样例文件
    example_files = get_example_files()
    if example_files:
        print(f"\n✓ 发现 {len(example_files)} 个样例文件:")
        for name in example_files.keys():
            print(f"  - {name}")
    else:
        print(f"\n⚠️  未发现样例文件")
        print(f"   请在 {EXAMPLE_DIR} 目录下添加CSV样例文件")
    
    # 启动界面
    port = 7865
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logging.warning(f"无效端口号，使用默认端口 {port}")
    
    print(f"\n{'='*80}")
    print(f"启动Web界面: http://0.0.0.0:{port}")
    print(f"{'='*80}\n")
    
    iface = create_gradio_interface()
    iface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )

if __name__ == '__main__':
    main()