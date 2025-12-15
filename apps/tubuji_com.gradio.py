import json
import datetime
import uuid

import joblib
import pandas as pd
import sys
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
import gradio as gr

selected_jt_file = None  # 机头选中文件
selected_hx_file = None  # 烘箱选中文件
BASE_DIR = Path(__file__).parent.parent
from utils.app_utils import AppUtils as util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plt = util.auto_config_chinese_font()
os.makedirs(f'{BASE_DIR}/output/tujiaoji_com/', exist_ok=True)

def predict_new_data(new_df, model_path: str = None,
                     model_name: str = None, scaler_name: str = None,feature_cols: str = None):
    """加载模型和标准化器，对新数据进行预测并输出分类概率"""
    model_path= model_path or './data/model/'
    model = joblib.load(model_path+'/'+model_name)
    scaler = joblib.load(model_path+'/'+scaler_name)

    with open(model_path+'/'+feature_cols, 'r') as f:
        feature_cols = f.read().splitlines()

    X_new = new_df[feature_cols]
    X_new_scaled = scaler.transform(X_new)
    y_new_pred = model.predict(X_new_scaled)
    y_new_pred_proba = model.predict_proba(X_new_scaled)

    result_df=pd.DataFrame()
    if 'jt_' in model_name: #机头
        result_df[f'正常'] = y_new_pred_proba[:, 0]
        result_df[f'速度异常'] = y_new_pred_proba[:, 1]
        result_df[f'张力异常'] = y_new_pred_proba[:, 2]
        result_df[f'泵异常'] = y_new_pred_proba[:, 3]
    elif 'hx_' in model_name: #烘箱
        result_df[f'正常'] = y_new_pred_proba[:, 0]
        result_df[f'烘箱温度异常'] = y_new_pred_proba[:, 1]
        result_df[f'发热包温度异常'] = y_new_pred_proba[:, 2]
        result_df[f'电机温度异常'] = y_new_pred_proba[:, 3]
        result_df[f'电机震动异常'] = y_new_pred_proba[:, 4]
        result_df[f'电流电压异常'] = y_new_pred_proba[:, 5]
    json_data = result_df.to_json(orient="records", force_ascii=False)
    return json_data


def split_csv_by_id(csv_path):
    df = pd.read_csv(csv_path)

    id_dataframe_dict = {}
    for idx in df['id'].unique():
        # 筛选当前id的行，保留DataFrame格式（而非Series）
        single_id_df = df[df['id'] == idx].reset_index(drop=True)
        id_dataframe_dict[int(idx)] = single_id_df

    # 校验是否拆分为12个DataFrame（匹配你的数据量）
    if len(id_dataframe_dict) != 12:
        print(f"警告：拆分后得到{len(id_dataframe_dict)}个DataFrame，预期12个")

    return id_dataframe_dict


def plot_combined_analysis_chart(prob_list, head_df, oven_prob_list, save_path='涂布机分析总图.png'):
    """
    绘制3行布局的整合分析图：
    1. 第1行：左侧扇形图 + 右侧柱状图（涂布机机头分析图）
    2. 第2-3行：2行6列12个柱状图（涂布机烘箱分析图）
    """
    # 步骤1：创建3行6列的画布网格
    fig = plt.figure(figsize=(24, 16))  # 适配2行6列的烘箱图
    gs = fig.add_gridspec(
        nrows=3, ncols=6,
        height_ratios=[1, 1, 1],  # 3行等高
        hspace=0.6, wspace=0.4     # 增大上下间距避免重叠
    )

    # 步骤2：绘制第1行 - 涂布机机头分析图
    # 2.1 左侧扇形图（占第1行第1列）
    ax_pie = fig.add_subplot(gs[0, 0])
    prob_dict = prob_list[0] if isinstance(prob_list, list) else prob_list
    pie_labels = list(prob_dict.keys())
    pie_sizes = list(prob_dict.values())
    pie_colors = ['#2E8B57', '#FF6347', '#FFD700', '#4169E1'][:len(pie_labels)]
    explode = [0.05 if s == max(pie_sizes) else 0 for s in pie_sizes]
    wedges, texts, autotexts = ax_pie.pie(
        pie_sizes, labels=pie_labels, colors=pie_colors,
        autopct='%1.1f%%', explode=explode, shadow=True,
        startangle=90, textprops={'fontsize': 9}
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax_pie.set_title('机头异常概率分布', fontsize=11, fontweight='bold', pad=10)

    # 2.2 右侧柱状图（占第1行第2-6列）
    ax_head_bar = fig.add_subplot(gs[0, 1:])
    bar_cols = ['YXSD', 'FJZLZ', 'QYZLZ', 'SJZLZ', 'BSZ']
    bar_labels = ['运行速度', '放卷张力值', '牵引张力值', '收卷张力值', '泵转速']
    bar_values = head_df[bar_cols].iloc[0].values
    bars = ax_head_bar.bar(
        bar_labels, bar_values,
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        width=0.6
    )
    for bar in bars:
        height = bar.get_height()
        ax_head_bar.text(
            bar.get_x() + bar.get_width()/2., height + max(bar_values)*0.01,
            f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    ax_head_bar.set_title('涂布机机头分析图', fontsize=14, fontweight='bold', pad=20)
    ax_head_bar.set_ylabel('参数值', fontsize=11)
    ax_head_bar.tick_params(axis='x', rotation=15)
    ax_head_bar.set_ylim(0, max(bar_values)*1.1)
    ax_head_bar.grid(axis='y', alpha=0.3)

    # 步骤3：绘制第2-3行 - 涂布机烘箱分析图
    oven_labels = ['正常', '烘箱温度异常', '发热包温度异常', '电机温度异常', '电机震动异常', '电流电压异常']
    oven_colors = ['#2E8B57', '#FF6347', '#FFD700', '#4169E1', '#8A2BE2', '#F08080']

    # 2行6列排列12个烘箱子图
    for idx, oven_prob in enumerate(oven_prob_list):
        row = 1 + (idx // 6)  # 第2行（idx0-5）/第3行（idx6-11）
        col = idx % 6         # 0-5列
        ax_oven = fig.add_subplot(gs[row, col])

        oven_dict = oven_prob[0] if isinstance(oven_prob, list) else oven_prob
        oven_values = [oven_dict.get(label, 0.0) for label in oven_labels]
        oven_bars = ax_oven.bar(
            range(len(oven_labels)), oven_values,
            color=oven_colors, width=0.6
        )
        for bar in oven_bars:
            height = bar.get_height()
            if height > 0:
                ax_oven.text(
                    bar.get_x() + bar.get_width()/2., height + max(oven_values)*0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold'
                )
        ax_oven.set_title(f'烘箱-{idx+1}', fontsize=10, fontweight='bold', pad=8)
        ax_oven.set_xticks(range(len(oven_labels)))
        ax_oven.set_xticklabels(oven_labels, rotation=45, ha='right', fontsize=8)
        ax_oven.set_ylim(0, 1.0)
        ax_oven.grid(axis='y', alpha=0.3)
        ax_oven.tick_params(axis='y', labelsize=9)

    # 烘箱分析图总标题
    fig.text(
        0.5, 0.62, '涂布机烘箱分析图',
        fontsize=16, fontweight='bold', ha='center', va='bottom'
    )

    # 整体标题 + 保存
    fig.suptitle('涂布机综合分析图', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(save_path)
    plt.close()
    #print(f"图片已保存至：{save_path}")


def generate_coater_diagnostic_report(head_prob_list, oven_prob_list):
    """
    生成涂布机综合诊断报告
    :param head_prob_list: 机头故障概率列表，格式如 [{"正常":0.21,"速度异常":0.11,"张力异常":0.67,"泵异常":0.01}]
    :param oven_prob_list: 烘箱故障概率列表，包含12组数据的嵌套列表
    :return: str - 完整的诊断报告文本
    """
    # ---------------------- 1. 基础信息初始化 ----------------------
    analysis_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_sep = "=" * 80
    section_sep = "-" * 80

    # ---------------------- 2. 解析机头数据 ----------------------
    head_prob_dict = head_prob_list[0] if isinstance(head_prob_list, list) else head_prob_list
    head_normal_prob = head_prob_dict.get("正常", 0.0)
    # 提取机头故障类型（排除"正常"）
    head_fault_items = {k: v for k, v in head_prob_dict.items() if k != "正常"}
    # 机头最大故障概率及类型
    head_max_fault_prob = max(head_fault_items.values()) if head_fault_items else 0.0
    head_max_fault_type = max(head_fault_items, key=head_fault_items.get) if head_fault_items else ""

    # ---------------------- 3. 解析烘箱数据 ----------------------
    oven_fault_details = []  # 存储异常烘箱信息 (烘箱编号, 故障类型, 故障概率)
    oven_normal_probs = []  # 存储各烘箱正常概率
    oven_max_fault_probs = []  # 存储各烘箱最大故障概率

    for idx, oven_item in enumerate(oven_prob_list):
        oven_idx = idx + 1  # 烘箱编号从1开始
        oven_prob_dict = oven_item[0] if isinstance(oven_item, list) else oven_item
        oven_normal_prob = oven_prob_dict.get("正常", 0.0)
        oven_normal_probs.append(oven_normal_prob)

        # 提取烘箱故障类型（排除"正常"）
        oven_fault_items = {k: v for k, v in oven_prob_dict.items() if k != "正常"}
        oven_max_fault_prob = max(oven_fault_items.values()) if oven_fault_items else 0.0
        oven_max_fault_type = max(oven_fault_items, key=oven_fault_items.get) if oven_fault_items else ""
        oven_max_fault_probs.append(oven_max_fault_prob)

        # 记录故障（故障概率 > 正常概率 或 故障概率 > 0.7）
        if oven_max_fault_prob > oven_normal_prob or oven_max_fault_prob > 0.7:
            oven_fault_details.append({
                "index": oven_idx,
                "fault_type": oven_max_fault_type,
                "fault_prob": oven_max_fault_prob,
                "normal_prob": oven_normal_prob
            })

    # 烘箱整体故障概率（所有烘箱最大故障概率的平均值）
    oven_avg_fault_prob = sum(oven_max_fault_probs) / len(oven_max_fault_probs) if oven_max_fault_probs else 0.0

    # ---------------------- 4. 健康状态评估 ----------------------
    # 状态判断规则：
    # - 严重：任意故障概率 > 0.7
    # - 预警：存在故障概率 > 正常概率
    # - 正常：所有故障概率 ≤ 正常概率 且 无严重故障
    status = "🟢 正常"
    status_desc = ""
    suggestion = ""

    # 检查严重故障
    is_severe = False
    # 机头严重故障
    if head_max_fault_prob > 0.7:
        is_severe = True
    # 烘箱严重故障
    for fault_prob in oven_max_fault_probs:
        if fault_prob > 0.7:
            is_severe = True
            break

    # 检查预警故障
    is_warning = False
    # 机头预警
    if head_max_fault_prob > head_normal_prob:
        is_warning = True
    # 烘箱预警
    for i in range(len(oven_normal_probs)):
        if oven_max_fault_probs[i] > oven_normal_probs[i]:
            is_warning = True
            break

    # 确定最终状态
    if is_severe:
        status = "🔴 严重"
        status_desc = "严重（存在故障概率大于0.7的情况）"
    elif is_warning:
        status = "🟡 预警"
        status_desc = "预警（存在故障概率大于正常概率的情况）"
    else:
        status_desc = "正常（正常概率值最大）"

    # 生成建议
    if status in ["🟡 预警", "🔴 严重"]:
        # 机头故障建议
        head_suggest = ""
        if head_max_fault_prob > head_normal_prob:
            head_suggest = f"机头{head_max_fault_type}可能性最大，建议优先检查该部位。"
        # 烘箱故障建议
        oven_suggest = ""
        if oven_fault_details:
            oven_fault_str = "、".join([f"烘箱{fault['index']}({fault['fault_type']})" for fault in oven_fault_details])
            oven_suggest = f"烘箱异常部位：{oven_fault_str}，建议重点排查这些烘箱的故障类型。"
        suggestion = head_suggest + oven_suggest
    else:
        # 正常状态的维护建议
        suggestion = """1. 机头：定期检查运行速度、张力值、泵转速等参数，保持润滑；
2. 烘箱：定期清理烘箱内部积尘，检查温度传感器和电机运行状态；
3. 整体：建议每周进行一次全面的设备巡检，确保各部位运行正常。"""

    # ---------------------- 5. 分析诊断 ----------------------
    diagnostic_content = ""
    if status in ["🟡 预警", "🔴 严重"]:
        # 机头诊断
        head_diagnostic = ""
        if head_max_fault_prob > head_normal_prob:
            head_diagnostic = f"机头故障：{head_max_fault_type}（概率{head_max_fault_prob:.2f}），"
            # 机头故障处理方法
            head_handle = {
                "速度异常": "检查电机转速传感器、调速器，清理传动部件积垢，校准速度参数。",
                "张力异常": "检查张力传感器、辊轴压力，调整放卷/收卷张力参数，更换磨损的张力辊。",
                "泵异常": "检查泵体压力、电机负载，清理泵腔杂质，更换密封件或轴承。"
            }
            head_diagnostic += head_handle.get(head_max_fault_type, "请检查该部位的传感器和执行机构。")

        # 烘箱诊断
        oven_diagnostic = ""
        if oven_fault_details:
            oven_diagnostic = "\n  烘箱故障：\n"
            oven_handle = {
                "烘箱温度异常": "检查温度传感器、加热管，清理通风口，校准温控器参数。",
                "发热包温度异常": "更换发热包或检查发热包供电线路，确保接触良好。",
                "电机温度异常": "检查电机散热风扇，清理电机积尘，测量电机绕组电阻，必要时更换电机。",
                "电机震动异常": "检查电机地脚螺栓是否松动，校准电机动平衡，更换磨损的轴承。",
                "电流电压异常": "检查供电线路电压稳定性，更换损坏的接触器或熔断器，校准电流传感器。"
            }
            for fault in oven_fault_details:
                oven_diagnostic += f"    烘箱{fault['index']}：{fault['fault_type']}（概率{fault['fault_prob']:.2f}）→ {oven_handle.get(fault['fault_type'], '请检查该烘箱的相关部件。')}\n"

        diagnostic_content = head_diagnostic + oven_diagnostic
    else:
        diagnostic_content = "设备各部位正常概率均为最大值，无明显故障风险，建议按计划进行常规维护。"

    # ---------------------- 6. 组装报告 ----------------------
    report = f"""
{report_sep}
涂布机综合诊断报告
{report_sep}
分析时间：{analysis_time}
分析部位：机头+12个烘箱

【预测统计】
{section_sep}
当前机头故障概率：
  正常：{head_normal_prob:.2f} | {', '.join([f'{k}：{v:.2f}' for k, v in head_fault_items.items()])}
当前烘箱故障概率：
  平均故障概率：{oven_avg_fault_prob:.2f} | 异常烘箱数量：{len(oven_fault_details)}

【健康状态评估】
{section_sep}
  状态: {status} （{status_desc}）
  建议: {suggestion}

【分析诊断】
{section_sep}
  {diagnostic_content}

{report_sep}
报告结束
{report_sep}
"""
    # 清理多余空行
    report = "\n".join([line.strip() for line in report.split("\n") if line.strip()])
    return report

def data(selected_jt_file,selected_hx_file,model_dir):
    #df_jt = pd.read_csv('../dataset/tujiaoji_com/jt/jt_2.csv')
    df_jt = pd.read_csv(selected_jt_file)
    prd_jt = predict_new_data(df_jt,model_path=model_dir,model_name='jt_rf_classifier.pkl',
                     scaler_name='jt_scaler.pkl', feature_cols='jt_feature_cols.txt')
    #print(prd_jt)

    #id_df_dict = split_csv_by_id('../dataset/tujiaoji_com/hx/hx_2.csv')
    id_df_dict = split_csv_by_id(selected_hx_file)
    hx_list=[]
    # 方式1：遍历字典，获取每个id和对应DataFrame
    for id_num, single_df in id_df_dict.items():
        id_num = f"{id_num:02d}"
        prd = predict_new_data(single_df,model_path=model_dir,model_name='hx_rf_classifier_'+id_num+'.pkl',
                     scaler_name='hx_scaler_'+id_num+'.pkl', feature_cols='hx_feature_cols.txt')
        hx_list.append(json.loads(prd))

    #print(hx_list)
    img_path=f'{BASE_DIR}/output/tujiaoji_com/'+uuid.uuid4().hex+'.png'
    plot_combined_analysis_chart(
        prob_list=json.loads(prd_jt),
        head_df=df_jt,
        oven_prob_list=hx_list,
        save_path=img_path
    )
    # 报告生成
    report = generate_coater_diagnostic_report(json.loads(prd_jt), hx_list)
    return img_path,report

def process_input(model_dir):
    """处理逻辑：获取选中的机头/烘箱文件和模型路径，后续可扩展"""
    global selected_jt_file, selected_hx_file
    # result = f"""
    # 选中的机头文件：{selected_jt_file or '未选择'}
    # 选中的烘箱文件：{selected_hx_file or '未选择'}
    # 选中的模型路径：{model_dir or '未选择'}
    # """
    img_path , result = data(selected_jt_file, selected_hx_file, model_dir)
    # 这里可替换为实际的处理逻辑，返回图片和文本结果
    return img_path, result


def set_selected_jt(file_path, buttons, file_paths):
    """更新机头文件选中状态"""
    global selected_jt_file
    selected_jt_file = file_path
    # 逐个更新按钮样式：选中的设为primary，其他为secondary
    return [gr.update(variant="primary" if fp == file_path else "secondary")
            for fp, btn in zip(file_paths, buttons)]


def set_selected_hx(file_path, buttons, file_paths):
    """更新烘箱文件选中状态"""
    global selected_hx_file
    selected_hx_file = file_path
    # 逐个更新按钮样式：选中的设为primary，其他为secondary
    return [gr.update(variant="primary" if fp == file_path else "secondary")
            for fp, btn in zip(file_paths, buttons)]


def create_interface():
    # 机头文件路径（jt子目录）
    cwru_dir_jt = os.path.join(BASE_DIR, "dataset", "tujiaoji_com", "jt")
    # 烘箱文件路径（hx子目录）
    cwru_dir_hx = os.path.join(BASE_DIR, "dataset", "tujiaoji_com", "hx")

    # 适配备选路径 - 机头
    if not os.path.exists(cwru_dir_jt):
        alt_paths = [
            os.path.join(f"{BASE_DIR}/dataset/tujiaoji_com/jt"),
            os.path.join(BASE_DIR, "./dataset/tujiaoji_com/jt"),
            os.path.join(BASE_DIR, "dataset/tujiaoji_com/jt"),
        ]
        for path in alt_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                cwru_dir_jt = abs_path
                break

    # 适配备选路径 - 烘箱
    if not os.path.exists(cwru_dir_hx):
        alt_paths = [
            os.path.join(f"{BASE_DIR}/dataset/tujiaoji_com/hx"),
            os.path.join(BASE_DIR, "./dataset/tujiaoji_com/hx"),
            os.path.join(BASE_DIR, "dataset/tujiaoji_com/hx"),
        ]
        for path in alt_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                cwru_dir_hx = abs_path
                break

    # 读取机头CSV文件
    preset_files_jt = {}
    if os.path.exists(cwru_dir_jt):
        for file_name in os.listdir(cwru_dir_jt):
            if file_name.endswith('.csv'):
                file_path = os.path.join(cwru_dir_jt, file_name)
                preset_files_jt[file_path] = f"📄 {file_name}"

    # 读取烘箱CSV文件
    preset_files_hx = {}
    if os.path.exists(cwru_dir_hx):
        for file_name in os.listdir(cwru_dir_hx):
            if file_name.endswith('.csv'):
                file_path = os.path.join(cwru_dir_hx, file_name)
                preset_files_hx[file_path] = f"📄 {file_name}"

    # 读取模型目录
    model_dir = os.path.join(BASE_DIR, "model", "tujiaoji_com")
    model_options = []
    if not os.path.exists(model_dir):
        alt_model_paths = [
            os.path.join(f"{BASE_DIR}/model/tujiaoji_com"),
            os.path.join("./model/tujiaoji_com"),
            os.path.join("model/tujiaoji_com"),
        ]
        for path in alt_model_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                model_dir = abs_path
                break

    if os.path.exists(model_dir):
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            if os.path.isdir(item_path):
                model_options.append((item, item_path))

    with gr.Blocks(title="涂布机综合诊断应用") as demo:
        gr.Markdown("# 🚀 涂布机综合诊断应用")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 选择机头测试文件")
                # 机头按钮（独立变量名，避免覆盖）
                jt_buttons = []
                jt_file_paths = list(preset_files_jt.keys())
                for file_path, display_text in preset_files_jt.items():
                    btn = gr.Button(display_text, variant="secondary", size="lg")
                    jt_buttons.append(btn)

                # 机头按钮绑定事件（使用partial或默认参数捕获正确的file_path）
                for i, file_path in enumerate(jt_file_paths):
                    # 关键：通过默认参数固定循环变量，避免闭包延迟绑定问题
                    def update_jt_btn(path=file_path):
                        return set_selected_jt(path, jt_buttons, jt_file_paths)

                    jt_buttons[i].click(
                        fn=update_jt_btn,
                        inputs=[],
                        outputs=jt_buttons
                    )

                gr.Markdown("### 选择烘箱测试文件")
                # 烘箱按钮（独立变量名）
                hx_buttons = []
                hx_file_paths = list(preset_files_hx.keys())
                for file_path, display_text in preset_files_hx.items():
                    btn = gr.Button(display_text, variant="secondary", size="lg")
                    hx_buttons.append(btn)

                # 烘箱按钮绑定事件
                for i, file_path in enumerate(hx_file_paths):
                    def update_hx_btn(path=file_path):
                        return set_selected_hx(path, hx_buttons, hx_file_paths)

                    hx_buttons[i].click(
                        fn=update_hx_btn,
                        inputs=[],
                        outputs=hx_buttons
                    )

                gr.Markdown("### 选择模型")
                model_dropdown = gr.Dropdown(
                    choices=model_options,
                    label="模型列表",
                    value=model_options[0][1] if model_options else ""
                )

                process_btn = gr.Button("处理", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 时序曲线图")
                plot_output = gr.Image(label="数据曲线", type="pil")

                gr.Markdown("### 处理结果")
                output_text = gr.Textbox(label="结果信息", lines=10, interactive=False)

        # 处理按钮事件（返回图片和文本）
        process_btn.click(
            fn=process_input,
            inputs=[model_dropdown],
            outputs=[plot_output, output_text]
        )

    return demo


def main():
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
    demo.launch(
        allowed_paths=[f'{BASE_DIR}/output'],
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )


if __name__ == "__main__":
    main()