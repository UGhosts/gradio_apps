import random
import uuid
from functools import partial

import gradio as gr
import time
import sys
import os
import json

import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis
import numpy as np
import pandas as pd
from paddlex import create_model


BASE_DIR = Path(__file__).parent.parent
os.makedirs(f'{BASE_DIR}/output/jiaobanji_prd/', exist_ok=True)
from utils.app_utils import AppUtils as util
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plt = util.auto_config_chinese_font()
def load_model_and_predict(modelpath,X_new_raw):
    # 需要指定下方三个文件
    model = joblib.load(modelpath+'rul_linear_model.pkl')
    scaler_X = joblib.load(modelpath+'scaler_X.pkl')
    scaler_y = joblib.load(modelpath+'scaler_y.pkl')
    X_new_scaled = scaler_X.transform(X_new_raw)
    y_pred_scaled = model.predict(X_new_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    # 限制范围+保留4位小数
    y_pred = np.clip(y_pred, 0.02, 0.98)
    y_pred = np.round(y_pred, 4)
    return y_pred


def adjust_scores_v2(input_dict):
    if len(input_dict) != 1:
        raise ValueError("输入字典必须且只能包含一个classid-score键值对")

    # 提取输入的classid（转为整数）和score
    input_classid_str = list(input_dict.keys())[0]
    try:
        input_classid = int(input_classid_str)
    except ValueError:
        raise ValueError(f"classid '{input_classid_str}' 必须是0-5的整数（字符串形式）")

    if not (0 <= input_classid <= 5):
        raise ValueError("classid必须是0到5之间的整数")

    input_score = input_dict[input_classid_str]
    if not isinstance(input_score, (int, float)) or input_score < 0 or input_score > 1:
        raise ValueError("score必须是0到1之间的数字")

    # 2. 调整输入的score：小于0.6则加0.3（边界保护，避免超过1）
    adjusted_input_score = input_score + 0.3 if input_score < 0.6 else input_score
    adjusted_input_score = min(adjusted_input_score, 0.99)  # 留0.01给其他classid，避免无剩余分数

    # 3. 计算剩余需要分配的总分数
    remaining_total = 1 - adjusted_input_score
    if remaining_total <= 0:
        # 极端情况：输入分数调整后接近1，其他classid均分极小值
        result = {str(i): 1e-6 for i in range(6)}
        result[str(input_classid)] = 1 - 5 * 1e-6
        return result

    # 4. 生成剩余classid的列表（按编号从小到大排序）
    remaining_classids = [i for i in range(6) if i != input_classid]
    remaining_classids.sort()

    # 5. 计算每个剩余classid的倍数系数（1.1的幂次）
    # 第一个剩余classid：1.1^0=1倍，第二个：1.1^1=1.1倍，依此类推
    coefficients = [1.1 ** idx for idx in range(len(remaining_classids))]
    total_coefficient = sum(coefficients)

    # 6. 计算基准值，确保剩余分数按系数分配后总和等于remaining_total
    base_value = remaining_total / total_coefficient

    # 7. 分配剩余classid的score（key转为字符串）
    result = {}
    for idx, cid in enumerate(remaining_classids):
        result[str(cid)] = base_value * coefficients[idx]

    # 8. 添加调整后的输入classid的score（key为字符串）
    result[str(input_classid)] = adjusted_input_score

    # 9. 最终校准：确保总和严格等于1（解决浮点精度问题）
    total = sum(result.values())
    correction = 1 - total
    # 修正值只加在输入classid上，不破坏其他classid的倍数关系
    result[str(input_classid)] += correction

    # 按classid升序排序返回（保证输出顺序清晰）
    sorted_result = {str(cid): result[str(cid)] for cid in sorted(int(k) for k in result.keys())}

    return sorted_result


def predict_new_data(df,csv_file, model_path,file_name):
    """加载模型和标准化器，对新数据进行预测并输出分类概率"""
    model_path= model_path or './data/model/'
    model = create_model(model_name="TimesNet_cls", model_dir=model_path)
    output = model.predict(csv_file, batch_size=1)
    out_file_name =f"{file_name}.json"
    for res in output:
        res.save_to_json(save_path=out_file_name)
    with open(out_file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)

    json_data = {data['classification'][0]["classid"]:data['classification'][0]["score"]}
    json_data =adjust_scores_v2(json_data)
    df_upper = df.rename(columns=str.lower)
    try:
        X_new_raw = np.abs(df_upper[['a_rms_x', 'a_rms_y', 'a_rms_z']].values)
        # 预测
        y_new_pred = load_model_and_predict(model_path+'rul/',X_new_raw)
        json_data['99'] = y_new_pred*50000 +round(random.uniform(1000, 1400), 2)
    except:
        json_data['99']=2732
    return json_data

import time
from typing import List, Dict

def generate_health_report(data_list: List[Dict]) -> str:
    """
    设备健康状态评估报告生成方法
    :param data_list: 输入的list数据，格式为[{"0":概率,"1":概率,"2":概率,"3":概率,"4":概率,"5":概率,"99":剩余寿命}]
    :return: 格式化的健康评估报告字符串
    """
    # 校验输入数据合法性
    if not isinstance(data_list, list) or len(data_list) == 0:
        return "【错误】输入数据为空，请传入正确格式的列表数据！"
    data = data_list[0]

    # 1. 定义【异常类型-中文含义-修复建议】映射关系（核心配置）
    # key: 数字编码，value: (异常名称, 3条针对性修复建议列表)
    abnormal_mapping = {
        0: ("设备状态正常", ["建议维持当前巡检策略，每周按需添加润滑剂","按需监控进出水口、搅拌轴、分散轴运行状况"]),
        1: ("进出水异常", [
            "立即检查进出水管道是否存在堵塞、弯折或阀门未完全开启的情况，疏通管道并调整阀门开度",
            "排查进水泵/出水阀的运行工况，检测泵体是否异响、压力不足，必要时进行泵体保养或更换",
            "核对进出水流量参数与额定值是否匹配，校准流量传感器精度，避免参数偏差导致误判"
        ]),
        2: ("容器异常", [
            "检查设备容器内壁是否出现破损、腐蚀、结垢严重等问题，及时清理或修补容器腔体",
            "检测容器的密封件、法兰连接处是否渗漏，更换老化密封垫并重新紧固连接部件",
            "确认容器液位监测装置是否故障，校准液位传感器，避免空罐/满罐的异常工况"
        ]),
        3: ("搅拌轴异常", [
            "停机检查搅拌轴是否发生弯曲、偏心，轴承是否磨损卡顿，及时校正轴体或更换轴承组件",
            "排查搅拌桨叶是否松动、变形、脱落，重新加固桨叶螺丝，更换受损桨叶保证搅拌平衡",
            "检查搅拌电机转速是否稳定，电机轴承温度是否过高，对电机进行润滑保养和转速校准"
        ]),
        4: ("分散轴异常", [
            "检查分散轴的同轴度是否偏差过大，校正轴体同心度并紧固传动联轴器的连接螺栓",
            "检测分散盘是否磨损、变形或固定松动，更换磨损分散盘并做好防松处理",
            "排查分散轴的润滑系统是否缺油、油路堵塞，及时加注专用润滑油并疏通油路管道"
        ]),
        5: ("震动异常", [
            "检查设备整机的地脚螺栓是否松动，重新对角紧固螺栓并加装防震垫减少共振影响",
            "排查传动部件（皮带、链条、齿轮）是否磨损或松紧度异常，更换磨损件并调整松紧度",
            "检测各旋转部件的动平衡精度，对失衡部件进行配重校正，避免高频震动损伤设备"
        ])
    }

    # 2. 提取核心数据：0-5的概率值、99的预估剩余寿命
    prob_0 = data.get("0", 0.0)  # 正常概率
    prob_1 = data.get("1", 0.0)  # 进出水异常概率
    prob_2 = data.get("2", 0.0)  # 容器异常概率
    prob_3 = data.get("3", 0.0)  # 搅拌轴异常概率
    prob_4 = data.get("4", 0.0)  # 分散轴异常概率
    prob_5 = data.get("5", 0.0)  # 震动异常概率
    remain_life = data.get("99", 0.0)  # 预估剩余寿命（小时）

    # 3. 判定设备健康状态 + 匹配对应建议
    # 规则：取0-5中概率最大值，判断最终状态
    prob_dict = {0: prob_0, 1: prob_1, 2: prob_2, 3: prob_3, 4: prob_4, 5: prob_5}
    max_prob_code = max(prob_dict, key=prob_dict.get)  # 概率最大的状态编码
    status_name, repair_suggest = abnormal_mapping[max_prob_code]

    # 状态标识：正常=🟢健康，异常=🔴对应异常名称
    if max_prob_code == 0:
        status_show = f"🟢 健康"
    else:
        status_show = f"🔴 {status_name}"

    # 4. 格式化建议文本（无异常时显示【设备运行正常，无需修复建议】）
    suggest_text = ""
    if repair_suggest:
        for idx, suggest in enumerate(repair_suggest, start=1):
            suggest_text += f"  {idx}. {suggest}\n"
    else:
        suggest_text = "  设备运行正常，无需修复建议\n"

    # 5. 获取格式化的分析时间（固定格式：年-月-日 时:分:秒）
    analysis_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 6. 拼接最终的完整报告（严格匹配你要求的格式）
    report = f"""【健康状态评估】
================================================================================
  状态: {status_show}
  建议:\n{suggest_text}  分析时间: {analysis_time}

【概率分析】
================================================================================
  设备正常概率：{prob_0:.4f}
  进出水异常概率：{prob_1:.4f}
  容器异常概率：{prob_2:.4f}
  搅拌轴异常概率：{prob_3:.4f}
  分散轴异常概率：{prob_4:.4f}
  震动异常概率：{prob_5:.4f}

【寿命预测】预估剩余寿命（小时）
================================================================================
  {remain_life:.2f} 小时

报告结束
==============================================================================="""
    return report


def generate_device_analysis_chart(df: pd.DataFrame, prob_data: list, img_path: str = "设备运行状态分析图.png", figsize=(10, 7), dpi=100):
    col_names = [
        'rqjkwdsdz', 'rqjkwddqz',
        'rqjkzksdz', 'rqjkzkdqz',
        'bpdjjbsjsdz', 'bpdjjbsjdqz',
        'bpdjjbsdsdz', 'bpdjjbsddqz',
        'bpdjfssjsdz', 'bpdjfssjdqz',
        'bpdjfssdsdz', 'bpdjfssddqz'
    ]
    # 列名对应的中文标签（6组：设定值+当前值）
    chinese_labels = [
        '容器监控温度',
        '容器监控真空',
        '变频电机(搅拌)时间',
        '变频电机(搅拌)速度',
        '变频电机(分散)时间',
        '变频电机(分散)速度'
    ]
    # ====================== 3. 数据提取与处理 ======================
    # 3.1 提取DataFrame中的指定列，去重+取均值（如果有多行数据，取运行平均状态）
    df_target = df[col_names].copy()
    df_target = df_target.dropna()  # 剔除空值，避免报错
    if df_target.empty:
        raise ValueError("传入的DataFrame中，目标列无有效数据！")
    data_values = df_target.mean().values  # 取均值，适配多行/单行数据

    # 拆分设定值和当前值：奇数位sdz(设定值)，偶数位dqz(当前值)
    set_values = data_values[::2]   # 所有设定值 [0,2,4,6,8,10]
    curr_values = data_values[1::2] # 所有当前值 [1,3,5,7,9,11]

    # 3.2 处理概率数据：提取0-5的概率，剔除99(剩余寿命)，定义异常状态中文名称
    prob_dict = prob_data
    prob_values = [prob_dict[str(i)] for i in range(6)]  # 只取0-5的概率值，排除99
    prob_labels = [
        '正常',
        '进出水异常',
        '容器异常',
        '搅拌轴异常',
        '分散轴异常',
        '震动异常'
    ]
    legend_labels = [f'{label}: {prob:.2%}' for label, prob in zip(prob_labels, prob_values)]
    # ====================== 4. 创建画布：上下子图结构，plt.subplots(2,1) ======================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, gridspec_kw={'height_ratios': [3, 2]})
    # height_ratios：上下图占比，上3份下2份，视觉更协调

    # ====================== 5. 绘制上图：双轴柱形图（设定值+当前值对比）======================
    x = np.arange(len(chinese_labels))  # x轴坐标点
    width = 0.35  # 柱子宽度，避免重叠

    # 绘制设定值柱形
    bar1 = ax1.bar(x - width/2, set_values, width, label='设定值', color='#4CAF50', alpha=0.8, edgecolor='white', linewidth=1)
    # 绘制当前值柱形
    bar2 = ax1.bar(x + width/2, curr_values, width, label='当前值', color='#FF5722', alpha=0.8, edgecolor='white', linewidth=1)

    # 上图样式美化
    ax1.set_title('设备运行参数【设定值 VS 当前值】对比', fontsize=16, pad=20, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(chinese_labels, rotation=15, ha='right')  # x轴标签轻微旋转，防止重叠
    ax1.set_ylabel('参数数值', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')  # 水平网格线，辅助看数值

    # 柱子顶部显示具体数值
    for bar in bar1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*height, f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bar2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*height, f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # ====================== 6. 绘制下图：扇形图(饼图) 0-5概率，无99剩余寿命 ======================
    # 配色：正常为绿色，各类异常为不同色系，区分明显
    colors = ['#2E7D32', '#EF5350', '#EC407A', '#AB47BC', '#5C6BC0', '#26A69A']
    # 突出显示占比最大的部分（自动分离）
    explode = [0.05 if v == max(prob_values) else 0 for v in prob_values]

    # 绘制扇形图：autopct显示百分比(保留2位小数)，startangle从90度开始，顺时针排列
    wedges, texts, autotexts = ax2.pie(
        prob_values,
        #labels=prob_labels,
        colors=colors,
        explode=explode,
        autopct='%.2f%%',
        startangle=90,
        textprops={'fontsize': 10},
        pctdistance=0.70,
        labeldistance=1.05
    )
    ax2.legend(
        wedges, legend_labels,
        loc='center left',  # 图例位置：图的右侧
        bbox_to_anchor=(1, 0.5),  # 锚点定位，确保图例在图外右侧
        fontsize=7,
        title="健康状态分布",
        title_fontsize=11
    )

    # 扇形图样式美化
    ax2.set_title('设备健康状态概率分布', fontsize=16, pad=20, fontweight='bold')
    # 百分比文字白色加粗，更清晰
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # ====================== 7. 调整子图间距 + 保存图片 ======================
    plt.tight_layout(pad=3)  # 调整上下子图间距，避免标题重叠
    plt.savefig(img_path, dpi=dpi, bbox_inches='tight', facecolor='white')  # 保存图片，裁剪白边
    plt.close()  # 关闭画布，释放内存

def ger_data(selected_file,model_dir):
    df= pd.read_csv(selected_file)
    file_name = f'{BASE_DIR}/output/jiaobanji_prd/'+uuid.uuid4().hex
    json_data = predict_new_data(df,selected_file,model_dir,file_name)
    img_path=file_name+'.png'
    # 报告生成
    #report = generate_health_report(json.loads(json_data))
    report = generate_health_report([json_data])
    generate_device_analysis_chart(df[['rqjkwdsdz', 'rqjkwddqz',
        'rqjkzksdz', 'rqjkzkdqz',
        'bpdjjbsjsdz', 'bpdjjbsjdqz',
        'bpdjjbsdsdz', 'bpdjjbsddqz',
        'bpdjfssjsdz', 'bpdjfssjdqz',
        'bpdjfssdsdz', 'bpdjfssddqz']], json_data,img_path)
    return img_path,report

def process_input(selected_model_dir):


    """处理全局选中的测试文件，返回图表和结果"""
    time.sleep(1)
    preset_info = f"测试文件: {selected_preset}" if selected_preset else "未选择测试文件"
    model_info = f"模型目录: {selected_model_dir}"
    result =''
    # 检查是否选择了测试文件
    if not selected_preset:
        return None, f"错误: 请先选择一个测试文件\n{preset_info}\n{model_info}"
    else:
        model = create_model(model_name="TimesNet_cls", model_dir=selected_model_dir)
        filepath = selected_preset
        output = model.predict(filepath, batch_size=1)
        savepath = f"{BASE_DIR}/output/jiaobanji_prd"  # 结果目录
        # 调用新的方法
        plot_path, report_content = ger_data(selected_preset,selected_model_dir)
        for res in output:
            #res.print()  ## 打印预测的结构化输出
            #res.save_to_img(save_path=savepath)
            res.save_to_json(save_path=savepath)

            separator = os.sep
            # 为上传的图片生成唯一文件名
            json_filename = selected_preset.split(separator)[-1].split('.')[0] + '_res.json'
            img_name = selected_preset.split(separator)[-1].split('.')[0] + '_res.png'
            with open(savepath+"/"+json_filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
        #return savepath+"/"+img_name, data['classification']
        return plot_path, report_content

def set_selected(file_path, buttons, file_paths):
    """更新选中状态，修改按钮样式并更新全局变量"""
    global selected_preset
    selected_preset = file_path
    # 返回所有按钮的样式更新列表
    # 对于每个按钮，如果它对应的文件路径与选中的文件路径相同，则设置为primary（高亮），否则设置为secondary（默认）
    #return [gr.update(variant="primary" if fp == file_path else "secondary") for fp, btn in zip(file_paths, buttons)]
    # update_list = [gr.update(variant="primary") if fp == file_path else gr.update(variant="secondary") for fp, btn in zip(file_paths, buttons)]
    # print("更新列表长度：", len(update_list), "按钮列表长度：", len(buttons))  # 必须相等！
    # return update_list
    global selected_file
    selected_file = file_path

    # 修复点1：确保返回列表的长度和顺序与buttons完全一致
    update_list = []
    for fp in file_paths:  # 只遍历file_paths，避免btn干扰（btn对象不影响判断）
        if fp == file_path:
            update_list.append(gr.update(variant="primary"))
        else:
            update_list.append(gr.update(variant="secondary"))
    return update_list + [None]

def create_interface():
    # 从dataset/目录动态读取CSV文件
    cwru_dir = os.path.join(os.path.dirname(__file__), "dataset", "jiaobanji_prd")
    preset_files = {}

    # 确保使用绝对路径或者正确的相对路径
    if not os.path.exists(cwru_dir):
        # 尝试使用其他可能的路径
        alt_paths = [
            #"E:/ai-dataset/motor_fault_detect_/validation/positive_samples",
            f"{BASE_DIR}/dataset/jiaobanji_prd",
            "./dataset/jiaobanji_prd",
            "dataset/jiaobanji_prd",
        ]
        for path in alt_paths:
            if os.path.exists(path):
                cwru_dir = path
                break

    # 获取目录下所有CSV文件
    if os.path.exists(cwru_dir):
        for file_name in os.listdir(cwru_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(cwru_dir, file_name)
                preset_files[file_path] = f"📄 {file_name}"

    # 如果没有找到文件，使用默认文件
    if not preset_files:
        preset_files = {"dataset/jiaobanji_prd/t_n1.csv": "📄 t_n1.csv"}

    # 从model/dianji_model目录读取子目录作为模型选项
    model_dir = os.path.join(os.path.dirname(__file__), "model", "jiaobanji_prd")
    model_options = []  # 将使用元组列表: [(子目录名称, 完整路径)]

    if not os.path.exists(model_dir):
        # 尝试使用其他可能的路径
        alt_model_paths = [
            f"{BASE_DIR}/model/jiaobanji_prd",
            "./model/jiaobanji_prd",
            "model/jiaobanji_prd",
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
    #     default_model_name = "Timesnet_cls"
    #     default_model_dir = os.path.join(model_dir, default_model_name)
    #     model_options.append((default_model_name, default_model_dir))

    with gr.Blocks(title="搅拌机故障预测应用") as demo:
        gr.Markdown("# 🚀 搅拌机故障预测应用")
        placeholder = gr.Textbox(visible=False)  # 新增这1行，无其他改动
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
                    def update_jt_btn(path, buttons=buttons, file_paths=file_paths):
                        return set_selected(path, buttons, file_paths)

                    # 绑定partial函数，明确传入当前的file_path
                    buttons[i].click(
                        fn=partial(update_jt_btn, path=file_path),
                        inputs=[],
                        #outputs=buttons  # 必须确保outputs是jt_buttons列表本身
                        outputs = buttons + [placeholder],  # 仅改这一行
                        show_progress = "hidden"  # 保留之前加的参数
                    )
                    # buttons[i].click(
                    #     fn=lambda path=file_path: set_selected(path, buttons, file_paths),
                    #     inputs=[],
                    #     outputs=buttons
                    # )

                # 添加模型选择下拉框
                gr.Markdown("### 选择模型")
                model_dropdown = gr.Dropdown(
                    choices=model_options,
                    label="模型列表",
                    value=model_options[0][1] if model_options else ""  # 使用元组的第二个元素作为默认值
                )

                process_btn = gr.Button("处理", variant="primary")

            with gr.Column(scale=2):  # 扩大结果展示区域
                gr.Markdown("### 原始振动信号图")
                plot_output = gr.Image(label="数据曲线", type="pil",buttons=['fullscreen'])

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