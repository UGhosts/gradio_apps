import gradio as gr
import time
import os
import pandas as pd
from paddlex import create_model
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from PIL import Image
import sys
import csv
# 设置中文字体支持，确保负号能够正确显示
plt.rcParams["font.family"] = ["DejaVu Sans", "SimHei"]  # 优先使用能够正确显示负号的字体
# 全局变量记录选中的测试文件
selected_preset = None
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']  # 兼容不同系统的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

def draw_pic(selected_preset):


    # 读取两个CSV文件（请替换为你的文件路径）
    file1 = selected_preset.replace('\\','/')
    file2 = selected_preset.replace('dataset','output').replace('..','.').replace('.csv','_res.csv')
    save_pic_name = file2+'.png'

    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        pm1 = df1['pm']
        pm2 = df2['pm']

        combined_pm = pd.concat([pm1, pm2], ignore_index=True)
        #print(f"拼接后总数据量：{len(combined_pm)}（第一部分：{len(pm1)}条，第二部分：{len(pm2)}条）")

        x1 = np.arange(len(pm1))
        x2 = np.arange(len(pm1), len(combined_pm))

        plt.figure(figsize=(12, 6))  # 设置图表大小
        # 第一部分PM数据（蓝色）
        plt.plot(x1, pm1, color='#1f77b4', linewidth=2, label='原数据')
        # 第二部分PM数据（橙色，与第一部分区分）
        plt.plot(x2, pm2, color='#ff7f0e', linewidth=2, label='预测数据')

        plt.title('数据展示', fontsize=14, pad=20)  # 标题
        plt.xlabel('时序', fontsize=12)  # x轴标签
        plt.ylabel('值', fontsize=12)  # y轴标签
        plt.legend(fontsize=11)  # 图例（区分两部分数据）
        plt.grid(alpha=0.3, linestyle='--')  # 网格线（辅助查看）
        plt.tight_layout()  # 自动调整布局，防止标签被截断

        plt.savefig(
            save_pic_name,
            dpi=300,
            bbox_inches='tight'
        )
        # 显示图表
        #plt.show()
        return save_pic_name,file2
    except Exception as e:
        print(f"其他错误：{str(e)}")


def process_input(selected_model_dir):
    """处理全局选中的测试文件，返回图表和结果"""
    time.sleep(1)
    preset_info = f"测试文件: {selected_preset}" if selected_preset else "未选择测试文件"
    model_info = f"模型目录: {selected_model_dir}"
    # 检查是否选择了测试文件
    if not selected_preset:
        return None, f"错误: 请先选择一个测试文件\n{preset_info}\n{model_info}"
    else:
        model = create_model(model_name="DLinear", model_dir=selected_model_dir)
        output = model.predict(selected_preset, batch_size=1)


        # 保存预测结果并处理显示
        result_df = None
        for res in output:
            #res.print(json_format=True)
            res.save_to_csv(save_path="./output/wendu/")
        save_pic_name,csv_file = draw_pic(selected_preset)
        result=''
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # 逐行读取并打印，用制表符分隔字段
            for row in reader:
                result+=str(row) +'\n'
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
    cwru_dir = os.path.join(os.path.dirname(__file__), "dataset", "wendu")
    preset_files = {}

    # 确保使用绝对路径或者正确的相对路径
    if not os.path.exists(cwru_dir):
        # 尝试使用其他可能的路径
        alt_paths = [
            "../dataset/wendu",
            "./dataset/wendu",
            "dataset/wendu",
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

    model_dir = os.path.join(os.path.dirname(__file__), "model", "wendu")
    model_options = []  # 将使用元组列表: [(子目录名称, 完整路径)]

    if not os.path.exists(model_dir):
        # 尝试使用其他可能的路径
        alt_model_paths = [
            "../model/wendu",
            "./model/wendu",
            "model/wendu",
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
    if not model_options:
        default_model_name = "DLinear"
        default_model_dir = os.path.join(model_dir, default_model_name)
        model_options.append((default_model_name, default_model_dir))

    with gr.Blocks(title="温度时序预测应用") as demo:
        gr.Markdown("# 🚀 温度时序预测应用")

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
                gr.Markdown("### 时序曲线图")
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
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)


if __name__ == "__main__":
    main()