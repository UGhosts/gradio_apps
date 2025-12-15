from flask import Flask, request, render_template, redirect, url_for
import os
import sys
import numpy as np
import pyAudioAnalysis.audioTrainTest as aT
from pathlib import Path

# 设置基础路径和系统路径
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

# 创建Flask应用
app = Flask(__name__)
# 设置模板目录路径
templates_dir = os.path.join(BASE_DIR, "templates")
app.template_folder = templates_dir

# 定义模型目录和标签映射
MODEL_DIR = os.path.join(BASE_DIR, "model", "fan_noise", "train", "models")
LABEL_MAPPING = {0: "异常噪声", 1: "正常噪声"}

# 设置测试文件目录
TEST_FILES_DIR = os.path.join(BASE_DIR, "dataset", "fan_noise_test")
if not os.path.exists(TEST_FILES_DIR):
    os.makedirs(TEST_FILES_DIR)
    print(f"已创建测试文件目录: {TEST_FILES_DIR}")

# 加载可用模型
def load_models():
    models = []
    if os.path.exists(MODEL_DIR):
        for file in os.listdir(MODEL_DIR):
            if "model" in file and not file.endswith("MEANS"):
                models.append(file)
    return models

# 加载测试文件
def load_test_files():
    test_files = []
    if os.path.exists(TEST_FILES_DIR):
        for file in os.listdir(TEST_FILES_DIR):
            if file.endswith(".wav"):
                test_files.append(file)
    return test_files

# 预测函数
def predict(audio_path, model_name):
    try:
        # 确定模型类型
        model_type = ""
        if "svm" in model_name.lower():
            model_type = "svm"
        elif "randomforest" in model_name.lower():
            model_type = "randomforest"
        elif "gradientboosting" in model_name.lower():
            model_type = "gradientboosting"
        elif "extratrees" in model_name.lower():
            model_type = "extratrees"
        else:
            return "无法确定模型类型"
        
        # 执行预测 - 结合train.py中的方法和音频预处理
        model_path = os.path.join(MODEL_DIR, model_name)
        print(f"使用模型: {model_name}，类型: {model_type}")
        print(f"模型路径: {model_path}")
        
        # 音频预处理 - 修复energy_entropy函数的重塑错误
        from pyAudioAnalysis import audioBasicIO as aIO
        import numpy as np
        import wave
        
        # 读取音频文件
        [Fs, x] = aIO.read_audio_file(audio_path)
        # 转换为单通道
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        
        print(f"音频长度: {x.shape[0]} 采样点")
        print(f"采样率: {Fs} Hz")
        print(f"音频数据形状: {x.shape}")
        print(f"音频数据大小: {x.size}")
        print(f"音频数据类型: {x.dtype}")
        
        # 关键修复：强制将音频长度设置为1秒（8000采样点）
        # 这确保了short_window和short_step参数的正确计算
        target_length = 8000
        if x.size != target_length:
            if x.size < target_length:
                print(f"⚠️  音频太短 ({x.size} 采样点)，扩展到 {target_length} 采样点")
                x = np.pad(x, (0, target_length - x.size), 'constant')
            else:
                print(f"⚠️  音频太长 ({x.size} 采样点)，截断到 {target_length} 采样点")
                x = x[:target_length]
            print(f"✅ 已调整音频数据长度为: {x.size} 采样点")
        
        # 重新保存修改后的音频
        with wave.open(audio_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(Fs)
            wf.writeframes(x.astype(np.int16).tobytes())
        
        # 使用pyAudioAnalysis的分类函数进行预测
        result = aT.file_classification(audio_path, model_path, model_type)
        
        # 处理预测结果 - 与train.py中的处理方式保持一致
        pred_label = 0
        probabilities = [0.0, 0.0]
        
        if isinstance(result, tuple):
            if len(result) == 2:
                # 预期格式：(prediction, probabilities)
                prediction, probabilities = result
                pred_label = int(np.argmax(probabilities))
            else:
                # 只使用第一个返回值作为预测结果
                pred_label = int(result[0])
                probabilities[pred_label] = 1.0
        else:
            # 如果是单个值，假设它是预测标签
            pred_label = int(result)
            probabilities[pred_label] = 1.0
        
        # 确保probabilities是数组格式
        if not isinstance(probabilities, (list, np.ndarray)):
            # 如果是标量，转换为二元分类的概率数组
            prob_value = float(probabilities)
            if prob_value >= 0.5:
                pred_label = 1
                probabilities = [1.0 - prob_value, prob_value]
            else:
                pred_label = 0
                probabilities = [1.0 - prob_value, prob_value]
        else:
            # 如果已经是数组，使用argmax获取标签
            pred_label = int(np.argmax(probabilities))
        
        print(f"分类结果 - 类别: {pred_label}, 概率: {probabilities}")
        
        # 返回结果
        return {
            "label": LABEL_MAPPING[pred_label],
            "probabilities": {
                "异常噪声": round(float(probabilities[0]) * 100, 2),
                "正常噪声": round(float(probabilities[1]) * 100, 2)
            }
        }
        
    except Exception as e:
        import traceback
        print(f"❌ 预测出错: {str(e)}")
        traceback.print_exc()
        return f"预测出错：{str(e)}\n{traceback.format_exc()}"

# 主页面
@app.route('/')
def index():
    models = load_models()
    test_files = load_test_files()
    return render_template('index.html', models=models, test_files=test_files)

# 处理文件选择和预测
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'audio_file' not in request.form or request.form.get('model') is None:
        return redirect(url_for('index'))
    
    selected_file = request.form.get('audio_file')
    model_name = request.form.get('model')
    
    if selected_file == '':
        return redirect(url_for('index'))
    
    # 从固定目录获取文件
    file_path = os.path.join(TEST_FILES_DIR, selected_file)
    
    if os.path.exists(file_path) and file_path.endswith('.wav'):
        # 添加详细的调试信息
        print(f"\n=== 处理文件: {selected_file} (模型: {model_name}) ===")
        print(f"文件路径: {file_path}")
        print(f"文件大小: {os.path.getsize(file_path)} bytes")
        
        # 进行预测
        result = predict(file_path, model_name)
        
        models = load_models()
        test_files = load_test_files()
        return render_template('index.html', models=models, test_files=test_files, result=result, audio_name=selected_file)
    
    return redirect(url_for('index'))

# 创建HTML模板
@app.route('/template')
def get_template():
    return '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>吊扇故障诊断</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
        }
        .container {
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="file"], select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }
        .probability {
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>吊扇故障诊断</h1>
        <form method="post" action="/predict">
            <div class="form-group">
                <label for="audio_file">选择测试文件 (.wav)</label>
                <select id="audio_file" name="audio_file" required>
                    <option value="">-- 请选择文件 --</option>
                    {% for file in test_files %}
                    <option value="{{ file }}">{{ file }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="form-group">
                <label for="model">选择模型</label>
                <select id="model" name="model" required>
                    {% for model in models %}
                    <option value="{{ model }}">{{ model }}</option>
                    {% endfor %}
                </select>
            </div>
            <button type="submit">开始预测</button>
        </form>
        {% if result %}
        <div class="result">
            <h3>预测结果</h3>
            <p>音频文件: {{ audio_name }}</p>
            {% if result and result.label and result.probabilities %}
            <p>分类结果: {{ result.label }}</p>
            <div class="probability">
                <p>异常噪声概率: {{ result.probabilities['异常噪声'] }}%</p>
                <p>正常噪声概率: {{ result.probabilities['正常噪声'] }}%</p>
            </div>
            {% else %}
            <p>{{ result }}</p>
            {% endif %}
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

# 运行应用
if __name__ == "__main__":
    # 为了简化部署，我们将HTML直接嵌入到代码中
    # 创建templates目录
    templates_dir = os.path.join(BASE_DIR, "templates")
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    # 写入HTML模板
    with open(os.path.join(templates_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(app.view_functions['get_template']())
    
    print("Flask应用启动中...")
    print(f"访问地址: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)