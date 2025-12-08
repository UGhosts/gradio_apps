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

# 设置上传文件的目录
UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制

# 加载可用模型
def load_models():
    models = []
    if os.path.exists(MODEL_DIR):
        for file in os.listdir(MODEL_DIR):
            if "model" in file and not file.endswith("MEANS"):
                models.append(file)
    return models

# 预测函数
def predict(audio_path, model_name):
    try:
        import pickle
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
        
        # 执行预测
        model_path = os.path.join(MODEL_DIR, model_name)
        
        # 先获取音频的特征，看看发生了什么
        from pyAudioAnalysis import ShortTermFeatures as aF
        from pyAudioAnalysis import audioBasicIO as aIO
        
        # 读取音频文件
        [Fs, x] = aIO.read_audio_file(audio_path)
        # 转换为单通道
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        
        # 添加详细的音频数据调试信息
        print(f"音频长度: {x.shape[0]} 采样点")
        print(f"采样率: {Fs} Hz")
        print(f"音频数据形状: {x.shape}")
        print(f"音频数据大小: {x.size}")
        print(f"音频数据类型: {x.dtype}")
        
        # 检查是否有6400大小的数组 - 这是用户报告的问题
        if x.size == 6400:
            print("⚠️  检测到大小为6400的音频数据，这会导致重塑错误")
            
            # 关键修复：将音频数据扩展到8000个采样点（添加1600个零）
            # 这样可以确保特征提取时能生成足够的时间帧
            x = np.pad(x, (0, 8000 - x.size), 'constant')
            print(f"✅ 已将音频数据扩展到: {x.size} 采样点")
        
        # 计算特征（与模型训练时使用的参数一致）
        win, step = 0.050, 0.025  # 50ms窗口，25ms步长
        win_size = int(win * Fs)
        step_size = int(step * Fs)
        
        # 确保窗口大小和步长有效
        if x.size < win_size:
            print(f"❌ 音频太短，无法提取特征: {x.size} < {win_size}")
            return "音频文件太短，无法进行分析"
        
        # 提取特征
        [f, fn] = aF.feature_extraction(x, Fs, win_size, step_size)
        print(f"特征提取结果 - 形状: {f.shape}, 特征数: {f.shape[0]}, 时间帧: {f.shape[1]}")
        
        # 计算特征统计值（均值和标准差）
        # 这与模型训练时使用的特征处理方式一致
        feature_vector = []
        for i in range(f.shape[0]):
            feature_vector.append(np.mean(f[i, :]))
            feature_vector.append(np.std(f[i, :]))
        
        feature_vector = np.array(feature_vector)
        print(f"特征向量 - 形状: {feature_vector.shape}, 大小: {feature_vector.size}")
        
        # 加载模型并执行分类
        try:
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
            
            # 检查特征数量是否匹配
            expected_features = model.n_features_in_
            print(f"模型期望特征数: {expected_features}")
            print(f"实际特征数: {feature_vector.size}")
            
            if feature_vector.size != expected_features:
                print(f"❌ 特征数量不匹配: {feature_vector.size} != {expected_features}")
                return "特征数量不匹配，模型可能需要更新"
            
            # 执行预测
            prediction = model.predict(feature_vector.reshape(1, -1))[0]
            probabilities = model.predict_proba(feature_vector.reshape(1, -1))[0]
            
            print(f"分类结果 - 类别: {prediction}, 概率: {probabilities}")
        except Exception as model_error:
            print(f"❌ 模型加载或预测出错: {str(model_error)}")
            import traceback
            traceback.print_exc()
            
            # 回退到使用pyAudioAnalysis的分类函数，但先确保音频足够长
            if x.size < 8000:
                print("❌ 回退到pyAudioAnalysis分类前，再次扩展音频")
                x = np.pad(x, (0, 8000 - x.size), 'constant')
                print(f"✅ 已将音频数据扩展到: {x.size} 采样点")
                
                # 重新保存修改后的音频
                import wave
                with wave.open(audio_path, 'w') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(Fs)
                    wf.writeframes(x.astype(np.int16).tobytes())
            
            print("回退到使用pyAudioAnalysis的分类函数")
            result = aT.file_classification(audio_path, model_path, model_type)
            
            # 处理预测结果
            pred_label = 0
            probabilities = [0.0, 0.0]
            
            if isinstance(result, tuple):
                if len(result) == 2:
                    prediction, probabilities = result
                    pred_label = int(np.argmax(probabilities))
                else:
                    pred_label = int(result[0])
                    probabilities[pred_label] = 1.0
            else:
                pred_label = int(result)
                probabilities[pred_label] = 1.0
            
            prediction = pred_label
        
        # 返回结果
        return {
            "label": LABEL_MAPPING[prediction],
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
    return render_template('index.html', models=models)

# 处理文件上传和预测
@app.route('/predict', methods=['POST'])
def predict_route():
    if 'audio' not in request.files or request.form.get('model') is None:
        return redirect(url_for('index'))
    
    audio = request.files['audio']
    model_name = request.form.get('model')
    
    if audio.filename == '':
        return redirect(url_for('index'))
    
    if audio and audio.filename.endswith('.wav'):
        # 保存文件
        filename = audio.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio.save(file_path)
        
        # 添加详细的调试信息
        print(f"\n=== 处理文件: {filename} (模型: {model_name}) ===")
        print(f"文件大小: {os.path.getsize(file_path)} bytes")
        
        # 进行预测
        result = predict(file_path, model_name)
        
        # 删除临时文件
        os.remove(file_path)
        
        models = load_models()
        return render_template('index.html', models=models, result=result, audio_name=filename)
    
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
    <title>风扇噪声分类系统</title>
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
        <h1>风扇噪声分类系统</h1>
        <form method="post" action="/predict" enctype="multipart/form-data">
            <div class="form-group">
                <label for="audio">上传音频文件 (.wav)</label>
                <input type="file" id="audio" name="audio" accept=".wav" required>
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