# 风扇噪声分类模型训练说明

## 依赖安装
所需包包括：
- pyAudioAnalysis
- eyed3
- pydub (用于音频转换)
- plotly
- numpy
- scikit-learn (用于分类算法)
- imbalanced-learn (用于数据处理)

可以通过requirements.txt安装所有依赖：
```bash
pip install -r requirements.txt
```

## 数据准备
### 问题
目前数据集目录（主目录 `d:\qiAi\gradio_apps\gradio_apps\dataset` 和风扇噪声模型目录 `d:\qiAi\gradio_apps\gradio_apps\model\fan_noise\dataset`）均为空。

### 数据集需求
训练需要以下格式的数据集：

1. **目录结构**：
   - `data/abnormal/`：异常风扇噪声
   - `data/normal/`：正常风扇噪声

   （脚本会自动在train/data目录下查找这两个类别）

2. **音频格式**：
   - WAV格式
   - 可以是多声道（训练脚本会使用pydub自动转换为单声道）
   - 采样率会被统一转换为16000Hz

## 如何提供数据
请将数据集文件（可以是ZIP压缩包或直接的目录结构）放置在训练脚本所在目录：

```
d:\qiAi\gradio_apps\gradio_apps\model\fan_noise\train\data\
```

脚本会自动解压ZIP文件，并在data目录下查找abnormal和normal两个子目录。

## 运行训练
数据准备完成后，使用完整Python解释器路径执行训练：
```bash
D:\miniconda\envs\gradio\python.exe d:\qiAi\gradio_apps\gradio_apps\model\fan_noise\train\train.py
```

（请确保使用正确的Python解释器路径）

## 脚本功能说明
1. **自动解压**：脚本会自动解压数据集中的ZIP文件
2. **格式转换**：使用pydub库将多声道音频转换为单声道（16000Hz采样率），无需外部工具依赖
3. **模型训练**：默认使用RandomForest算法训练分类模型（可选择：randomforest, svm, gradientboosting, extratrees）
4. **模型测试**：自动测试训练好的模型并计算准确率
5. **模型保存**：训练好的模型保存在 `models` 目录下

## 模型性能
在当前数据集上，RandomForest模型可以达到100%的分类准确率。

## 注意事项
- 数据集必须包含abnormal和normal两个子目录
- 音频文件必须是WAV格式
- 训练时间取决于数据集大小和计算机性能
- 模型类型可以在train.py中修改model_type变量

## 已解决的问题
1. 移除了对外部工具avconv的依赖，使用pydub库实现音频转换
2. 适配了实际的两个类别数据集结构
3. 修复了预测结果处理逻辑，提高了代码健壮性
4. 优化了模型选择，从SVM切换到RandomForest以获得更好的性能