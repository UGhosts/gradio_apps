# 风扇噪声分类训练脚本
# Fan Noise Classification Training Script

import os
import sys
import numpy as np
import pathlib
import zipfile
import shutil
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioTrainTest as aT

# 设置数据集路径
DATASET_PATH = pathlib.Path("d:/qiAi/gradio_apps/gradio_apps/model/fan_noise/dataset/")
OUTPUT_PATH = pathlib.Path("d:/qiAi/gradio_apps/gradio_apps/model/fan_noise/train/")

# 创建必要的目录
def create_directories():
    """创建训练所需的目录结构"""
    (OUTPUT_PATH / "data").mkdir(exist_ok=True)
    (OUTPUT_PATH / "models").mkdir(exist_ok=True)
    print("目录结构创建完成")

# 解压数据集
def extract_dataset():
    """解压数据集文件"""
    print("开始解压数据集...")
    print(f"数据集路径: {DATASET_PATH}")
    print(f"输出路径: {OUTPUT_PATH / 'data'}")
    
    # 列出所有zip文件
    zip_files = list(DATASET_PATH.glob("*.zip"))
    print(f"找到的zip文件: {[zf.name for zf in zip_files]}")
    
    # 假设数据集包含train.zip和test.zip
    for zip_file in zip_files:
        if zip_file.exists():
            print(f"解压 {zip_file.name}...")
            print(f"zip文件大小: {zip_file.stat().st_size} 字节")
            try:
                with zipfile.ZipFile(str(zip_file), 'r') as zip_ref:
                    # 打印zip文件内容
                    zip_contents = zip_ref.namelist()
                    print(f"zip文件包含 {len(zip_contents)} 个文件/目录")
                    if len(zip_contents) > 0:
                        print(f"前5个内容: {zip_contents[:5]}")
                    
                    zip_ref.extractall(str(OUTPUT_PATH / "data"))
                print(f"{zip_file.name} 解压完成")
                
                # 检查解压后的内容
                extracted_contents = list((OUTPUT_PATH / "data").glob("*"))
                print(f"解压后目录包含 {len(extracted_contents)} 个文件/目录")
                if len(extracted_contents) > 0:
                    print(f"解压后的内容: {[item.name for item in extracted_contents]}")
            except Exception as e:
                print(f"解压 {zip_file.name} 时出错: {e}")
                import traceback
                traceback.print_exc()
    
    # 检查是否存在data/train/abnormal和data/train/normal目录结构
    train_abnormal_path = OUTPUT_PATH / "data" / "train" / "abnormal"
    train_normal_path = OUTPUT_PATH / "data" / "train" / "normal"
    
    if train_abnormal_path.exists() and train_normal_path.exists():
        print("检测到data/train/abnormal和data/train/normal目录结构")
        print("将abnormal和normal目录移动到data目录下")
        
        # 将abnormal和normal目录移动到data目录下
        import shutil
        if (OUTPUT_PATH / "data" / "abnormal").exists():
            shutil.rmtree(str(OUTPUT_PATH / "data" / "abnormal"))
        if (OUTPUT_PATH / "data" / "normal").exists():
            shutil.rmtree(str(OUTPUT_PATH / "data" / "normal"))
        
        shutil.move(str(train_abnormal_path), str(OUTPUT_PATH / "data"))
        shutil.move(str(train_normal_path), str(OUTPUT_PATH / "data"))
        
        # 删除空的train目录
        if (OUTPUT_PATH / "data" / "train").exists() and not list((OUTPUT_PATH / "data" / "train").iterdir()):
            (OUTPUT_PATH / "data" / "train").rmdir()
            print("删除空的train目录")
        
        print("目录结构调整完成")
    
    print("所有数据集解压完成")

# 转换音频格式为单声道
def convert_to_mono():
    """将多声道音频转换为单声道"""
    print("开始转换音频格式...")
    fs = 16000  # 采样率
    num_channels = 1  # 单声道
    
    from pydub import AudioSegment
    
    # 获取所有需要转换的.wav文件
    audio_files = list((OUTPUT_PATH / "data").rglob("*.wav"))
    print(f"找到 {len(audio_files)} 个音频文件")
    
    for audio_file in audio_files:
        # 跳过已经转换的文件（如果有）
        if "Fs16000_NC1" in str(audio_file):
            continue
        
        try:
            print(f"转换文件: {audio_file.name}")
            # 加载音频文件
            sound = AudioSegment.from_wav(str(audio_file))
            
            # 转换为单声道
            sound = sound.set_channels(num_channels)
            
            # 转换采样率
            sound = sound.set_frame_rate(fs)
            
            # 保存转换后的文件，覆盖原文件
            sound.export(str(audio_file), format="wav")
            print(f"{audio_file.name} 转换完成")
        except Exception as e:
            print(f"转换 {audio_file.name} 时出错: {e}")
    print("音频格式转换完成")

# 训练模型
def train_model():
    """训练分类模型"""
    print("开始训练模型...")
    
    # 检查训练数据目录结构
    train_path = OUTPUT_PATH / "data"
    if not train_path.exists():
        print("训练数据目录不存在，请检查数据集结构")
        return
    
    # 获取所有类别目录
    class_dirs = []
    expected_classes = ["abnormal", "normal"]
    for item in train_path.iterdir():
        if item.is_dir() and item.name in expected_classes:
            class_dirs.append(str(item))
    
    if not class_dirs:
        # 如果没有找到预期的类别，尝试获取所有子目录
        class_dirs = [str(item) for item in train_path.iterdir() if item.is_dir()]
        print(f"警告: 未找到预期的类别目录，使用所有子目录: {[os.path.basename(d) for d in class_dirs]}")
    
    if not class_dirs:
        print("未找到类别目录，请检查数据集结构")
        return
    
    # 按字母顺序排序类别目录（确保一致性）
    class_dirs.sort()
    print(f"发现类别: {[os.path.basename(d) for d in class_dirs]}")
    
    # 检查每个类别目录下的音频文件数量
    for class_dir in class_dirs:
        audio_files = list(pathlib.Path(class_dir).glob("*.wav"))
        print(f"类别 {os.path.basename(class_dir)} 包含 {len(audio_files)} 个音频文件")
        if len(audio_files) == 0:
            print(f"警告: 类别 {os.path.basename(class_dir)} 目录下没有音频文件")
    
    # 训练参数
    window_size = 1.0
    window_step = 1.0
    model_type = "randomforest"  # 可以选择 "svm", "randomforest", "gradientboosting" 或 "extratrees"
    model_name = f"fan_noise_{model_type}_model"
    
    print(f"使用 {model_type} 算法训练模型...")
    try:
        aT.extract_features_and_train(
            class_dirs, 
            window_size, 
            window_step, 
            aT.shortTermWindow, 
            aT.shortTermStep, 
            model_type, 
            str(OUTPUT_PATH / "models" / model_name), 
            False
        )
        print(f"模型训练完成，保存为: {model_name}")
        return model_name
    except Exception as e:
        print(f"训练模型时出错: {e}")
        return None

# 测试模型
def test_model(model_name, model_type="randomforest"):
    """测试训练好的模型"""
    print("开始测试模型...")
    
    # 检查测试数据目录
    test_path = OUTPUT_PATH / "data"
    
    # 检查是否有单独的测试目录
    if (OUTPUT_PATH / "data" / "test").exists():
        test_path = OUTPUT_PATH / "data" / "test"
    
    # 获取所有测试类别目录
    test_class_dirs = []
    for item in test_path.iterdir():
        if item.is_dir():
            test_class_dirs.append(str(item))
    
    if not test_class_dirs:
        print("未找到测试类别目录，跳过测试")
        return
    
    # 收集所有测试文件
    test_files = []
    for class_dir in test_class_dirs:
        for file in pathlib.Path(class_dir).glob("*.wav"):
            test_files.append(str(file))
    
    if not test_files:
        print("未找到测试文件，跳过测试")
        return
    
    print(f"找到 {len(test_files)} 个测试文件")
    
    # 评估模型
    correct = 0
    total = len(test_files)
    
    for test_file in test_files:
        try:
            # 获取真实标签
            file_name = os.path.basename(test_file)
            # 根据目录名确定真实标签
            if "abnormal" in test_file:
                real_label = 0
            elif "normal" in test_file:
                real_label = 1
            else:
                continue
            
            # 预测 - 处理可能的返回值格式
            result = aT.file_classification(test_file, str(OUTPUT_PATH / "models" / model_name), model_type)
            
            # 检查返回值格式
            if isinstance(result, tuple):
                if len(result) == 2:
                    # 预期格式：(prediction, probabilities)
                    prediction, probabilities = result
                else:
                    # 只使用第一个返回值作为预测结果
                    pred_label = int(result[0])
                    probabilities = [0.0] * 2
                    probabilities[pred_label] = 1.0
            else:
                # 如果是单个值，假设它是预测标签
                pred_label = int(result)
                probabilities = [0.0] * 2
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
            
            # 检查是否正确
            if pred_label == real_label:
                correct += 1
                print(f"✓ 文件 {file_name}: 真实标签={real_label}, 预测标签={pred_label}, 概率={probabilities[pred_label]:.4f}")
            else:
                print(f"✗ 文件 {file_name}: 真实标签={real_label}, 预测标签={pred_label}, 概率={probabilities}")
                
        except Exception as e:
            print(f"处理文件 {test_file} 时出错: {e}")
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0
    print(f"\n测试完成: 正确率 = {correct}/{total} = {accuracy:.4f}")

# 主函数
def main():
    print("=== 风扇噪声分类训练程序 ===")
    
    try:
        # 创建目录
        create_directories()
        
        # 清理data目录，以便重新解压
        data_path = OUTPUT_PATH / "data"
        if data_path.exists():
            print("清理现有的data目录...")
            shutil.rmtree(str(data_path))
            data_path.mkdir()
            print("data目录清理完成")
        
        # 解压数据集
        extract_dataset()
        
        # 转换音频格式
        convert_to_mono()
        
        # 训练模型
        model_name = train_model()
        
        # 测试模型
        if model_name:
            test_model(model_name)
        
        print("\n=== 训练程序执行完成 ===")
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()