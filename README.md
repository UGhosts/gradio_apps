# Gradio应用框架

## 📖 项目介绍

这是一个基于Gradio的应用框架，用于快速部署和管理深度学习应用。该框架提供了统一的应用入口、灵活的模型管理和友好的用户界面，使开发者能够轻松构建和分享AI应用。

## 🚀 项目特点

- **统一入口**：通过`main.py`作为统一入口，可以方便地启动不同的应用
- **模块化设计**：应用、模型、数据集和文档分离，便于维护和扩展
- **灵活配置**：支持自定义端口、动态加载模型和数据集
- **可视化界面**：基于Gradio构建的交互式界面，支持实时数据处理和结果展示
- **多应用支持**：可以在同一个框架下管理多个不同的应用

## 📁 项目结构

```
gradio/
├── apps/                # 应用实现目录
│   └── cwru_cls.gradio.py  # CWRU轴承故障分类应用
├── dataset/             # 数据集目录
│   └── cwru_cls/        # CWRU轴承故障数据集
├── model/               # 模型存储目录
│   └── cwru_cls/        # CWRU轴承故障分类模型
├── output/              # 输出结果目录
│   └── cwru_cls/        # CWRU应用的输出结果
├── doc/                 # 文档目录
│   └── cwru_cls.md      # CWRU应用的详细说明文档
├── main.py              # 主入口文件
└── README.md            # 项目说明文档
```

## 🛠️ 环境要求

- CUDA 11.8 + cuDNN 8.9
- Python 3.10 
- Gradio
- PaddleX
- Pandas
- Matplotlib
- PIL (Pillow)

## 📦 安装指南

1. 克隆或下载项目代码

2. 安装基础依赖：
   ```bash
    conda env create -f environment_win.yml
   ```
3. 安装paddle相关依赖
    ```bash
      conda activate gradio
      pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
      git clone https://gitee.com/paddlepaddle/PaddleX.git
      cd PaddleX
      pip install -e ".[base]"
      paddlex --install PaddleTS
    ```
## 🚀 快速开始

### 通过主入口启动应用

```bash
python main.py <应用名> [端口号]
```

**示例**：
- 启动CWRU轴承故障分类应用（使用默认端口7860）：
  ```bash
  python main.py cwru_cls
  ```
- 指定端口启动应用：
  ```bash
  python main.py cwru_cls 8000
  ```

### 直接启动应用

```bash
python apps/<应用名>.gradio.py [端口号]
```

**示例**：
```bash
python apps/cwru_cls.gradio.py 7860
```

## 📱 应用功能说明

| 应用名 | 应用说明 | 默认端口 |
|-------|---------|---------|
| cwru_cls | 用于分析和识别轴承故障类型的工具，基于凯斯西储大学（Case Western Reserve University）的轴承故障数据集开发。详细说明请参考 <mcfile name="cwru_cls.md" path="doc/cwru_cls.md"></mcfile> | 7860 |


## 📚 文档说明

每个应用的详细使用说明存放在`doc/`目录下，文件名与应用名对应。例如，CWRU轴承故障分类应用的说明文档为`doc/cwru_cls.md`。

请参考相应的应用文档获取更详细的使用指南、数据格式要求和模型说明。

## 🔧 开发指南

### 添加新应用

1. 在`apps/`目录下创建新的应用文件，命名格式为`<应用名>.gradio.py`
2. 实现应用的核心功能，确保包含`launch_app()`或`main()`函数作为启动入口
3. 在`model/`目录下创建相应的模型文件夹，保存利用paddlex进行二次模型开发导出的模型文件
4. 在`dataset/`目录下准备示例数据集
5. 在`doc/`目录下编写应用说明文档

### 应用开发规范

- 应用应支持从命令行参数获取端口号
- 模型应存储在`model/<应用名>/`目录下，支持动态加载
- 数据集应存储在`dataset/<应用名>/`目录下
- 输出结果应保存到`output/<应用名>/`目录下
- 应用应提供清晰的用户界面和操作指引

## 🤝 贡献指南

欢迎对本项目进行贡献！如果你有任何建议或改进，请提交Issue或Pull Request。

## 📝 版本历史

- 初始版本：支持CWRU轴承故障分类应用，实现基本的模型加载、数据处理和结果展示功能

## 📄 许可证

本项目采用MIT许可证。

## 📧 联系我们

如有任何问题或建议，请联系项目维护者。