<div align="center">

# MedSeg-UNet

### 基于改进 DoubleU-Net 的医学图像语义分割与 Flask 交互式分析平台

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-black.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-ISIC2018-orange.svg)](https://challenge.isic-archive.com/)

[**🇬🇧 English**](README.md) | **🇨🇳 简体中文**

</div>

---

## 📖 项目简介

**MedSeg-UNet** 是一个端到端的研究与工程项目，致力于解决**医学图像中病灶区域的语义分割**问题。项目将改进的深度学习分割网络与可投入使用的 Flask Web 平台相结合，使临床医生和研究人员能够上传皮肤镜图像、执行像素级分割、获取定量的形态学指标，并通过集成的讯飞星火大语言模型进行自然语言交互问答。

本项目在 **ISIC2018 皮肤病变数据集**上实现并对比了四种分割架构：

1. **U-Net** —— 经典的编码器-解码器基线模型。
2. **MultiResUNet** —— 引入多尺度残差块，适应目标尺寸差异较大的场景。
3. **DoubleU-Net** —— 堆叠两个 U-Net，采用 VGG-19 编码器、SE 注意力机制与 ASPP 空洞卷积。
4. **改进的 DoubleU-Net**（*本项目贡献*）—— 在 DoubleU-Net 基础上引入 **MultiResBlock** 与 **ResPath** 模块，扩大感受野、缩小编码器与解码器之间的语义鸿沟，并缓解深层网络的梯度衰减问题。

> 🎯 **核心结果：** 改进的 DoubleU-Net 在 **IoU 0.8306 / Dice 0.8951 / Recall 0.9267** 上取得最优，在临床最关键的指标上全面超越基线模型。

---

## ✨ 项目亮点

- 🔬 **四种分割模型对比实验**，统一的训练与评测流程。
- 🧠 **原创架构改进** —— 将 MultiResBlock 与 ResPath 注入 DoubleU-Net。
- 🖥️ **全栈 Flask Web 应用** —— 浏览器内完成上传、分割、可视化与分析。
- 🤖 **大模型智能助手** —— 集成讯飞星火大模型，支持对模型、指标、结果的自然语言问答。
- 📊 **完善的分割后分析** —— 病灶面积、边界长度、形态复杂度及运行时性能指标。
- 🗂️ **本地化会话管理** —— 自动保存历史记录、关键词全文检索高亮、重命名、删除。
- 🎨 **现代化 UI** —— 浅色/深色主题切换、响应式布局、悬停动画与动态反馈。

---

## 🧠 网络架构

### 基线模型：DoubleU-Net

DoubleU-Net 堆叠两个 U-Net 子网络：第一个执行粗分割，第二个进行细化增强。它采用预训练的 **VGG-19** 编码器、**SE（Squeeze-and-Excitation）注意力机制**与 **ASPP（空洞空间金字塔池化）**，增强多尺度特征建模能力。

### 本项目改进：MultiResBlock + ResPath

原始 DoubleU-Net 虽然强大，但存在以下问题：
- **固定感受野**（每层仅两个 3×3 卷积），难以同时捕捉细节与全局上下文。
- **语义鸿沟**：浅层编码器特征与深层解码器特征之间存在差距，尤其是首个 VGG-19 层与最深解码层拼接时。
- **通道冗余**：并非每条通道都同等重要。
- **梯度衰减**：深层堆叠架构中信息流动受阻。

针对上述问题，本项目引入：

| 模块 | 作用 |
|------|------|
| **MultiResBlock** | 仿 Inception 的并行多尺度卷积（3×3、5×5、7×7）配合残差连接，扩大感受野并捕捉多尺度特征。 |
| **ResPath** | 残差跳跃连接，在编码器特征与解码器特征融合前进行渐进式精炼，缩小语义鸿沟。 |

<div align="center">

```
<img width="623" height="360" alt="image" src="https://github.com/user-attachments/assets/8a399a13-8c37-4799-8535-d0e4b2606a5b" />
```

</div>

---

## 📊 实验结果

### 训练配置

| 参数 | 取值 |
|------|------|
| 图像尺寸 | 256 × 256 |
| 批大小 | 8 |
| 训练轮次 | 50 |
| 优化器 | Adam |
| 损失函数 | BCEWithLogitsLoss |
| 学习率 | 1e-4 |
| 权重衰减 | 1e-5 |
| 硬件 | NVIDIA GeForce RTX 3060 |
| 框架 | Python 3.8 + PyTorch |

### ISIC2018 上的定量对比

| 模型 | IoU ↑ | Dice ↑ | Accuracy ↑ | Precision ↑ | Recall ↑ |
|------|:-----:|:------:|:----------:|:-----------:|:--------:|
| U-Net | 0.7939 | 0.8648 | 0.9238 | 0.9127 | 0.8677 |
| MultiResUNet | 0.8015 | 0.8761 | 0.9259 | 0.8743 | 0.9157 |
| DoubleU-Net | 0.8269 | 0.8944 | **0.9412** | 0.8939 | 0.9248 |
| **改进的 DoubleU-Net（本项目）** | **0.8306** | **0.8951** | 0.9354 | **0.8973** | **0.9267** |

> 💡 改进模型在 **IoU、Dice、Precision、Recall** 上取得最佳成绩——这些指标对临床病灶定位与边界勾画最为关键。准确率略有下降是一种刻意的权衡：模型对前景病灶区域更加敏感专注，代价是少量背景像素被误判。

### 数据增强策略

| 增强方式 | 取值 |
|----------|------|
| RandomHorizontalFlip（水平翻转） | p = 0.5 |
| RandomVerticalFlip（垂直翻转） | p = 0.5 |
| RandomRotation（随机旋转） | ±30° |
| ColorJitter（颜色扰动） | brightness=0.2, contrast=0.2, saturation=0.2 |
| Normalize（标准化） | mean=[0.708, 0.582, 0.536], std=[0.157, 0.166, 0.181] |

> 📌 标准化统计量是**基于 ISIC2018 训练集重新计算**的，而非沿用 ImageNet 预设值。由于皮肤镜图像色彩分布较窄，自定义归一化显著提升了收敛速度与训练稳定性。

---

## 🖥️ 系统架构

平台采用**前后端分离**架构：

```
┌──────────────────────────────────────────────────────────────┐
│                       浏览器（前端）                          │
│   HTML + CSS + JavaScript   │   浅色/深色主题   │   localStorage
│   • 图像上传                │   • 聊天界面       │   • 历史记录
│   • 模型选择                │   • 结果渲染       │   • 搜索
└──────────────────┬───────────────────────────────────────────┘
                   │  HTTP / JSON
                   ▼
┌──────────────────────────────────────────────────────────────┐
│                     Flask 后端                                │
│   /predict  →  图像预处理 → 模型推理 → 生成掩膜                │
│   /chat     →  调用星火大模型 API → 自然语言回复               │
│   /metrics  →  形态学与性能分析                                │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│              PyTorch 模型库（4 个模型）                        │
│   U-Net  │  MultiResUNet  │  DoubleU-Net  │  改进的 DoubleU-Net │
└──────────────────────────────────────────────────────────────┘
```

### 核心功能模块

| 模块 | 描述 |
|------|------|
| **图像分割** | 上传皮肤镜图像，选择模型，获取原图、二值掩膜与边界叠加可视化。 |
| **定量分析** | 基于二值掩膜计算病灶覆盖率、面积、边界长度与形态复杂度。 |
| **性能分析** | 通过 `psutil` 监控模型加载耗时、推理耗时、总处理时间与内存占用。 |
| **星火大模型助手** | 以自然语言提问模型架构、指标含义、数据集背景与分割结果解读。 |
| **会话管理** | 自动按时间戳保存历史，支持关键词全文检索高亮、重命名、删除与上下文预览。 |
| **UI/UX** | 浅色/深色主题切换、响应式布局、可折叠侧边栏、悬停动画与动态按钮反馈。 |

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 推荐 Python 3.8+
git clone https://github.com/your-username/MedSeg-UNet.git
cd MedSeg-UNet

# 创建虚拟环境
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据集

从 [ISIC Archive](https://challenge.isic-archive.com/landing/2018/) 下载 **ISIC2018 Task 1（病灶边界分割）**，按如下结构组织：

```
data/
├── ISIC2018_Task1-2_Training_Input/        # 2594 张原始图像
├── ISIC2018_Task1_Training_GroundTruth/     # 2594 张真实标注掩膜
├── ISIC2018_Task1-2_Validation_Input/      # 100 张验证图像
└── ISIC2018_Task1_Validation_GroundTruth/   # 100 张验证掩膜
```

### 3. 训练模型

```bash
python train.py \
    --model improved_doubleunet \
    --data_dir ./data \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --image_size 256 \
    --output ./checkpoints
```

### 4. 启动 Web 系统

```bash
# 配置讯飞星火 API 密钥
export SPARK_API_KEY="your_api_key_here"
export SPARK_API_SECRET="your_api_secret_here"

# 启动 Flask 服务
python app.py
# 浏览器访问 http://127.0.0.1:5000
```

---

## 📁 项目结构

```
MedSeg-UNet/
├── models/                         # 网络定义
│   ├── unet.py
│   ├── multiresunet.py
│   ├── doubleunet.py
│   └── improved_doubleunet.py     # 本项目改进架构
├── modules/
│   ├── multires_block.py           # MultiResBlock
│   └── respath.py                  # ResPath
├── data/                            # 数据集（已 gitignore）
├── utils/
│   ├── dataset.py                  # PyTorch Dataset 与数据增强
│   ├── metrics.py                  # IoU、Dice、Precision、Recall
│   └── visualize.py                # 边界叠加与区域着色可视化
├── checkpoints/                    # 训练权重
├── static/                         # 前端资源（CSS/JS/图像）
├── templates/                      # HTML 模板
├── train.py                        # 训练入口
├── predict.py                      # 推理工具
├── app.py                          # Flask 应用
├── spark_client.py                 # 讯飞星火大模型客户端
├── requirements.txt
├── README.md                        # 英文版
└── README.zh-CN.md                  # 中文版（本文件）
```

---

## 📈 评价指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **IoU** | TP / (TP + FP + FN) | 预测区域与真实区域的重叠度，对小目标敏感。 |
| **Dice** | 2TP / (2TP + FP + FN) | 类 F1 的重叠度，对小面积病灶尤为敏感。 |
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | 整体正确率，在不均衡数据上存在偏倚。 |
| **Precision** | TP / (TP + FP) | 正类预测的可信度（低误报）。 |
| **Recall** | TP / (TP + FN) | 真实前景的覆盖率（低漏检）。 |

---

## 🩺 应用场景

- **临床初筛** —— 为皮肤科医生自动勾画病灶边界并量化形态学指标。
- **科研与模型评测** —— 提供可复现的平台，以标准化指标评估新分割架构。
- **医学教育** —— 交互式教学工具，学生上传病例、查看分析、向大模型提问概念。
- **远程辅助诊疗** —— 未来可接入远程医疗平台，实现患者自助筛查与医生远程复核。

---

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| 深度学习 | PyTorch、torchvision |
| 后端 | Flask、Gunicorn（生产环境） |
| 前端 | HTML5、CSS3、原生 JavaScript |
| 大语言模型 | 讯飞星火 Web API |
| 系统监控 | psutil |
| 可视化 | Matplotlib、OpenCV |
| 数据集 | ISIC2018 |

---

## 📚 参考文献

本项目基于以下奠基性工作：

1. Ronneberger 等，《U-Net: Convolutional Networks for Biomedical Image Segmentation》，MICCAI 2015。
2. Ibtehaz & Rahman，《MultiResUNet: Rethinking the U-Net Architecture for Multimodal Biomedical Image Segmentation》，Neural Networks 2020。
3. Jha 等，《Double U-Net: A Deep Convolutional Neural Network for Medical Image Segmentation》，IEEE CBMS 2020。
4. Zhou 等，《UNet++: Redesigning Skip Connections to Exploit Multiscale Features》，IEEE TMI 2020。
5. Chen 等，《DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs》，IEEE TPAMI 2018。

完整参考文献列表见配套论文文档。

---

## 📄 开源协议

本项目基于 **MIT 协议**开源 —— 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 感谢 **ISIC Archive** 提供公开的皮肤镜图像数据集。
- 感谢 **PyTorch**、**Flask**、**OpenCV** 等开源社区。
- 感谢 U-Net、MultiResUNet、DoubleU-Net 的作者提供基础架构。
- 感谢**讯飞**提供星火大语言模型 API。

---

<div align="center">

**⭐ 如果本项目对您有帮助，欢迎 Star 支持！**

*MedSeg-UNet —— 连接深度学习研究与临床级医学图像分析。*

</div>
