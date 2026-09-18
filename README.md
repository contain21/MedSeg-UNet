<div align="center">

# MedSeg-UNet

### An Improved DoubleU-Net for Medical Image Semantic Segmentation with a Flask-Based Interactive Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-black.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-ISIC2018-orange.svg)](https://challenge.isic-archive.com/)

**🇬🇧 English** | [**🇨🇳 简体中文**](README.zh-CN.md)

</div>

---

## 📖 Overview

**MedSeg-UNet** is an end-to-end research and engineering project dedicated to solving the problem of **semantic segmentation of lesion regions in medical images**. The project combines an improved deep-learning segmentation network with a production-ready Flask web platform, enabling clinicians and researchers to upload dermatoscopic images, perform pixel-level segmentation, obtain quantitative morphological metrics, and engage in natural-language Q&A through an integrated iFlytek Spark large language model.

This project implements and benchmarks four segmentation architectures on the **ISIC2018 skin-lesion dataset**:

1. **U-Net** — the classic encoder-decoder baseline model.
2. **MultiResUNet** — introduces multi-scale residual blocks to handle scenarios with large variations in target size.
3. **DoubleU-Net** — stacks two U-Nets, using a VGG-19 encoder, SE attention mechanism, and ASPP atrous convolution.
4. **Improved DoubleU-Net** (*this project's contribution*) — introduces **MultiResBlock** and **ResPath** modules into DoubleU-Net to enlarge the receptive field, narrow the semantic gap between encoder and decoder, and alleviate gradient decay in deep networks.

> 🎯 **Key Result:** The improved DoubleU-Net achieves the best performance on **IoU 0.8306 / Dice 0.8951 / Recall 0.9267**, comprehensively surpassing the baseline models on the metrics most critical to clinical lesion localization.

---

## ✨ Highlights

- 🔬 **Comparative experiments across four segmentation models** with a unified training and evaluation pipeline.
- 🧠 **Original architectural improvement** — injecting MultiResBlock and ResPath into DoubleU-Net.
- 🖥️ **Full-stack Flask web application** — upload, segmentation, visualization, and analysis all within the browser.
- 🤖 **LLM-powered assistant** — integrated iFlytek Spark large language model for natural-language Q&A about models, metrics, and results.
- 📊 **Comprehensive post-segmentation analysis** — lesion area, boundary length, morphological complexity, and runtime performance metrics.
- 🗂️ **Localized session management** — automatic history saving, full-text keyword search with highlighting, renaming, and deletion.
- 🎨 **Modern UI** — light/dark theme switching, responsive layout, hover animations, and dynamic feedback.

---

## 🧠 Network Architecture

### Baseline Model: DoubleU-Net

DoubleU-Net stacks two U-Net sub-networks: the first performs coarse segmentation, and the second refines and enhances it. It employs a pre-trained **VGG-19** encoder, **SE (Squeeze-and-Excitation) attention**, and **ASPP (Atrous Spatial Pyramid Pooling)** to strengthen multi-scale feature modeling.

### This Project's Improvement: MultiResBlock + ResPath

Although the original DoubleU-Net is powerful, it suffers from the following issues:
- **Fixed receptive field** (only two 3×3 convolutions per layer), making it difficult to simultaneously capture fine details and global context.
- **Semantic gap**: a mismatch exists between shallow encoder features and deep decoder features, especially when concatenating the first VGG-19 layer with the deepest decoder layer.
- **Channel redundancy**: not every channel contributes equally.
- **Gradient decay**: information flow is impeded in deep stacked architectures.

To address these issues, this project introduces:

| Module | Role |
|--------|------|
| **MultiResBlock** | An Inception-style parallel multi-scale convolution (3×3, 5×5, 7×7) combined with residual connections, enlarging the receptive field and capturing multi-scale features. |
| **ResPath** | A residual skip connection that progressively refines encoder features before they are fused with decoder features, narrowing the semantic gap. |

<div align="center">

<img width="623" height="360" alt="image" src="https://github.com/user-attachments/assets/8a399a13-8c37-4799-8535-d0e4b2606a5b" />

</div>

---

## 📊 Experimental Results

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 256 × 256 |
| Batch size | 8 |
| Epochs | 50 |
| Optimizer | Adam |
| Loss function | BCEWithLogitsLoss |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Hardware | NVIDIA GeForce RTX 3060 |
| Framework | Python 3.8 + PyTorch |

### Quantitative Comparison on ISIC2018

| Model | IoU ↑ | Dice ↑ | Accuracy ↑ | Precision ↑ | Recall ↑ |
|-------|:-----:|:------:|:----------:|:-----------:|:--------:|
| U-Net | 0.7939 | 0.8648 | 0.9238 | 0.9127 | 0.8677 |
| MultiResUNet | 0.8015 | 0.8761 | 0.9259 | 0.8743 | 0.9157 |
| DoubleU-Net | 0.8269 | 0.8944 | **0.9412** | 0.8939 | 0.9248 |
| **Improved DoubleU-Net (Ours)** | **0.8306** | **0.8951** | 0.9354 | **0.8973** | **0.9267** |

> 💡 The improved model achieves the best results on **IoU, Dice, Precision, and Recall** — the metrics most critical for clinical lesion localization and boundary delineation. The slight drop in accuracy is a deliberate trade-off: the model becomes more sensitive and focused on foreground lesion regions, at the cost of a small number of background pixels being misclassified.

### Data Augmentation Strategy

| Augmentation | Value |
|--------------|-------|
| RandomHorizontalFlip | p = 0.5 |
| RandomVerticalFlip | p = 0.5 |
| RandomRotation | ±30° |
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.2 |
| Normalize | mean=[0.708, 0.582, 0.536], std=[0.157, 0.166, 0.181] |

<div align="center">
<img width="530" height="205" alt="image" src="https://github.com/user-attachments/assets/0438da96-f25a-4464-9af4-d1d936dad4d3" />

</div>
<div align="center">
<img width="545" height="204" alt="image" src="https://github.com/user-attachments/assets/958460a4-ebd1-431b-b942-c806aa969db8" />

</div>

> 📌 The normalization statistics are **recomputed from the ISIC2018 training set**, rather than reusing the default ImageNet values. Because dermatoscopic images have a relatively narrow color distribution, custom normalization significantly improves convergence speed and training stability.
<div align="center">
<img width="600" height="247" alt="image" src="https://github.com/user-attachments/assets/cb6d0638-d657-4c68-b857-aa127e198ede" />
<img width="596" height="288" alt="image" src="https://github.com/user-attachments/assets/3598dd42-3109-4a84-8684-bf5ed7f0362a" />
<img width="611" height="287" alt="image" src="https://github.com/user-attachments/assets/0004e6f3-0700-4883-9e7e-8f3715a89344" />

</div>
---

## 🚀 training data
<div align="center">
<img width="444" height="250" alt="image" src="https://github.com/user-attachments/assets/b8e13b49-407f-413c-b36f-1691abbf11ed" />
<img width="461" height="259" alt="image" src="https://github.com/user-attachments/assets/c914c4a8-c8a7-43e3-940f-c5b277f78e7c" />
<img width="432" height="250" alt="image" src="https://github.com/user-attachments/assets/6785c3dc-b7af-494b-bccb-b3f9de1eb82a" />
<img width="427" height="250" alt="image" src="https://github.com/user-attachments/assets/4c28f6a6-4049-453a-8488-ecca018af246" />

</div>

### Qualitative analysis
<div align="center">
<img width="526" height="265" alt="image" src="https://github.com/user-attachments/assets/d3dbe99f-0bcf-4038-a6e6-6470a53088ba" />
<img width="584" height="294" alt="image" src="https://github.com/user-attachments/assets/ba052a0a-318c-4e55-a328-5833e9b990d7" />
<img width="589" height="297" alt="image" src="https://github.com/user-attachments/assets/06cfafc1-5d4c-4c2a-a63a-3a9d2d3930d7" />
<img width="624" height="314" alt="image" src="https://github.com/user-attachments/assets/bf62be38-871e-46ed-9fb9-56aefd1f60cc" />

</div>


##  🖥️ System Architecture

The platform adopts a **front-end / back-end separation** architecture:

┌──────────────────────────────────────────────────────────────┐
│ Browser (Front-end) │
│ HTML + CSS + JavaScript │ Light/Dark Theme │ localStorage
│ • Image upload │ • Chat interface │ • History
│ • Model selection │ • Result rendering │ • Search
└──────────────────┬───────────────────────────────────────────┘
│ HTTP / JSON
▼
┌──────────────────────────────────────────────────────────────┐
│ Flask Back-end │
│ /predict → Image preprocessing → Model inference → Mask │
│ /chat → Call Spark LLM API → Natural-language reply │
│ /metrics → Morphological & performance analysis │
└──────────────────┬───────────────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────────────┐
│ PyTorch Model Zoo (4 models) │
│ U-Net │ MultiResUNet │ DoubleU-Net │ Improved DoubleU-Net │
└──────────────────────────────────────────────────────────────┘


### Core Functional Modules

| Module | Description |
|--------|-------------|
| **Image Segmentation** | Upload a dermatoscopic image, select a model, and obtain visualizations of the original image, binary mask, and boundary overlay. |
| **Quantitative Analysis** | Compute lesion coverage, area, boundary length, and morphological complexity based on the binary mask. |
| **Performance Analysis** | Monitor model loading time, inference time, total processing time, and memory usage via `psutil`. |
| **Spark LLM Assistant** | Ask natural-language questions about model architecture, metric definitions, dataset background, and segmentation result interpretation. |
| **Session Management** | Automatically save history by timestamp; supports full-text keyword search with highlighting, renaming, deletion, and context preview. |
| **UI/UX** | Light/dark theme switching, responsive layout, collapsible sidebar, hover animations, and dynamic button feedback. |

<div align="center">
<img width="591" height="293" alt="image" src="https://github.com/user-attachments/assets/086f3dde-90b4-4ed0-89a3-613cfb5b69d3" />
<img width="493" height="323" alt="image" src="https://github.com/user-attachments/assets/c5b4e5c5-680a-4686-a085-1cc03da87a5b" />
<img width="622" height="338" alt="image" src="https://github.com/user-attachments/assets/266aacb8-38ce-435a-a65d-e73d05df18a2" />
<img width="624" height="116" alt="image" src="https://github.com/user-attachments/assets/d273a1d1-d95e-4a74-a8bd-7417f03220aa" />
<img width="593" height="268" alt="image" src="https://github.com/user-attachments/assets/fb53d10d-8004-41d6-95ea-8cf75917b90b" />
<img width="624" height="146" alt="image" src="https://github.com/user-attachments/assets/685c1676-f7ee-40db-8c9b-c754cb5706da" />
<img width="624" height="278" alt="image" src="https://github.com/user-attachments/assets/71b51f60-7279-44ce-84a3-f4b30bb8f871" />
<img width="623" height="309" alt="image" src="https://github.com/user-attachments/assets/ab9539c2-b78e-44d3-ba45-f0445bde13dc" />
<img width="624" height="310" alt="image" src="https://github.com/user-attachments/assets/15fcf9e5-47fb-4227-a2f0-330880991961" />
</div>
---


## 📈 Evaluation Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **IoU** | TP / (TP + FP + FN) | Overlap between predicted and ground-truth regions; sensitive to small targets. |
| **Dice** | 2TP / (2TP + FP + FN) | An F1-like overlap measure; especially sensitive to small-area lesions. |
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness; biased on imbalanced data. |
| **Precision** | TP / (TP + FP) | Trustworthiness of positive predictions (low false positives). |
| **Recall** | TP / (TP + FN) | Coverage of the true foreground (low false negatives). |

---

## 🩺 Application Scenarios

- **Clinical Pre-screening** — Automatically delineates lesion boundaries and quantifies morphological metrics for dermatologists.
- **Research & Model Benchmarking** — Provides a reproducible platform for evaluating new segmentation architectures with standardized metrics.
- **Medical Education** — An interactive teaching tool where students upload cases, view analyses, and ask the LLM about concepts.
- **Remote Assisted Diagnosis** — Can be integrated into telemedicine platforms in the future, enabling patient self-screening and remote physician review.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch, torchvision |
| Back-end | Flask, Gunicorn (production) |
| Front-end | HTML5, CSS3, vanilla JavaScript |
| Large Language Model | iFlytek Spark Web API |
| System Monitoring | psutil |
| Visualization | Matplotlib, OpenCV |
| Dataset | ISIC2018 |

---

## 📚 References

This project builds upon the following foundational works:

1. Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
2. Ibtehaz & Rahman, *MultiResUNet: Rethinking the U-Net Architecture for Multimodal Biomedical Image Segmentation*, Neural Networks 2020.
3. Jha et al., *Double U-Net: A Deep Convolutional Neural Network for Medical Image Segmentation*, IEEE CBMS 2020.
4. Zhou et al., *UNet++: Redesigning Skip Connections to Exploit Multiscale Features*, IEEE TMI 2020.
5. Chen et al., *DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs*, IEEE TPAMI 2018.

See the accompanying thesis document for the full reference list.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Thanks to the **ISIC Archive** for providing the public dermatoscopic image dataset.
- Thanks to the open-source communities behind **PyTorch**, **Flask**, and **OpenCV**.
- Thanks to the authors of U-Net, MultiResUNet, and DoubleU-Net for their foundational architectures.
- Thanks to **iFlytek** for the Spark large language model API.

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**

*MedSeg-UNet — Bridging deep learning research and clinical-grade medical image analysis.*

</div>
