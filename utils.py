import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
import os
import time
import unet2018 as u1
import multiresunet as ums
import doubleunet as ud
import doublemultiatt as ug

# 图像预处理（与训练时完全一致）
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.708, 0.582, 0.536], std=[0.157, 0.166, 0.181])
    ])

    # 读取图像并转为RGB（与ISIC2018Dataset一致）
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))

    # 应用预处理
    return transform(image).unsqueeze(0)  # 添加batch维度


def load_model(model_path,name):
    """加载训练好的UNet模型"""
    if name == 'UNet':
        model = u1.UNet().to('cpu')
    elif name == 'MultiResUNet':
        model = ums.MultiResUnet(input_channels=3, num_classes=1).to('cpu')
    elif name == 'DoubleU-Net':
        model = ud.build_doubleunet().to('cpu')
    elif name == 'Improved DoubleU-Net':
        model = ug.build_doubleunet().to('cpu')
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_mask(model, input_path, output_path):
    """生成分割掩膜并保存"""
    # 1. 预处理
    input_tensor = preprocess_image(input_path)
    # 2. 预测（与训练时输出处理一致）
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, tuple):  # DoubleUNet 两个输出
            output = output[1]
        pred_mask = (torch.sigmoid(output) > 0.5).float().squeeze().cpu().numpy() * 255
    # 3. 保存为PNG（与你的可视化代码一致）
    mask = pred_mask.astype(np.uint8)
    Image.fromarray(mask).save(output_path)
    return mask


def generate_boundary_overlay(original_path, mask_path, output_path, line_width=2):
    """
    在原始图像上绘制分割边界红线
    :param original_path: 原始图像路径
    :param mask_path: 分割掩膜路径
    :param output_path: 输出路径
    :param line_width: 红线宽度
    """
    # 读取原始图像（保持原始尺寸）
    original = cv2.imread(original_path)
    original = cv2.resize(original, (256, 256))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # 读取分割掩膜并调整大小
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

    # 二值化处理掩膜
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原始图像上绘制红色轮廓
    overlay = original.copy()
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), line_width)

    # 保存结果
    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, overlay)
    return overlay


def analyze_segmentation(mask_path):
    """
    分析分割结果并返回各种指标
    :param mask_path: 分割掩膜路径
    :return: 包含各种指标的字典（确保所有值为Python原生类型）
    """
    # 读取分割掩膜
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 计算各种指标
    mask_area = int(np.sum(binary_mask == 255))  # 转换为Python int
    total_pixels = int(binary_mask.size)  # 转换为Python int
    coverage_ratio = float(mask_area / total_pixels * 100)  # 转换为Python float

    # 计算边界信息
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary_length = float(sum(cv2.arcLength(cnt, True) for cnt in contours))  # 转换为Python float

    return {
        "coverage_ratio": round(coverage_ratio, 2),  # 已经是Python float
        "lesion_area": mask_area,  # 已经是Python int
        "boundary_complexity": float(round(boundary_length / (2 * np.pi * np.sqrt(mask_area / np.pi)), 2)),
        # 转换为Python float
        "num_regions": int(len(contours)),  # 转换为Python int
        "boundary_length": float(round(boundary_length, 2))  # 转换为Python float
    }

def process_image_full_pipeline(model, input_path, output_dir, timestamp):
    """
    完整处理流程：预测+生成所有结果图
    :return: 生成的各个文件路径
    """
    # 1. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 2. 生成文件名
    original_filename = f"original_{timestamp}.png"
    seg_filename = f"seg_{timestamp}.png"
    overlay_filename = f"overlay_{timestamp}.png"

    original_path = os.path.join(output_dir, original_filename)
    seg_path = os.path.join(output_dir, seg_filename)
    overlay_path = os.path.join(output_dir, overlay_filename)

    # 3. 保存原始图像（调整大小后的）
    original_img = cv2.imread(input_path)
    original_img = cv2.resize(original_img, (256, 256))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(original_img).save(original_path)

    # 4. 预测分割结果
    mask = predict_mask(model, input_path, seg_path)

    # 5. 生成叠加图
    generate_boundary_overlay(original_path, seg_path, overlay_path)

    return {
        'original_path': original_path,
        'seg_path': seg_path,
        'overlay_path': overlay_path
    }
