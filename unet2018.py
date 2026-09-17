import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用于进度条显示
import datetime

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 日志目录配置
log_dir = "logpro"
os.makedirs(log_dir, exist_ok=True)  # 自动创建日志目录
log_filename = f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_path = os.path.join(log_dir, log_filename)

def log_message(message, print_to_console=True):
    with open(log_path, "a") as f:
        f.write(message + "\n")
    if print_to_console:
        print(message)
# 自定义数据集类
# 自定义数据集类 (关键修改点)
class ISIC2018Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(256, 256), mode='train'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.mode = mode

        # 定义图像和掩膜的不同增强流程
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            # 空间变换（需与掩膜同步）
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            # 颜色变换（仅图像）
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.708, 0.582, 0.536], [0.157, 0.166, 0.181])
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            # 空间变换（需与图像同步）
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ToTensor()
        ])

        # 文件列表匹配检查
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        assert len(self.image_files) == len(self.mask_files), "图像和掩膜数量不匹配"
        for img, mask in zip(self.image_files, self.mask_files):
            assert img.replace('.jpg', '') == mask.replace('_segmentation.png', ''), f"文件 {img} 和 {mask} 不匹配"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name.replace('.jpg', '_segmentation.png')

        # 读取图像和掩膜
        image = cv2.imread(os.path.join(self.image_dir, img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)

        # 调整尺寸
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size)

        # 同步随机种子以确保相同的空间变换
        if self.mode == 'train':
            seed = torch.randint(0, 2 ** 32, (1,)).item()

            # 处理图像
            torch.manual_seed(seed)
            image = self.image_transform(image)

            # 处理掩膜（使用相同的随机参数）
            torch.manual_seed(seed)
            mask = self.mask_transform(mask)
        else:
            # 验证/测试集仅做归一化
            image = transforms.ToTensor()(image)
            image = transforms.Normalize([0.708, 0.582, 0.536], [0.157, 0.166, 0.181])(image)
            mask = transforms.ToTensor()(mask)

        # 二值化掩膜并调整维度
        mask = (mask > 0.5).float()

        return image, mask

# U-Net模型定义（保持原结构）
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            return block

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        self.bottleneck = conv_block(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.decoder4 = conv_block(1024, 512)
        self.decoder3 = conv_block(512, 256)
        self.decoder2 = conv_block(256, 128)
        self.decoder1 = conv_block(128, 64)

        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        #return torch.sigmoid(self.conv_final(dec1))
        return self.conv_final(dec1)  # 移除torch.sigmoid


# 训练函数（增加验证步骤）
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10):
    best_iou = 0.0
    epochs_no_improve = 0

    # 记录初始信息
    log_message("\n===== Training Configuration =====")
    log_message(f"Epochs: {num_epochs}")
    log_message(f"Batch size: {train_loader.batch_size}")
    log_message(f"Optimizer: {optimizer}")
    log_message(f"Criterion: {criterion.__class__.__name__}")
    log_message("=" * 35 + "\n")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        # 使用上下文管理器确保进度条正确关闭
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}') as progress_bar:
            for images, masks in progress_bar:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                progress_bar.set_postfix({'loss': loss.item()})

        # 验证阶段
        val_loss, val_iou, val_dice = evaluate_model(model, val_loader, criterion)

        # 记录训练结果
        log_message(
            f'Epoch {epoch + 1} | '
            f'Train Loss: {train_loss / len(train_loader.dataset):.4f} | '
            f'Val Loss: {val_loss:.4f} | '
            f'Val IoU: {val_iou:.4f} | '
            f'Val Dice: {val_dice:.4f}',
            print_to_console=True
        )

        # 早停判断
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_model_zengqiang.pth')
            epochs_no_improve = 0
            log_message(f"New best model saved at epoch {epoch + 1} (IoU: {best_iou:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                stop_msg = f'Early stopping triggered at epoch {epoch + 1}'
                log_message(stop_msg)
                break

# 增强版评估函数
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    iou_scores = []
    dice_scores = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)

            # 计算指标
            preds = (torch.sigmoid(outputs) > 0.5).float()
            intersection = (preds * masks).sum((1, 2, 3))
            union = (preds + masks).sum((1, 2, 3)) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            dice = (2. * intersection + 1e-6) / (preds.sum((1, 2, 3)) + masks.sum((1, 2, 3)) + 1e-6)

            iou_scores.extend(iou.cpu().numpy())
            dice_scores.extend(dice.cpu().numpy())

    return (
        total_loss / len(dataloader.dataset),
        np.mean(iou_scores),
        np.mean(dice_scores)
    )

if __name__ == "__main__":
    # 路径配置
    data_paths = {
        'train': {
            'image_dir': 'data/ISIC2018_Task1-2_Training_Input',
            'mask_dir': 'data/ISIC2018_Task1_Training_GroundTruth'
        },
        'val': {
            'image_dir': 'data/ISIC2018_Task1-2_Validation_Input',
            'mask_dir': 'data/ISIC2018_Task1_Validation_GroundTruth'
        },
        'test': {
            'image_dir': 'data/ISIC2018_Task1-2_Test_Input',
            'mask_dir': 'data/ISIC2018_Task1_Test_GroundTruth'
        }
    }

    # 创建数据集
    train_dataset = ISIC2018Dataset(**data_paths['train'], img_size=(256, 256),  mode='train')
    val_dataset = ISIC2018Dataset(**data_paths['val'], img_size=(256, 256), mode='val')
    test_dataset = ISIC2018Dataset(**data_paths['test'], img_size=(256, 256), mode='test')

    # 创建数据加载器
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()  # 更稳定的损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # 记录开始信息
    log_message(f"\n{'=' * 30} Training Start {'=' * 30}")
    log_message(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Device: {device}")
    log_message(f"Training samples: {len(train_dataset)}")
    log_message(f"Validation samples: {len(val_dataset)}")
    log_message(f"Test samples: {len(test_dataset)}\n")

    # 训练模型
    try:
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
    except Exception as e:
        log_message(f"\nTraining interrupted by error: {str(e)}", print_to_console=True)
        raise

    # 最终测试评估
    model.load_state_dict(torch.load('best_model_zengqiang.pth'))
    test_loss, test_iou, test_dice = evaluate_model(model, test_loader, criterion)
    log_message("\n===== Final Test Results =====")
    log_message(f"Test Loss: {test_loss:.4f}")
    log_message(f"Test IoU: {test_iou:.4f}")
    log_message(f"Test Dice: {test_dice:.4f}")
    log_message("=" * 30)

    # 可视化保存结果
    output_dir = 'sr2018'
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i in range(min(10, len(test_dataset))):
            image, mask = test_dataset[i]
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            pred_mask = (output.squeeze() > 0.5).cpu().numpy().astype(np.uint8)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1).imshow(image.permute(1, 2, 0))
            plt.subplot(1, 3, 2).imshow(mask.squeeze(), cmap='gray')
            plt.subplot(1, 3, 3).imshow(pred_mask, cmap='gray')
            plt.savefig(f'{output_dir}/result_{i}.png', bbox_inches='tight')
            plt.close()

    # 记录结束信息
    log_message(f"\n{'=' * 30} Training Complete {'=' * 30}")
    log_message(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Log file saved at: {os.path.abspath(log_path)}")