import os
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# =====================
# —— MultiResUnet 架构 —— 这里是你提供的模型代码
# =====================
class Conv2d_batchnorm(torch.nn.Module):
    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride=(1, 1), activation='relu'):
        super().__init__()
        self.activation = activation
        self.conv1 = torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size,
                                     stride=stride, padding='same')
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        if self.activation == 'relu':
            return torch.nn.functional.relu(x)
        else:
            return x


class Multiresblock(torch.nn.Module):
    def __init__(self, num_in_channels, num_filters, alpha=1.67):
        super().__init__()
        self.alpha = alpha
        self.W = num_filters * alpha

        filt_cnt_3x3 = int(self.W * 0.167)
        filt_cnt_5x5 = int(self.W * 0.333)
        filt_cnt_7x7 = int(self.W * 0.5)
        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7

        self.shortcut = Conv2d_batchnorm(num_in_channels, num_out_filters, kernel_size=(1, 1), activation='None')
        self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size=(3, 3), activation='relu')
        self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size=(3, 3), activation='relu')
        self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size=(3, 3), activation='relu')
        self.batch_norm1 = torch.nn.BatchNorm2d(num_out_filters)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        shrtct = self.shortcut(x)
        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)
        x = torch.cat([a, b, c], axis=1)
        x = self.batch_norm1(x)
        x = x + shrtct
        x = self.batch_norm2(x)
        x = torch.nn.functional.relu(x)
        return x


class Respath(torch.nn.Module):
    def __init__(self, num_in_filters, num_out_filters, respath_length):
        super().__init__()
        self.respath_length = respath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        for i in range(self.respath_length):
            if i == 0:
                self.shortcuts.append(
                    Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size=(1, 1), activation='None'))
                self.convs.append(
                    Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size=(3, 3), activation='relu'))
            else:
                self.shortcuts.append(
                    Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size=(1, 1), activation='None'))
                self.convs.append(
                    Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size=(3, 3), activation='relu'))
            self.bns.append(torch.nn.BatchNorm2d(num_out_filters))

    def forward(self, x):
        for i in range(self.respath_length):
            shortcut = self.shortcuts[i](x)
            x = self.convs[i](x)
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)
            x = x + shortcut
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)
        return x


class MultiResUnet(torch.nn.Module):
    def __init__(self, input_channels, num_classes, alpha=1.67):
        super().__init__()
        self.alpha = alpha

        # Encoder Path
        self.multiresblock1 = Multiresblock(input_channels, 32)
        self.in_filters1 = int(32 * self.alpha * 0.167) + int(32 * self.alpha * 0.333) + int(32 * self.alpha * 0.5)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.respath1 = Respath(self.in_filters1, 32, respath_length=4)

        self.multiresblock2 = Multiresblock(self.in_filters1, 32 * 2)
        self.in_filters2 = int(32 * 2 * self.alpha * 0.167) + int(32 * 2 * self.alpha * 0.333) + int(
            32 * 2 * self.alpha * 0.5)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.respath2 = Respath(self.in_filters2, 32 * 2, respath_length=3)

        self.multiresblock3 = Multiresblock(self.in_filters2, 32 * 4)
        self.in_filters3 = int(32 * 4 * self.alpha * 0.167) + int(32 * 4 * self.alpha * 0.333) + int(
            32 * 4 * self.alpha * 0.5)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.respath3 = Respath(self.in_filters3, 32 * 4, respath_length=2)

        self.multiresblock4 = Multiresblock(self.in_filters3, 32 * 8)
        self.in_filters4 = int(32 * 8 * self.alpha * 0.167) + int(32 * 8 * self.alpha * 0.333) + int(
            32 * 8 * self.alpha * 0.5)
        self.pool4 = torch.nn.MaxPool2d(2)
        self.respath4 = Respath(self.in_filters4, 32 * 8, respath_length=1)

        self.multiresblock5 = Multiresblock(self.in_filters4, 32 * 16)
        self.in_filters5 = int(32 * 16 * self.alpha * 0.167) + int(32 * 16 * self.alpha * 0.333) + int(
            32 * 16 * self.alpha * 0.5)

        # Decoder path
        self.upsample6 = torch.nn.ConvTranspose2d(self.in_filters5, 32 * 8, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters1 = 32 * 8 * 2
        self.multiresblock6 = Multiresblock(self.concat_filters1, 32 * 8)
        self.in_filters6 = int(32 * 8 * self.alpha * 0.167) + int(32 * 8 * self.alpha * 0.333) + int(
            32 * 8 * self.alpha * 0.5)

        self.upsample7 = torch.nn.ConvTranspose2d(self.in_filters6, 32 * 4, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters2 = 32 * 4 * 2
        self.multiresblock7 = Multiresblock(self.concat_filters2, 32 * 4)
        self.in_filters7 = int(32 * 4 * self.alpha * 0.167) + int(32 * 4 * self.alpha * 0.333) + int(
            32 * 4 * self.alpha * 0.5)

        self.upsample8 = torch.nn.ConvTranspose2d(self.in_filters7, 32 * 2, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters3 = 32 * 2 * 2
        self.multiresblock8 = Multiresblock(self.concat_filters3, 32 * 2)
        self.in_filters8 = int(32 * 2 * self.alpha * 0.167) + int(32 * 2 * self.alpha * 0.333) + int(
            32 * 2 * self.alpha * 0.5)

        self.upsample9 = torch.nn.ConvTranspose2d(self.in_filters8, 32, kernel_size=(2, 2), stride=(2, 2))
        self.concat_filters4 = 32 * 2
        self.multiresblock9 = Multiresblock(self.concat_filters4, 32)
        self.in_filters9 = int(32 * self.alpha * 0.167) + int(32 * self.alpha * 0.333) + int(32 * self.alpha * 0.5)

        self.conv_final = Conv2d_batchnorm(self.in_filters9, num_classes , kernel_size=(1, 1), activation='None')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_multires1 = self.multiresblock1(x)
        x_pool1 = self.pool1(x_multires1)
        x_multires1 = self.respath1(x_multires1)

        x_multires2 = self.multiresblock2(x_pool1)
        x_pool2 = self.pool2(x_multires2)
        x_multires2 = self.respath2(x_multires2)

        x_multires3 = self.multiresblock3(x_pool2)
        x_pool3 = self.pool3(x_multires3)
        x_multires3 = self.respath3(x_multires3)

        x_multires4 = self.multiresblock4(x_pool3)
        x_pool4 = self.pool4(x_multires4)
        x_multires4 = self.respath4(x_multires4)

        x_multires5 = self.multiresblock5(x_pool4)

        up6 = torch.cat([self.upsample6(x_multires5), x_multires4], axis=1)
        x_multires6 = self.multiresblock6(up6)

        up7 = torch.cat([self.upsample7(x_multires6), x_multires3], axis=1)
        x_multires7 = self.multiresblock7(up7)

        up8 = torch.cat([self.upsample8(x_multires7), x_multires2], axis=1)
        x_multires8 = self.multiresblock8(up8)

        up9 = torch.cat([self.upsample9(x_multires8), x_multires1], axis=1)
        x_multires9 = self.multiresblock9(up9)

        out = self.conv_final(x_multires9)

        return out


# =====================
# —— 数据集类（不变） ——
# =====================

class ISIC2018Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(256, 256), mode='train'):
        self.image_dir, self.mask_dir = image_dir, mask_dir
        self.img_size, self.mode = img_size, mode
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.708, 0.582, 0.536], [0.157, 0.166, 0.181])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor()
        ])
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, self.images[idx])), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_segmentation.png')),
                         cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        msk = cv2.resize(msk, self.img_size)
        if self.mode == 'train':
            seed = torch.randint(0, 2 ** 32, (1,)).item()
            torch.manual_seed(seed)
            img = self.image_transform(img)
            torch.manual_seed(seed)
            msk = self.mask_transform(msk)
        else:
            img = transforms.Normalize([0.708, 0.582, 0.536], [0.157, 0.166, 0.181])(
                transforms.ToTensor()(img))
            msk = transforms.ToTensor()(msk)
        return img, (msk > 0.5).float()


# =====================
# —— 训练与评估 ——
# =====================

def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss, ious, dices = 0, [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            total_loss += loss.item() * imgs.size(0)
            preds = torch.sigmoid(preds) > 0.5
            inter = (preds & masks.bool()).sum((1, 2, 3)).float()
            union = (preds | masks.bool()).sum((1, 2, 3)).float()
            ious.append(((inter + 1e-6) / (union + 1e-6)).mean().item())
            dices.append(((2 * inter + 1e-6) / (preds.sum((1, 2, 3)) + masks.sum((1, 2, 3)) + 1e-6)).mean().item())
    return total_loss / len(loader.dataset), np.mean(ious), np.mean(dices)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10):
    best_iou, epochs_no_improve = 0, 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        val_loss, val_iou, val_dice = evaluate_model(model, val_loader, criterion)
        print(
            f"[{epoch + 1}] TrainLoss:{running_loss / len(train_loader.dataset):.4f} " + f"ValLoss:{val_loss:.4f} IoU:{val_iou:.4f} Dice:{val_dice:.4f}")
        if val_iou > best_iou:
            best_iou = val_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_multiresunet.pth')
            print(' best iou:',best_iou)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break


# =====================
# —— 主流程 ——
# =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuDNN 是否启用:", torch.backends.cudnn.enabled)  # 返回 True 表示启用
    data_paths = {
        'train': {'image_dir': 'data/ISIC2018_Task1-2_Training_Input',
                  'mask_dir': 'data/ISIC2018_Task1_Training_GroundTruth'},
        'val': {'image_dir': 'data/ISIC2018_Task1-2_Validation_Input',
                'mask_dir': 'data/ISIC2018_Task1_Validation_GroundTruth'},
        'test': {'image_dir': 'data/ISIC2018_Task1-2_Test_Input', 'mask_dir': 'data/ISIC2018_Task1_Test_GroundTruth'}
    }
    train_ds = ISIC2018Dataset(**data_paths['train'], mode='train')
    val_ds = ISIC2018Dataset(**data_paths['val'], mode='val')
    test_ds = ISIC2018Dataset(**data_paths['test'], mode='test')
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = MultiResUnet(input_channels=3, num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=15)

    # 最终测试
    model.load_state_dict(torch.load('best_multiresunet.pth'))
    test_loss, test_iou, test_dice = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss:{test_loss:.4f} IoU:{test_iou:.4f} Dice:{test_dice:.4f}")
