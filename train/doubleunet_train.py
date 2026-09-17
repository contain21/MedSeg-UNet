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
# —— DoubleU-Net 架构 ——
# 引入您提供的模型定义
# =====================
import torch.nn.functional as F
from torchvision.models import vgg19

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding,
                      dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        return self.relu(x) if self.act else x

class squeeze_excitation_block(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//ratio, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2,2)),
            Conv2D(in_c, out_c, kernel_size=1, padding=0)
        )
        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=6, dilation=6)
        self.c3 = Conv2D(in_c, out_c, kernel_size=3, padding=12, dilation=12)
        self.c4 = Conv2D(in_c, out_c, kernel_size=3, padding=18, dilation=18)
        self.c5 = Conv2D(out_c*5, out_c, kernel_size=1, padding=0)
    def forward(self, x):
        x0 = self.avgpool(x)
        x0 = F.interpolate(x0, size=x.shape[2:], mode="bilinear", align_corners=True)
        x1, x2, x3, x4 = self.c1(x), self.c2(x), self.c3(x), self.c4(x)
        xc = torch.cat([x0, x1, x2, x3, x4], dim=1)
        return self.c5(xc)

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.c1 = Conv2D(in_c, out_c)
        self.c2 = Conv2D(out_c, out_c)
        self.se = squeeze_excitation_block(out_c)
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return self.se(x)

class encoder1(nn.Module):
    def __init__(self):
        super().__init__()
        net = vgg19(pretrained=True)
        self.x1 = net.features[:4]
        self.x2 = net.features[4:9]
        self.x3 = net.features[9:18]
        self.x4 = net.features[18:27]
        self.x5 = net.features[27:36]
    def forward(self, x):
        x1 = self.x1(x)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        x5 = self.x5(x4)
        return x5, [x4, x3, x2, x1]

class decoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(64 + 512, 256)
        self.c2 = conv_block(512, 128)
        self.c3 = conv_block(256, 64)
        self.c4 = conv_block(128, 32)
    def forward(self, x, skips):
        for idx, c in enumerate([self.c1, self.c2, self.c3, self.c4]):
            x = self.up(x)
            x = torch.cat([x, skips[idx]], dim=1)
            x = c(x)
        return x

class encoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.c1 = conv_block(3,32)
        self.c2 = conv_block(32,64)
        self.c3 = conv_block(64,128)
        self.c4 = conv_block(128,256)
    def forward(self, x):
        s=[]
        for c in [self.c1,self.c2,self.c3,self.c4]:
            x = c(x)
            s.append(x)
            x = self.pool(x)
        return x, s[::-1]  # reverse for skip order

class decoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(832, 256)
        self.c2 = conv_block(640, 128)
        self.c3 = conv_block(320, 64)
        self.c4 = conv_block(160, 32)
    def forward(self, x, skip1, skip2):
        skips2 = skip2
        for idx, c in enumerate([self.c1,self.c2,self.c3,self.c4]):
            x = self.up(x)
            x = torch.cat([x, skip1[idx], skips2[idx]], dim=1)
            x = c(x)
        return x

class build_doubleunet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1 = encoder1()
        self.a1 = ASPP(512,64)
        self.d1 = decoder1()
        self.y1 = nn.Conv2d(32,1,1)
        self.e2 = encoder2()
        self.a2 = ASPP(256,64)
        self.d2 = decoder2()
        self.y2 = nn.Conv2d(32,1,1)
    def forward(self, x):
        x0 = x
        x, skips1 = self.e1(x)
        x = self.a1(x)
        x = self.d1(x, skips1)
        y1 = self.y1(x)
        x2_in = x0 * torch.sigmoid(y1)
        x, skips2 = self.e2(x2_in)
        x = self.a2(x)
        x = self.d2(x, skips1, skips2)
        y2 = self.y2(x)
        return y1, y2

# =====================
# —— 数据集类（不变） ——
# =====================
class ISIC2018Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(256,256), mode='train'):
        self.image_dir, self.mask_dir = image_dir, mask_dir
        self.img_size, self.mode = img_size, mode
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.2,0.2,0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.708,0.582,0.536],[0.157,0.166,0.181])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor()
        ])
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.masks  = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir,self.images[idx])), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(os.path.join(self.mask_dir,self.images[idx].replace('.jpg','_segmentation.png')), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        msk = cv2.resize(msk, self.img_size)
        if self.mode=='train':
            seed = torch.randint(0,2**32,(1,)).item()
            torch.manual_seed(seed); img = self.image_transform(img)
            torch.manual_seed(seed); msk = self.mask_transform(msk)
        else:
            img = transforms.Normalize([0.708,0.582,0.536],[0.157,0.166,0.181])(
                  transforms.ToTensor()(img))
            msk = transforms.ToTensor()(msk)
        return img, (msk>0.5).float()

# =====================
# —— 训练与评估 ——
# =====================
def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss, ious, dices = 0, [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            y1, y2 = model(imgs)
            loss = criterion(y1, masks) + criterion(y2, masks)
            total_loss += loss.item()*imgs.size(0)
            preds = torch.sigmoid(y2)>0.5
            inter = (preds & masks.bool()).sum((1,2,3)).float()
            union = (preds | masks.bool()).sum((1,2,3)).float()
            ious.append(((inter+1e-6)/(union+1e-6)).mean().item())
            dices.append(((2*inter+1e-6)/(preds.sum((1,2,3))+masks.sum((1,2,3))+1e-6)).mean().item())
    return total_loss/len(loader.dataset), np.mean(ious), np.mean(dices)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10):
    best_iou, epochs_no_improve = 0, 0
    for epoch in range(num_epochs):
        model.train(); running_loss=0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            y1, y2 = model(imgs)
            loss = criterion(y1,masks) + criterion(y2,masks)
            loss.backward(); optimizer.step()
            running_loss += loss.item()*imgs.size(0)
        val_loss, val_iou, val_dice = evaluate_model(model, val_loader, criterion)
        print(f"[{epoch+1}] TrainLoss:{running_loss/len(train_loader.dataset):.4f} "+
              f"ValLoss:{val_loss:.4f} IoU:{val_iou:.4f} Dice:{val_dice:.4f}")
        if val_iou>best_iou:
            best_iou=val_iou; epochs_no_improve=0
            torch.save(model.state_dict(),'best_doubleunet.pth')
        else:
            epochs_no_improve+=1
            if epochs_no_improve>=patience:
                print("Early stopping"); break

# =====================
# —— 主流程 ——
# =====================
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_paths = {
      'train':{'image_dir':'data/ISIC2018_Task1-2_Training_Input','mask_dir':'data/ISIC2018_Task1_Training_GroundTruth'},
      'val':  {'image_dir':'data/ISIC2018_Task1-2_Validation_Input','mask_dir':'data/ISIC2018_Task1_Validation_GroundTruth'},
      'test': {'image_dir':'data/ISIC2018_Task1-2_Test_Input','mask_dir':'data/ISIC2018_Task1_Test_GroundTruth'}
    }
    train_ds = ISIC2018Dataset(**data_paths['train'], mode='train')
    val_ds   = ISIC2018Dataset(**data_paths['val'], mode='val')
    test_ds  = ISIC2018Dataset(**data_paths['test'], mode='test')
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False)

    model = build_doubleunet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10)

    # 最终测试
    model.load_state_dict(torch.load('best_doubleunet.pth'))
    test_loss, test_iou, test_dice = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss:{test_loss:.4f} IoU:{test_iou:.4f} Dice:{test_dice:.4f}")
