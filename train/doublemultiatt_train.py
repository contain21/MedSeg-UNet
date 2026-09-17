import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19
import torch.optim as optim
from tqdm import tqdm

# =====================
# —— MultiRes & ResPath 模块 ——
# =====================
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class Multiresblock(nn.Module):
    def __init__(self, num_in_channels, num_out_channels):
        super().__init__()
        # 分配各卷积分支的通道数，总和为num_out_channels
        ratio = [0.167, 0.333, 0.5]
        f3 = int(num_out_channels * ratio[0])
        f5 = int(num_out_channels * ratio[1])
        f7 = num_out_channels - f3 - f5

        self.shortcut = nn.Conv2d(num_in_channels, num_out_channels, kernel_size=1, bias=False)
        self.conv3x3 = nn.Conv2d(num_in_channels, f3, kernel_size=3, padding=1, bias=False)
        self.conv5x5 = nn.Conv2d(f3, f5, kernel_size=3, padding=1, bias=False)
        self.conv7x7 = nn.Conv2d(f5, f7, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        a = self.relu(self.conv3x3(x))
        b = self.relu(self.conv5x5(a))
        c = self.relu(self.conv7x7(b))
        x = torch.cat([a, b, c], dim=1)
        x = self.bn(x) + shortcut
        return self.relu(x)

class Respath(nn.Module):
    def __init__(self, in_channels, out_channels, length):
        super().__init__()
        self.length = length
        self.layers = nn.ModuleList()
        for i in range(length):
            if i == 0:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            else:
                conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.layers.append(nn.Sequential(
                conv,
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)  # 残差连接
            x = F.relu(x)
        return x

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.multires = Multiresblock(in_c, out_c)  # 确保输出通道为out_c
        self.se = squeeze_excitation_block(out_c)
    def forward(self, x):
        x = self.multires(x)
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
        self.respaths = nn.ModuleList([
            Respath(512, 512, 2),  # 处理encoder1的跳跃特征
            Respath(256, 256, 2),
            Respath(128, 128, 2),
            Respath(64, 64, 2)
        ])
        self.c1 = conv_block(64+512, 256)
        self.c2 = conv_block(512, 128)
        self.c3 = conv_block(256, 64)
        self.c4 = conv_block(128, 32)

    def forward(self, x, skips):
        for idx, c in enumerate([self.c1, self.c2, self.c3, self.c4]):
            x = self.up(x)
            skip = self.respaths[idx](skips[idx])
            x = torch.cat([x, skip], dim=1)
            x = c(x)
        return x

class encoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.c1 = conv_block(3, 32)
        self.c2 = conv_block(32, 64)
        self.c3 = conv_block(64, 128)
        self.c4 = conv_block(128, 256)

    def forward(self, x):
        s = []
        for c in [self.c1, self.c2, self.c3, self.c4]:
            x = c(x)
            s.append(x)
            x = self.pool(x)
        return x, s[::-1]
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
        # 编码器部分
        self.e1 = encoder1()
        self.e2 = encoder2()  # 使用改进后的encoder2

        # 注意力模块
        self.a1 = ASPP(512, 64)
        self.a2 = ASPP(256, 64)

        # 解码器部分
        self.d1 = decoder1()  # 包含Respath处理
        self.d2 = decoder2()  # 需类似修改

        # 输出层
        self.y1 = nn.Conv2d(32, 1, 1)
        self.y2 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # 前向传播逻辑保持不变
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
# —— Dataset & Training (unchanged) ——
# =====================
class ISIC2018Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(256,256), mode='train'):
        self.image_dir, self.mask_dir = image_dir, mask_dir
        self.img_size, self.mode = img_size, mode
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30), transforms.ColorJitter(0.2,0.2,0.2), transforms.ToTensor(),
            transforms.Normalize([0.708,0.582,0.536],[0.157,0.166,0.181])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30), transforms.ToTensor()
        ])
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir,self.images[idx])), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(os.path.join(self.mask_dir,self.images[idx].replace('.jpg','_segmentation.png')), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size); msk = cv2.resize(msk, self.img_size)
        if self.mode=='train':
            seed=torch.randint(0,2**32,(1,)).item(); torch.manual_seed(seed)
            img=self.image_transform(img); torch.manual_seed(seed); msk=self.mask_transform(msk)
        else:
            img=transforms.Normalize([0.708,0.582,0.536],[0.157,0.166,0.181])(transforms.ToTensor()(img))
            msk=transforms.ToTensor()(msk)
        return img,(msk>0.5).float()

def evaluate_model(model, loader, criterion):
    model.eval(); total_loss, ious, dices = 0, [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            y1,y2 = model(imgs)
            loss = criterion(y1,masks) + criterion(y2,masks)
            total_loss += loss.item()*imgs.size(0)
            preds = torch.sigmoid(y2)>0.5
            inter=(preds & masks.bool()).sum((1,2,3)).float()
            union=(preds | masks.bool()).sum((1,2,3)).float()
            ious.append(((inter+1e-6)/(union+1e-6)).mean().item())
            dices.append(((2*inter+1e-6)/(preds.sum((1,2,3))+masks.sum((1,2,3))+1e-6)).mean().item())
    return total_loss/len(loader.dataset), np.mean(ious), np.mean(dices)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10):
    best_iou, epochs_no_improve = 0, 0
    for epoch in range(num_epochs):
        model.train(); run_loss=0
        for imgs,masks in tqdm(train_loader,desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs,masks=imgs.to(device),masks.to(device)
            optimizer.zero_grad(); y1,y2=model(imgs)
            loss=criterion(y1,masks)+criterion(y2,masks)
            loss.backward(); optimizer.step()
            run_loss+=loss.item()*imgs.size(0)
        val_loss,val_iou,val_dice=evaluate_model(model,val_loader,criterion)
        print(f"[{epoch+1}] TrainLoss:{run_loss/len(train_loader.dataset):.4f} "+
              f"ValLoss:{val_loss:.4f} IoU:{val_iou:.4f} Dice:{val_dice:.4f}")
        if val_iou>best_iou:
            best_iou,val_iou=val_iou,val_iou; epochs_no_improve=0
            torch.save(model.state_dict(),'best_fuse.pth')
            print("best_iou:",best_iou)
        else:
            epochs_no_improve+=1
            if epochs_no_improve>=patience: print("Early stopping"); break

if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = {
      'train':{'image_dir':'data/ISIC2018_Task1-2_Training_Input','mask_dir':'data/ISIC2018_Task1_Training_GroundTruth'},
      'val':  {'image_dir':'data/ISIC2018_Task1-2_Validation_Input','mask_dir':'data/ISIC2018_Task1_Validation_GroundTruth'},
      'test': {'image_dir':'data/ISIC2018_Task1-2_Test_Input','mask_dir':'data/ISIC2018_Task1_Test_GroundTruth'}
    }
    train_loader=DataLoader(ISIC2018Dataset(**paths['train'],mode='train'),batch_size=8,shuffle=True,num_workers=4)
    val_loader=DataLoader(ISIC2018Dataset(**paths['val'],mode='val'),batch_size=8)
    test_loader=DataLoader(ISIC2018Dataset(**paths['test'],mode='test'),batch_size=8)

    model=build_doubleunet().to(device)
    criterion=nn.BCEWithLogitsLoss()
    optimizer=optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)

    train_model(model,train_loader,val_loader,criterion,optimizer,num_epochs=50,patience=10)

    model.load_state_dict(torch.load('best_fuse.pth'))
    test_loss,test_iou,test_dice=evaluate_model(model,test_loader,criterion)
    print(f"Test Loss:{test_loss:.4f} IoU:{test_iou:.4f} Dice:{test_dice:.4f}")
