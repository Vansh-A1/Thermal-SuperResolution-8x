import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pytorch_msssim import ssim
import math

torch.cuda.empty_cache()


class SRCNN_DATA(Dataset):
    def __init__(self, lq_dir, gt_dir):
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        
        self.to_tensor = transforms.ToTensor()  
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5]) 
        
        self.image_name = sorted(os.listdir(lq_dir))  
        
        self.upsample = transforms.Resize(
            (448, 640),   
            interpolation=transforms.InterpolationMode.BICUBIC
        )                       
    
    def __len__(self):
        return len(self.image_name)   
    
    def __getitem__(self, idx):   
        filename = self.image_name[idx]
        
        lr_path = os.path.join(self.lq_dir, filename)  
        gt_path = os.path.join(self.gt_dir, filename)   

        lr_image = Image.open(lr_path).convert("L")
        gt_image = Image.open(gt_path).convert("L")

        lr_image = self.upsample(lr_image)   

        lr_image = self.to_tensor(lr_image)
        gt_image = self.to_tensor(gt_image)

        lr_image = self.normalize(lr_image)
        gt_image = self.normalize(gt_image)
            
        return lr_image, gt_image



class SRCNN(nn.Module):
    def __init__(self, num_channels=1):  
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x



def denormalize(x):
    return (x * 0.5) + 0.5   # [-1,1] → [0,1]

def calculate_psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return 100
    return 10 * torch.log10(1.0 / mse)

def calculate_ssim(sr, hr):
    return ssim(sr, hr, data_range=1.0, size_average=True)



train_dataset = SRCNN_DATA(
    lq_dir="/home/projectwork/Deep_learning/thermal/train/LQ",
    gt_dir="/home/projectwork/Deep_learning/thermal/train/HR"
)

val_dataset = SRCNN_DATA(
    lq_dir="/home/projectwork/Deep_learning/thermal/val/LQ",
    gt_dir="/home/projectwork/Deep_learning/thermal/val/HR"
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRCNN().to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 100   

best_psnr = 0



for epoch in range(num_epochs):
    
    
    model.train()
    train_loss = 0

    for lr, gt in train_loader:
        lr, gt = lr.to(device), gt.to(device)

        sr = model(lr)
        loss = criterion(sr, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)


    
    model.eval()
    val_psnr = 0
    val_ssim = 0

    with torch.no_grad():
        for lr, gt in val_loader:
            lr, gt = lr.to(device), gt.to(device)

            sr = model(lr)

            
            sr = denormalize(sr)
            gt = denormalize(gt)

            sr = torch.clamp(sr, 0, 1)

            val_psnr += calculate_psnr(sr, gt)
            val_ssim += calculate_ssim(sr, gt)

    avg_psnr = val_psnr / len(val_loader)
    avg_ssim = val_ssim / len(val_loader)


    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(model.state_dict(), "best_SRCNN.pth")
        print("🔥 Best model updated!")

    print(f"""
Epoch [{epoch+1}/{num_epochs}]
Train Loss: {avg_train_loss:.4f}
Val PSNR: {avg_psnr:.2f}
Val SSIM: {avg_ssim:.4f}
""")


torch.save(model.state_dict(), "last_SRCNN.pth")
print(" Training complete!")