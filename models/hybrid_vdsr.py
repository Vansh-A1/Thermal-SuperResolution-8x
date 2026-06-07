import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19
from torchmetrics.image import StructuralSimilarityIndexMeasure
from PIL import Image
import os

torch.cuda.empty_cache()


def calculate_psnr(output, target):
    output = (output + 1) / 2
    target = (target + 1) / 2
    mse = torch.mean((output - target) ** 2)
    if mse == 0:
        return 100
    return (10 * torch.log10(1.0 / mse)).item()


class VDSR_DATA(Dataset):
    def __init__(self, lq_dir, gt_dir):
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

        self.image_names = sorted(os.listdir(lq_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]

        lr = Image.open(os.path.join(self.lq_dir, name)).convert("L")
        hr = Image.open(os.path.join(self.gt_dir, name)).convert("L")

        lr = self.normalize(self.to_tensor(lr))
        hr = self.normalize(self.to_tensor(hr))

        return lr, hr



class VDSR_PixelShuffle(nn.Module):
    def __init__(self, scale=8):
        super().__init__()
        self.scale = scale

        layers = []
        layers.append(nn.Conv2d(1, 64, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(18):
            layers.append(nn.Conv2d(64, 64, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, scale * scale, 3, padding=1))

        self.body = nn.Sequential(*layers)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        out = self.body(x)
        out = self.pixel_shuffle(out)
        return out



class VGG_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features[:35].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, sr, hr):
        sr = sr.repeat(1, 3, 1, 1)
        hr = hr.repeat(1, 3, 1, 1)
        return F.mse_loss(self.vgg(sr), self.vgg(hr))



def diffusive_patch_loss(sr, hr):
    kernel = torch.ones((1, 1, 3, 3), device=sr.device) / 9.0
    sr_blur = F.conv2d(sr, kernel, padding=1)
    hr_blur = F.conv2d(hr, kernel, padding=1)
    return F.mse_loss(sr_blur, hr_blur)



def total_loss(sr, hr, vgg_loss_fn):
    l2 = F.mse_loss(sr, hr)
    vgg = vgg_loss_fn(sr, hr)
    diff = diffusive_patch_loss(sr, hr)

    return l2 + 0.01 * vgg + 0.1 * diff



train_dataset = VDSR_DATA(
    "/home/projectwork/Deep_learning/thermal/train/LQ",
    "/home/projectwork/Deep_learning/thermal/train/HR"
)

val_dataset = VDSR_DATA(
    "/home/projectwork/Deep_learning/thermal/val/LQ",
    "/home/projectwork/Deep_learning/thermal/val/HR"
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VDSR_PixelShuffle(scale=8).to(device)
vgg_loss_fn = VGG_Loss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

num_epochs = 60
best_ssim = 0



for epoch in range(num_epochs):

    model.train()
    train_loss = 0

    for lr, hr in train_loader:
        lr, hr = lr.to(device), hr.to(device)

        sr = model(lr)

        loss = total_loss(sr, hr, vgg_loss_fn)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    
    model.eval()
    val_psnr = 0
    val_ssim = 0

    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr)

            psnr = calculate_psnr(sr, hr)
            val_psnr += psnr

            sr_img = (sr + 1) / 2
            hr_img = (hr + 1) / 2

            val_ssim += ssim_metric(sr_img, hr_img).item()

    val_psnr /= len(val_loader)
    val_ssim /= len(val_loader)

    
    if val_ssim > best_ssim:
        best_ssim = val_ssim
        torch.save(model.state_dict(), "vdsr_hybrid_best.pth")
        print(" Best model saved!")

    print(f"""
Epoch [{epoch+1}/{num_epochs}]
Train Loss: {train_loss:.4f}
PSNR: {val_psnr:.2f}
SSIM: {val_ssim:.4f}
""")

# ---------------- FINAL SAVE ----------------
torch.save(model.state_dict(), "vdsr_hybrid_last.pth")
print("✅ Training complete!")