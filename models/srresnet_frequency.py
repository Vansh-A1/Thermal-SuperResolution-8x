import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure
from PIL import Image
import os

torch.backends.cudnn.benchmark = True

# ================== PATHS ==================
LQ_TRAIN = "/home/projectwork/Deep_learning/thermal/train/LQ"
GT_TRAIN = "/home/projectwork/Deep_learning/thermal/train/HR"
LQ_VAL   = "/home/projectwork/Deep_learning/thermal/val/LQ"
GT_VAL   = "/home/projectwork/Deep_learning/thermal/val/HR"

# ================== DATASET ==================
class SRDataset(Dataset):
    def __init__(self, lq_dir, gt_dir):
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        self.names = sorted(os.listdir(lq_dir))

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        lr = Image.open(os.path.join(self.lq_dir, name)).convert("L")
        hr = Image.open(os.path.join(self.gt_dir, name)).convert("L")

        lr = self.normalize(self.to_tensor(lr))
        hr = self.normalize(self.to_tensor(hr))

        return lr, hr

# ================== MODEL ==================
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class SRResNet(nn.Module):
    def __init__(self, scale=8, num_blocks=16):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 9, padding=4)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_blocks)]
        )

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        up_layers = []
        for _ in range(3):  # 2^3 = 8x
            up_layers += [
                nn.Conv2d(64, 256, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ]
        self.upsample = nn.Sequential(*up_layers)

        self.conv3 = nn.Conv2d(64, 1, 9, padding=4)

    def forward(self, x):
        x1 = self.conv1(x)
        res = self.res_blocks(x1)
        res = self.conv2(res)
        x = x1 + res
        x = self.upsample(x)
        return self.conv3(x)

# ================== LOSSES ==================

# --- 1. L1 (Structure) ---
def l1_loss(sr, hr):
    return F.l1_loss(sr, hr)

# --- 2. Edge-aware diffusion ---
class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.sobel_x = torch.tensor(
            [[[-1,0,1],[-2,0,2],[-1,0,1]]], dtype=torch.float32
        ).unsqueeze(0)

        self.sobel_y = torch.tensor(
            [[[-1,-2,-1],[0,0,0],[1,2,1]]], dtype=torch.float32
        ).unsqueeze(0)

    def forward(self, sr, hr):
        kx = self.sobel_x.to(sr.device, sr.dtype)
        ky = self.sobel_y.to(sr.device, sr.dtype)

        sr_dx = F.conv2d(sr, kx, padding=1)
        hr_dx = F.conv2d(hr, kx, padding=1)

        sr_dy = F.conv2d(sr, ky, padding=1)
        hr_dy = F.conv2d(hr, ky, padding=1)

        grad_hr = torch.sqrt(hr_dx**2 + hr_dy**2)

        # edge-aware weighting
        weight = torch.exp(-5 * grad_hr)

        loss = weight * ((sr_dx - hr_dx)**2 + (sr_dy - hr_dy)**2)
        return loss.mean()

# --- 3. FFT loss (texture) ---
def fft_loss(sr, hr):
    sr_fft = torch.fft.fft2(sr)
    hr_fft = torch.fft.fft2(hr)
    return F.l1_loss(torch.abs(sr_fft), torch.abs(hr_fft))

# --- TOTAL LOSS ---
def total_loss(sr, hr, edge_loss_fn):
    l1 = l1_loss(sr, hr)
    edge = edge_loss_fn(sr, hr)
    fft = fft_loss(sr, hr)

    return (
        1.0 * l1 +
        0.05 * edge +
        0.05* fft
    )

# ================== METRICS ==================
def calculate_psnr(sr, hr):
    sr = (sr + 1) / 2
    hr = (hr + 1) / 2
    mse = torch.mean((sr - hr) ** 2)
    return (10 * torch.log10(1.0 / mse)).item()

# ================== DATA ==================
train_loader = DataLoader(SRDataset(LQ_TRAIN, GT_TRAIN), batch_size=4, shuffle=True)
val_loader   = DataLoader(SRDataset(LQ_VAL, GT_VAL), batch_size=1)

# ================== SETUP ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRResNet().to(device)
edge_loss_fn = EdgeAwareLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# ================== TRAIN ==================
best_psnr = 0
epochs = 80

for epoch in range(epochs):

    # ----- TRAIN -----
    model.train()
    train_loss = 0

    for lr, hr in train_loader:
        lr, hr = lr.to(device), hr.to(device)

        sr = model(lr)
        loss = total_loss(sr, hr, edge_loss_fn)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ----- VALIDATION -----
    model.eval()
    val_psnr, val_ssim, val_loss = 0, 0, 0

    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr)

            val_loss += total_loss(sr, hr, edge_loss_fn).item()
            val_psnr += calculate_psnr(sr, hr)

            sr_img = (sr + 1) / 2
            hr_img = (hr + 1) / 2

            val_ssim += ssim_metric(sr_img, hr_img).item()

    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)
    val_ssim /= len(val_loader)

    # ----- SAVE BEST -----
    if val_psnr > best_psnr:
        best_psnr = val_psnr
        torch.save(model.state_dict(), "frequency_srresnet_best_2.pth")
        print("🔥 Saved best model")

    print(f"""
Epoch [{epoch+1}/{epochs}]
Train Loss: {train_loss:.4f}
Val Loss: {val_loss:.4f}
PSNR: {val_psnr:.2f}
SSIM: {val_ssim:.4f}
""")

torch.save(model.state_dict(), "last_frequency_srresnet_2.pth")
print("✅ Training complete")