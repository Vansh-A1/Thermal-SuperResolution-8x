import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from torchmetrics.image import StructuralSimilarityIndexMeasure
from PIL import Image
import os

torch.backends.cudnn.benchmark = True

# ===================== PATHS =====================
LQ_TRAIN = "/home/projectwork/Deep_learning/thermal/train/LQ"
GT_TRAIN = "/home/projectwork/Deep_learning/thermal/train/HR"
LQ_VAL   = "/home/projectwork/Deep_learning/thermal/val/LQ"
GT_VAL   = "/home/projectwork/Deep_learning/thermal/val/HR"

# ===================== DATASET =====================
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

# ===================== EDGE (SOBEL) =====================
def get_edges(x):
    sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], device=x.device, dtype=x.dtype)
    sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], device=x.device, dtype=x.dtype)

    edge_x = F.conv2d(x, sobel_x, padding=1)
    edge_y = F.conv2d(x, sobel_y, padding=1)

    return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)

# ===================== RESIDUAL BLOCK =====================
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

# ===================== MODEL =====================
class SRResNet_Edge(nn.Module):
    def __init__(self, scale=8, num_blocks=16):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 9, padding=4)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_blocks)]
        )

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        # Learnable edge scaling
        self.edge_weight = nn.Parameter(torch.tensor(0.1))

        # Upsampling ×8
        up_layers = []
        for _ in range(3):
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

        # 🔥 EDGE INJECTION
        edges = get_edges(x)
        edges = F.interpolate(edges, size=res.shape[-2:], mode='bilinear', align_corners=False)

        res = res + self.edge_weight * edges

        x = x1 + res
        x = self.upsample(x)
        x = self.conv3(x)

        return x

# ===================== LOSSES =====================
class VGG_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:35].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, sr, hr):
        sr = sr.repeat(1, 3, 1, 1)
        hr = hr.repeat(1, 3, 1, 1)
        return F.mse_loss(self.vgg(sr), self.vgg(hr))

class DiffusiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("kernel_x", torch.tensor([[[[-1, 1]]]], dtype=torch.float32))
        self.register_buffer("kernel_y", torch.tensor([[[[-1], [1]]]], dtype=torch.float32))

    def forward(self, sr, hr):
        kx = self.kernel_x.to(sr.device, sr.dtype)
        ky = self.kernel_y.to(sr.device, sr.dtype)

        sr_dx = F.conv2d(sr, kx, padding=(0,1))
        sr_dy = F.conv2d(sr, ky, padding=(1,0))

        hr_dx = F.conv2d(hr, kx, padding=(0,1))
        hr_dy = F.conv2d(hr, ky, padding=(1,0))

        return F.mse_loss(sr_dx, hr_dx) + F.mse_loss(sr_dy, hr_dy)

def total_loss(sr, hr, vgg_loss_fn, diff_loss_fn):
    l1 = F.l1_loss(sr, hr)
    vgg = vgg_loss_fn(sr, hr)
    edge = F.l1_loss(get_edges(sr), get_edges(hr))
    diff = diff_loss_fn(sr, hr)

    return (
        1.0 * l1 +
        0.01 * vgg +
        0.5 * edge +
        0.03 * diff
    )

# ===================== METRICS =====================
def calculate_psnr(sr, hr):
    sr = (sr + 1) / 2
    hr = (hr + 1) / 2
    mse = torch.mean((sr - hr) ** 2)
    return (10 * torch.log10(1.0 / mse)).item()

# ===================== DATA =====================
train_dataset = SRDataset(LQ_TRAIN, GT_TRAIN)
val_dataset = SRDataset(LQ_VAL, GT_VAL)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1)

# ===================== SETUP =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRResNet_Edge().to(device)
vgg_loss_fn = VGG_Loss().to(device)
diff_loss_fn = DiffusiveLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

best_ssim = 0
num_epochs = 80

# ===================== TRAIN =====================
for epoch in range(num_epochs):

    model.train()
    train_loss = 0

    for lr, hr in train_loader:
        lr, hr = lr.to(device), hr.to(device)

        sr = model(lr)
        loss = total_loss(sr, hr, vgg_loss_fn, diff_loss_fn)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------- VALIDATION --------
    model.eval()
    val_psnr = 0
    val_ssim = 0

    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)

            sr = model(lr)

            val_psnr += calculate_psnr(sr, hr)

            sr_img = (sr + 1) / 2
            hr_img = (hr + 1) / 2

            val_ssim += ssim_metric(sr_img, hr_img).item()

    val_psnr /= len(val_loader)
    val_ssim /= len(val_loader)

    # -------- SAVE BEST --------
    if val_ssim > best_ssim:
        best_ssim = val_ssim
        torch.save(model.state_dict(), "best_edge_model.pth")
        print("🔥 Best model saved!")

    print(f"""
Epoch [{epoch+1}/{num_epochs}]
Train Loss: {train_loss:.4f}
PSNR: {val_psnr:.2f}
SSIM: {val_ssim:.4f}
Edge Weight: {model.edge_weight.item():.4f}
""")

# Save last
torch.save(model.state_dict(), "last_edge_model.pth")
print("✅ Training complete!")