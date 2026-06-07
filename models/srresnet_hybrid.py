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
# ---------------- DATASET ----------------
class SRDataset(Dataset):
    def __init__(self, lq_dir, gt_dir):
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

        self.names = sorted(os.listdir(lq_dir))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        lr = Image.open(os.path.join(self.lq_dir, name)).convert("L")
        hr = Image.open(os.path.join(self.gt_dir, name)).convert("L")

        lr = self.normalize(self.to_tensor(lr))
        hr = self.normalize(self.to_tensor(hr))

        return lr, hr


# ---------------- RESIDUAL BLOCK ----------------
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


# ---------------- SRRESNET ----------------
class SRResNet(nn.Module):
    def __init__(self, scale=8, num_blocks=16):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 9, padding=4)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_blocks)]
        )

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        # Upsampling (PixelShuffle ×3 → 8x)
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

        x = x1 + res
        x = self.upsample(x)
        x = self.conv3(x)

        return x


# ---------------- VGG LOSS ----------------
class VGG_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:35].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, sr, hr):
        # grayscale → 3-channel
        sr = sr.repeat(1, 3, 1, 1)
        hr = hr.repeat(1, 3, 1, 1)
        return F.mse_loss(self.vgg(sr), self.vgg(hr))


# ---------------- DIFFUSIVE LOSS ----------------
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


# ---------------- TOTAL LOSS ----------------
def total_loss(sr, hr, vgg_loss_fn, diff_loss_fn):
    l1 = F.l1_loss(sr, hr)
    vgg = vgg_loss_fn(sr, hr)
    diff = diff_loss_fn(sr, hr)

    return l1 + 0.01 * vgg + 0.05 * diff


# ---------------- PSNR ----------------
def calculate_psnr(sr, hr):
    sr = (sr + 1) / 2
    hr = (hr + 1) / 2
    mse = torch.mean((sr - hr) ** 2)
    return (10 * torch.log10(1.0 / mse)).item()


# ---------------- DATA ----------------
train_dataset = SRDataset(
    "/home/projectwork/Deep_learning/thermal/train/LQ",
    "/home/projectwork/Deep_learning/thermal/train/HR"
)

val_dataset = SRDataset(
    "/home/projectwork/Deep_learning/thermal/val/LQ",
    "/home/projectwork/Deep_learning/thermal/val/HR"
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)


# ---------------- SETUP ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRResNet(scale=8).to(device)
vgg_loss_fn = VGG_Loss().to(device)
diff_loss_fn = DiffusiveLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

best_ssim = 0
num_epochs = 80


# ---------------- TRAIN LOOP ----------------
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

    if val_ssim > best_ssim:
        best_ssim = val_ssim
        torch.save(model.state_dict(), "srresnet_diffusive_best.pth")
        print("🔥 Best model saved!")

    print(f"""
Epoch [{epoch+1}/{num_epochs}]
Train Loss: {train_loss:.4f}
PSNR: {val_psnr:.2f}
SSIM: {val_ssim:.4f}
""")

torch.save(model.state_dict(), "srresnet_diffusive_last.pth")
print("✅ Training complete!")