import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import os

# ---------------- DATASET ----------------
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.names = sorted(os.listdir(lr_dir))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        lr = Image.open(os.path.join(self.lr_dir, name)).convert("RGB")
        hr = Image.open(os.path.join(self.hr_dir, name)).convert("RGB")

        return self.transform(lr), self.transform(hr)


# ---------------- METRICS ----------------
def psnr(fake, real):
    fake = (fake + 1) / 2
    real = (real + 1) / 2
    mse = F.mse_loss(fake, real)
    return 10 * torch.log10(1 / mse)

def ssim(fake, real):
    fake = (fake + 1) / 2
    real = (real + 1) / 2
    C1, C2 = 0.01**2, 0.03**2
    mu_x, mu_y = fake.mean(), real.mean()
    sigma_x, sigma_y = fake.var(), real.var()
    sigma_xy = ((fake - mu_x)*(real - mu_y)).mean()
    return ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))


# ---------------- MODEL (same as yours) ----------------
class ResidualBlock(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.BatchNorm2d(c),
            nn.PReLU(),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.BatchNorm2d(c)
        )
    def forward(self, x):
        return x + self.block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, c, scale):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c * scale**2, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.PReLU()
        )
    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(nn.Conv2d(3,64,9,1,4), nn.PReLU())
        self.residuals = nn.Sequential(*[ResidualBlock() for _ in range(16)])
        self.convblock = nn.Sequential(nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64))
        self.upsample = nn.Sequential(
            UpsampleBlock(64,2),
            UpsampleBlock(64,2),
            UpsampleBlock(64,2)
        )
        self.final = nn.Sequential(nn.Conv2d(64,3,9,1,4), nn.Tanh())

    def forward(self,x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x)
        x = x + initial
        x = self.upsample(x)
        return self.final(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i,o,s): return nn.Sequential(
            nn.Conv2d(i,o,3,s,1),
            nn.BatchNorm2d(o),
            nn.LeakyReLU(0.2)
        )

        self.net = nn.Sequential(
            nn.Conv2d(3,64,3,1,1), nn.LeakyReLU(0.2),
            block(64,64,2), block(64,128,1),
            block(128,128,2), block(128,256,1),
            block(256,256,2), block(256,512,1),
            block(512,512,2),
            nn.AdaptiveAvgPool2d((6,6)),
            nn.Flatten(),
            nn.Linear(512*6*6,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,1)
        )

    def forward(self,x): return self.net(x)

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features)[:36])
        for p in self.features.parameters(): p.requires_grad = False

    def forward(self,x):
        x = (x+1)/2
        return self.features(x)


# ---------------- SETUP ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

G = Generator().to(device)
D = Discriminator().to(device)
VGG = VGGFeatureExtractor().to(device).eval()

opt_G = torch.optim.Adam(G.parameters(), lr=1e-4)
opt_D = torch.optim.Adam(D.parameters(), lr=1e-4)

bce = nn.BCEWithLogitsLoss()
mse = nn.MSELoss()

# ---------------- DATA ----------------
train_loader = DataLoader(
    SRDataset(
        "/home/projectwork/Deep_learning/thermal/train/LR_x8",
        "/home/projectwork/Deep_learning/thermal/train/GT"
    ),
    batch_size=4, shuffle=True
)

val_loader = DataLoader(
    SRDataset(
        "/home/projectwork/Deep_learning/thermal/val/LR_x8",
        "/home/projectwork/Deep_learning/thermal/val/GT"
    ),
    batch_size=1, shuffle=False
)

# ---------------- VALIDATION ----------------
def validate():
    G.eval()
    total_psnr, total_ssim = 0, 0

    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            fake = G(lr)

            total_psnr += psnr(fake, hr).item()
            total_ssim += ssim(fake, hr).item()

    G.train()
    return total_psnr/len(val_loader), total_ssim/len(val_loader)


# ---------------- PHASE 1 ----------------
print("🔥 Pretraining...")
for epoch in range(15):
    for lr, hr in train_loader:
        lr, hr = lr.to(device), hr.to(device)

        fake = G(lr)
        loss = mse(fake, hr)

        opt_G.zero_grad()
        loss.backward()
        opt_G.step()

    psnr_val, ssim_val = validate()
    print(f"[PRE] Epoch {epoch} | Loss: {loss.item():.4f} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f}")

torch.save(G.state_dict(), "G_pretrained.pth")


# ---------------- PHASE 2 ----------------
print("🔥 SRGAN Training...")
best_psnr = 0

for epoch in range(50):
    for lr, hr in train_loader:
        lr, hr = lr.to(device), hr.to(device)

        fake = G(lr)

        # D
        real_out = D(hr)
        fake_out = D(fake.detach())
        loss_D = bce(real_out, torch.ones_like(real_out)*0.9) + \
                 bce(fake_out, torch.zeros_like(fake_out))

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # G
        fake_out = D(fake)
        adv_loss = bce(fake_out, torch.ones_like(fake_out))
        vgg_loss = mse(VGG(fake), VGG(hr))
        content_loss = mse(fake, hr)

        loss_G = content_loss + 1e-3*adv_loss + 2e-6*vgg_loss

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    psnr_val, ssim_val = validate()

    print(f"[GAN] Epoch {epoch} | G: {loss_G.item():.4f} | D: {loss_D.item():.4f} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f}")

    if psnr_val > best_psnr:
        best_psnr = psnr_val
        torch.save(G.state_dict(), "best_srgan.pth")