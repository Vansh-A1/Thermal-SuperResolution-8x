import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# ---------------- CONFIG ----------------
LR_DIR = "/home/projectwork/Deep_learning/thermal/val/LQ"
HR_DIR = "/home/projectwork/Deep_learning/thermal/val/HR"

MODEL_PATH = "/home/projectwork/Deep_learning/VANSH_WORK/models_thermal/sr_gan/best_srgan.pth"
# OPTIONAL:
# MODEL_PATH = "/home/projectwork/Deep_learning/VANSH_WORK/models_thermal/sr_gan/G_pretrained.pth"

SAVE_DIR = "./sr_gan_output"

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def denorm(t):
    return (t + 1) / 2

# ---------------- METRICS ----------------
def psnr(fake, real):
    fake = denorm(fake)
    real = denorm(real)
    mse = F.mse_loss(fake, real)
    return 10 * torch.log10(1 / mse)

def ssim(fake, real):
    fake = denorm(fake)
    real = denorm(real)

    C1, C2 = 0.01**2, 0.03**2
    mu_x, mu_y = fake.mean(), real.mean()
    sigma_x, sigma_y = fake.var(), real.var()
    sigma_xy = ((fake - mu_x)*(real - mu_y)).mean()

    return ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))

# ---------------- MODEL ----------------
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
        self.initial = nn.Sequential(nn.Conv2d(3, 64, 9, 1, 4), nn.PReLU())
        self.residuals = nn.Sequential(*[ResidualBlock() for _ in range(16)])
        self.convblock = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))
        self.upsample = nn.Sequential(
            UpsampleBlock(64, 2),
            UpsampleBlock(64, 2),
            UpsampleBlock(64, 2)
        )
        self.final = nn.Sequential(nn.Conv2d(64, 3, 9, 1, 4), nn.Tanh())

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x)
        x = x + initial
        x = self.upsample(x)
        return self.final(x)

# ---------------- LOAD MODEL ----------------
G = Generator().to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)

# 🔥 handle both formats
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    G.load_state_dict(checkpoint["state_dict"])
elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    G.load_state_dict(checkpoint["model_state_dict"])
else:
    G.load_state_dict(checkpoint)

G.eval()
print(f"✅ Loaded model: {MODEL_PATH}")

# ---------------- INFERENCE ----------------
lr_files = sorted(os.listdir(LR_DIR))

total_psnr, total_ssim = 0, 0
count = 0

with torch.no_grad():
    for file in lr_files:

        lr_path = os.path.join(LR_DIR, file)
        hr_path = os.path.join(HR_DIR, file)

        if not os.path.exists(hr_path):
            print(f"⚠️ Skipping {file} (no GT)")
            continue

        # load
        lr = transform(Image.open(lr_path).convert("RGB")).unsqueeze(0).to(device)
        hr = transform(Image.open(hr_path).convert("RGB")).unsqueeze(0).to(device)

        # inference
        sr = G(lr)

        # resize GT if needed
        if sr.shape != hr.shape:
            hr = F.interpolate(hr, size=sr.shape[-2:], mode='bilinear', align_corners=False)

        # metrics
        p = psnr(sr, hr).item()
        s = ssim(sr, hr).item()

        total_psnr += p
        total_ssim += s
        count += 1

        print(f"{file} -> PSNR: {p:.2f}, SSIM: {s:.4f}")

        # save image
        sr_img = denorm(sr.squeeze()).cpu().permute(1,2,0).numpy()
        sr_img = (sr_img * 255).clip(0,255).astype("uint8")

        Image.fromarray(sr_img).save(os.path.join(SAVE_DIR, file))

# ---------------- FINAL ----------------
if count > 0:
    print("\n🔥 FINAL RESULTS")
    print(f"Images: {count}")
    print(f"Average PSNR: {total_psnr/count:.2f}")
    print(f"Average SSIM: {total_ssim/count:.4f}")
    print(f"📁 Saved at: {SAVE_DIR}")
else:
    print("❌ No valid pairs found")