import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure
from PIL import Image
import os
import lpips

# =========================
# PATHS
# =========================
model_path = "/home/projectwork/Deep_learning/VANSH_WORK/models_thermal/srresnet_diffusion/srresnet_diffusive_best.pth"

lq_dir = "/home/projectwork/Deep_learning/thermal/val/LQ"
hr_dir = "/home/projectwork/Deep_learning/thermal/val/HR"

save_dir = "SRRESNET+DIFFUSION_OUTPUT"
os.makedirs(save_dir, exist_ok=True)


# =========================
# MODEL (same as training)
# =========================
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
        res = self.conv2(self.res_blocks(x1))
        x = x1 + res
        x = self.upsample(x)
        return self.conv3(x)


# =========================
# METRICS
# =========================
def calculate_psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2)
    return 10 * torch.log10(1.0 / mse)


# =========================
# LOAD MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRResNet().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_metric = lpips.LPIPS(net='alex').to(device)


# =========================
# TRANSFORMS
# =========================
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.5], std=[0.5])


# =========================
# INFERENCE LOOP
# =========================
psnr_total = 0
ssim_total = 0
lpips_total = 0
count = 0

image_names = sorted(os.listdir(lq_dir))

with torch.no_grad():
    for name in image_names:

        # ---- LOAD ----
        lr = Image.open(os.path.join(lq_dir, name)).convert("L")
        hr = Image.open(os.path.join(hr_dir, name)).convert("L")

        lr = normalize(to_tensor(lr)).unsqueeze(0).to(device)
        hr = normalize(to_tensor(hr)).unsqueeze(0).to(device)

        # ---- INFERENCE ----
        sr = model(lr)

        # ---- DENORMALIZE ----
        sr_img = torch.clamp((sr + 1) / 2, 0, 1)
        hr_img = torch.clamp((hr + 1) / 2, 0, 1)

        # ---- SAVE OUTPUT ----
        save_path = os.path.join(save_dir, name)
        transforms.ToPILImage()(sr_img.squeeze(0).cpu()).save(save_path)

        # ---- METRICS ----
        psnr = calculate_psnr(sr_img, hr_img)
        ssim = ssim_metric(sr_img, hr_img)

        # LPIPS needs 3-channel
        sr_lp = sr_img.repeat(1,3,1,1)
        hr_lp = hr_img.repeat(1,3,1,1)

        lp = lpips_metric(sr_lp, hr_lp)

        psnr_total += psnr.item()
        ssim_total += ssim.item()
        lpips_total += lp.item()
        count += 1


# =========================
# FINAL RESULTS
# =========================
print("\n===== FINAL RESULTS =====")
print(f"Average PSNR : {psnr_total / count:.2f}")
print(f"Average SSIM : {ssim_total / count:.4f}")
print(f"Average LPIPS: {lpips_total / count:.4f}")