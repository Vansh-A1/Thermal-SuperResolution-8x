import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ================== PATHS ==================
LQ_VAL = "/home/projectwork/Deep_learning/thermal/val/LQ"
GT_VAL = "/home/projectwork/Deep_learning/thermal/val/HR"
SAVE_DIR = "/home/projectwork/Deep_learning/VANSH_WORK/srresnet_edge_aware"
MODEL_PATH = "/home/projectwork/Deep_learning/VANSH_WORK/frequency_srresnet_best_2.pth"

os.makedirs(SAVE_DIR, exist_ok=True)

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
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 9, padding=4)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(16)]
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
        res = self.res_blocks(x1)
        res = self.conv2(res)
        x = x1 + res
        x = self.upsample(x)
        return self.conv3(x)

# ================== SETUP ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SRResNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ================== TRANSFORMS ==================
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.5], std=[0.5])

# ================== METRIC STORAGE ==================
psnr_list = []
ssim_list = []
missing = 0

# ================== INFERENCE ==================
with torch.no_grad():
    for img_name in os.listdir(LQ_VAL):

        # ---- Load LR ----
        lr_path = os.path.join(LQ_VAL, img_name)
        lr = Image.open(lr_path).convert("L")

        lr_tensor = normalize(to_tensor(lr)).unsqueeze(0).to(device)

        # ---- Forward ----
        sr = model(lr_tensor)

        # ---- Denormalize ----
        sr = (sr.squeeze(0).cpu() + 1) / 2
        sr = torch.clamp(sr, 0, 1)

        # ---- Convert to numpy ----
        sr_np = (sr.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        # ---- Save ----
        save_name = img_name.replace(".bmp", "_out.bmp")
        Image.fromarray(sr_np.squeeze(), mode="L").save(
            os.path.join(SAVE_DIR, save_name)
        )

        # ---- Load GT ----
        gt_path = os.path.join(GT_VAL, img_name)
        if not os.path.exists(gt_path):
            print(f"Missing GT: {img_name}")
            missing += 1
            continue

        gt = Image.open(gt_path).convert("L")
        gt_np = np.array(gt)

        # ---- Resize (safety) ----
        if sr_np.shape != gt_np.shape:
            sr_np = cv2.resize(sr_np, (gt_np.shape[1], gt_np.shape[0]))

        # ---- Metrics ----
        psnr_list.append(psnr(gt_np, sr_np, data_range=255))
        ssim_list.append(ssim(gt_np, sr_np, data_range=255))

        print(f"Processed: {img_name}")

# ================== FINAL RESULTS ==================
print("\n========== RESULTS ==========")
print("Total images:", len(psnr_list))
print("Missing:", missing)
print("Average PSNR:", np.mean(psnr_list))
print("Average SSIM:", np.mean(ssim_list))