import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

# ===== PATHS =====
MODEL_PATH = "/home/projectwork/Deep_learning/VANSH_WORK/best_edge_model.pth"

LQ_VAL = "/home/projectwork/Deep_learning/thermal/val/LQ"
GT_VAL = "/home/projectwork/Deep_learning/thermal/val/HR"

SAVE_DIR = "/home/projectwork/Deep_learning/VANSH_WORK/edge_injection/output2"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===================== EDGE =====================
def get_edges(x):
    sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], device=x.device, dtype=x.dtype)
    sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], device=x.device, dtype=x.dtype)

    edge_x = F.conv2d(x, sobel_x, padding=1)
    edge_y = F.conv2d(x, sobel_y, padding=1)

    return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)

# ===================== MODEL =====================
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class SRResNet_Edge(nn.Module):
    def __init__(self, scale=8, num_blocks=16):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 9, padding=4)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_blocks)]
        )

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)

        self.edge_weight = nn.Parameter(torch.tensor(0.1))

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

        edges = get_edges(x)
        edges = F.interpolate(edges, size=res.shape[-2:], mode='bilinear', align_corners=False)

        res = res + self.edge_weight * edges

        x = x1 + res
        x = self.upsample(x)
        x = self.conv3(x)

        return x

# ===== LOAD MODEL =====
model = SRResNet_Edge().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Model loaded successfully!")

# ===== METRICS =====
def calculate_psnr(sr, hr):
    mse = np.mean((sr - hr) ** 2)
    return 10 * np.log10(1.0 / mse)

def calculate_ssim(img1, img2):
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, data_range=1.0)

# ===== EVALUATION =====
psnr_list = []
ssim_list = []

files = sorted(os.listdir(LQ_VAL))

for i, f in enumerate(files):

    lq_path = os.path.join(LQ_VAL, f)
    gt_path = os.path.join(GT_VAL, f)

    if not os.path.exists(gt_path):
        continue

    # ---- LOAD ----
    lq = Image.open(lq_path).convert("L")
    gt = Image.open(gt_path).convert("L")

    lq_np = np.array(lq).astype(np.float32) / 255.0
    gt_np = np.array(gt).astype(np.float32) / 255.0

    # ---- NORMALIZATION (IMPORTANT) ----
    lq_np = (lq_np - 0.5) / 0.5
    gt_np = (gt_np - 0.5) / 0.5

    lq_tensor = torch.from_numpy(lq_np).unsqueeze(0).unsqueeze(0).to(device)

    # ---- INFERENCE ----
    with torch.no_grad():
        sr = model(lq_tensor)

    sr_np = sr.squeeze().cpu().numpy()

    # ---- DENORMALIZE ----
    sr_np = (sr_np + 1) / 2
    gt_np = (gt_np + 1) / 2

    sr_np = np.clip(sr_np, 0, 1)

    # ---- CROP ----
    scale = 8
    crop = scale

    if sr_np.shape[0] > 2*crop:
        sr_crop = sr_np[crop:-crop, crop:-crop]
        gt_crop = gt_np[crop:-crop, crop:-crop]
    else:
        sr_crop = sr_np
        gt_crop = gt_np

    # ---- METRICS ----
    psnr_val = calculate_psnr(sr_crop, gt_crop)
    ssim_val = calculate_ssim(sr_crop, gt_crop)

    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

    # ---- SAVE IMAGE ----
    sr_img = (sr_np * 255).astype(np.uint8)

    save_name = f"edge_output_{os.path.splitext(f)[0]}.png"
    save_path = os.path.join(SAVE_DIR, save_name)

    Image.fromarray(sr_img).save(save_path)

    print(f"[{i+1}/{len(files)}] {f} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f}")

# ===== FINAL =====
print("\n========== FINAL RESULTS ==========")
print(f"Average PSNR: {np.mean(psnr_list):.4f}")
print(f"Average SSIM: {np.mean(ssim_list):.4f}")