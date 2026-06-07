import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ---------------- MODEL ----------------
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
        return self.pixel_shuffle(self.body(x))


# ---------------- PSNR ----------------
def calculate_psnr(output, target):
    output = (output + 1) / 2
    target = (target + 1) / 2
    mse = torch.mean((output - target) ** 2)
    if mse == 0:
        return 100
    return (10 * torch.log10(1.0 / mse)).item()


# ---------------- PATHS ----------------
model_path = "/home/projectwork/Deep_learning/VANSH_WORK/models_thermal/vdslr_hybrid/vdsr_hybrid_best.pth"
input_folder = "/home/projectwork/Deep_learning/thermal/val/LQ"
gt_folder = "/home/projectwork/Deep_learning/thermal/val/HR"   # 🔥 REQUIRED
output_folder = "/home/projectwork/Deep_learning/output_hybrid_vdsr"

os.makedirs(output_folder, exist_ok=True)


# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- LOAD MODEL ----------------
model = VDSR_PixelShuffle(scale=8).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# ---------------- TRANSFORMS ----------------
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.5], std=[0.5])

ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


# ---------------- METRIC STORAGE ----------------
total_psnr = 0
total_ssim = 0
count = 0


# ---------------- INFERENCE LOOP ----------------
with torch.no_grad():
    for name in sorted(os.listdir(input_folder)):

        lr_path = os.path.join(input_folder, name)
        hr_path = os.path.join(gt_folder, name)

        if not os.path.exists(hr_path):
            continue  # skip if GT not found

        # Load images
        lr_img = Image.open(lr_path).convert("L")
        hr_img = Image.open(hr_path).convert("L")

        # Preprocess
        lr = normalize(to_tensor(lr_img)).unsqueeze(0).to(device)
        hr = normalize(to_tensor(hr_img)).unsqueeze(0).to(device)

        # Forward
        sr = model(lr)

        # -------- METRICS --------
        psnr = calculate_psnr(sr, hr)

        sr_img_norm = (sr + 1) / 2
        hr_img_norm = (hr + 1) / 2

        ssim = ssim_metric(sr_img_norm, hr_img_norm).item()

        total_psnr += psnr
        total_ssim += ssim
        count += 1

        # -------- SAVE IMAGE --------
        sr_save = (sr.squeeze(0).cpu() + 1) / 2
        sr_save = torch.clamp(sr_save, 0, 1)
        sr_pil = transforms.ToPILImage()(sr_save)

        save_path = os.path.join(output_folder, name)
        sr_pil.save(save_path)

        print(f"{name} → PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")


# ---------------- FINAL AVERAGE ----------------
avg_psnr = total_psnr / count
avg_ssim = total_ssim / count

print("\n🔥 FINAL RESULTS")
print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.4f}")