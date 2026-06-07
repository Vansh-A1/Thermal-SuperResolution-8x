import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from torchmetrics.image import StructuralSimilarityIndexMeasure

# =========================
# CONFIG
# =========================
MODEL_NAME = "thermal_realesrgan_v2"

MODEL_PATH = "/home/projectwork/Deep_learning/VANSH_WORK/models_thermal/my_model2/best_G.pth"

LQ_VAL = "/home/projectwork/Deep_learning/thermal/val/LQ"
GT_VAL = "/home/projectwork/Deep_learning/thermal/val/HR"

OUTPUT_DIR = f"/home/projectwork/Deep_learning/VANSH_WORK/models_thermal/my_model2/{MODEL_NAME}_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANNELS = 64
RRDB_BLOCKS = 8

# =========================
# DATASET
# =========================
class SRDataset(Dataset):
    def __init__(self, lq_dir, gt_dir):

        self.lq_dir = lq_dir
        self.gt_dir = gt_dir

        self.names = sorted(os.listdir(lq_dir))

        self.to_tensor = transforms.ToTensor()

        self.normalize = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        name = self.names[idx]

        lr = Image.open(
            os.path.join(self.lq_dir, name)
        ).convert("L")

        hr = Image.open(
            os.path.join(self.gt_dir, name)
        ).convert("L")

        lr = self.normalize(self.to_tensor(lr))
        hr = self.normalize(self.to_tensor(hr))

        return lr, hr, name


# =========================
# MODEL
# =========================
class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=CHANNELS, growth=32, scale=0.2):
        super().__init__()

        self.scale = scale

        self.c1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.c2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.c3 = nn.Conv2d(channels + growth * 2, growth, 3, 1, 1)
        self.c4 = nn.Conv2d(channels + growth * 3, growth, 3, 1, 1)
        self.c5 = nn.Conv2d(channels + growth * 4, channels, 3, 1, 1)

        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):

        x1 = self.act(self.c1(x))

        x2 = self.act(self.c2(torch.cat([x, x1], 1)))

        x3 = self.act(self.c3(torch.cat([x, x1, x2], 1)))

        x4 = self.act(self.c4(torch.cat([x, x1, x2, x3], 1)))

        return x + self.scale * self.c5(
            torch.cat([x, x1, x2, x3, x4], 1)
        )


class RRDB(nn.Module):
    def __init__(self, channels=CHANNELS):
        super().__init__()

        self.rdb1 = ResidualDenseBlock(channels)
        self.rdb2 = ResidualDenseBlock(channels)
        self.rdb3 = ResidualDenseBlock(channels)

    def forward(self, x):

        return x + 0.2 * self.rdb3(
            self.rdb2(
                self.rdb1(x)
            )
        )


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        channels=CHANNELS,
        n_rrdb=RRDB_BLOCKS
    ):
        super().__init__()

        self.conv_first = nn.Conv2d(
            in_ch,
            channels,
            3,
            1,
            1
        )

        self.body = nn.Sequential(
            *[RRDB(channels) for _ in range(n_rrdb)]
        )

        self.conv_body = nn.Conv2d(
            channels,
            channels,
            3,
            1,
            1
        )

        self.up1 = nn.Conv2d(
            channels,
            channels * 4,
            3,
            1,
            1
        )

        self.up2 = nn.Conv2d(
            channels,
            channels * 4,
            3,
            1,
            1
        )

        self.up3 = nn.Conv2d(
            channels,
            channels * 4,
            3,
            1,
            1
        )

        self.conv_hr = nn.Conv2d(
            channels,
            channels,
            3,
            1,
            1
        )

        self.conv_last = nn.Conv2d(
            channels,
            out_ch,
            3,
            1,
            1
        )

        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):

        fea = self.conv_first(x)

        fea = fea + self.conv_body(
            self.body(fea)
        )

        fea = self.act(
            F.pixel_shuffle(self.up1(fea), 2)
        )

        fea = self.act(
            F.pixel_shuffle(self.up2(fea), 2)
        )

        fea = self.act(
            F.pixel_shuffle(self.up3(fea), 2)
        )

        out = self.conv_last(
            self.act(self.conv_hr(fea))
        )

        return out


# =========================
# METRICS
# =========================
def psnr(sr, hr):

    mse = F.mse_loss(sr, hr).clamp(min=1e-10)

    return (
        10 * torch.log10(1.0 / mse)
    ).item()


# =========================
# DEVICE
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

# =========================
# MODEL LOAD
# =========================
model = RRDBNet().to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

print(f"Loaded model from: {MODEL_PATH}")

# =========================
# DATA
# =========================
val_loader = DataLoader(
    SRDataset(LQ_VAL, GT_VAL),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# =========================
# METRICS
# =========================
ssim_metric = StructuralSimilarityIndexMeasure(
    data_range=1.0
).to(device)

total_psnr = 0.0
total_ssim = 0.0

# =========================
# INFERENCE
# =========================
with torch.no_grad():

    for lr_img, hr_img, name in val_loader:

        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        sr_img = model(lr_img)

        sr_01 = torch.clamp(
            (sr_img + 1) / 2,
            0,
            1
        )

        hr_01 = torch.clamp(
            (hr_img + 1) / 2,
            0,
            1
        )

        cur_psnr = psnr(sr_01, hr_01)

        cur_ssim = ssim_metric(
            sr_01,
            hr_01
        ).item()

        total_psnr += cur_psnr
        total_ssim += cur_ssim

        # =========================
        # SAVE IMAGE
        # =========================
        save_path = os.path.join(
            OUTPUT_DIR,
            name[0]
        )

        save_image(sr_01, save_path)

        print(
            f"{name[0]} | "
            f"PSNR: {cur_psnr:.2f} | "
            f"SSIM: {cur_ssim:.4f}"
        )

# =========================
# FINAL RESULTS
# =========================
n = len(val_loader)

avg_psnr = total_psnr / n
avg_ssim = total_ssim / n

print("\n=========================")
print(f"Average PSNR : {avg_psnr:.2f} dB")
print(f"Average SSIM : {avg_ssim:.4f}")
print("=========================")