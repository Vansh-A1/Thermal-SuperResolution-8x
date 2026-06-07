import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from torchmetrics.image import StructuralSimilarityIndexMeasure
from PIL import Image
import lpips

torch.backends.cudnn.benchmark = True

# =========================
# CONFIG
# =========================
SCALE        = 8          # 8x super-resolution
BATCH_SIZE   = 4
NUM_EPOCHS   = 80
LR           = 2e-4
RRDB_BLOCKS  = 8
CHANNELS     = 64
SAVE_DIR     = "realesrgan_8x_weights"

os.makedirs(SAVE_DIR, exist_ok=True)

LQ_TRAIN = "/home/projectwork/Deep_learning/thermal/train/LQ"
GT_TRAIN = "/home/projectwork/Deep_learning/thermal/train/HR"
LQ_VAL   = "/home/projectwork/Deep_learning/thermal/val/LQ"
GT_VAL   = "/home/projectwork/Deep_learning/thermal/val/HR"

# =========================
# DATASET
# =========================
class SRDataset(Dataset):
    def __init__(self, lq_dir, gt_dir):
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        self.names  = sorted(os.listdir(lq_dir))
        self.to_tensor  = transforms.ToTensor()
        self.normalize  = transforms.Normalize(mean=[0.5], std=[0.5])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        lr = Image.open(os.path.join(self.lq_dir, name)).convert("L")
        hr = Image.open(os.path.join(self.gt_dir, name)).convert("L")
        return (
            self.normalize(self.to_tensor(lr)),
            self.normalize(self.to_tensor(hr)),
        )

# =========================
# GENERATOR — RRDB + 8x upsample
# =========================
class ResidualDenseBlock(nn.Module):
    """5-conv dense block with residual scaling."""
    def __init__(self, channels=CHANNELS, growth=32, scale=0.2):
        super().__init__()
        self.scale = scale
        self.c1 = nn.Conv2d(channels,           growth,   3, 1, 1)
        self.c2 = nn.Conv2d(channels + growth,  growth,   3, 1, 1)
        self.c3 = nn.Conv2d(channels + growth*2, growth,  3, 1, 1)
        self.c4 = nn.Conv2d(channels + growth*3, growth,  3, 1, 1)
        self.c5 = nn.Conv2d(channels + growth*4, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], 1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.c4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.c5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + self.scale * x5


class RRDB(nn.Module):
    def __init__(self, channels=CHANNELS, scale=0.2):
        super().__init__()
        self.scale = scale
        self.rdb1 = ResidualDenseBlock(channels)
        self.rdb2 = ResidualDenseBlock(channels)
        self.rdb3 = ResidualDenseBlock(channels)

    def forward(self, x):
        return x + self.scale * self.rdb3(self.rdb2(self.rdb1(x)))


class RRDBNet(nn.Module):
    """
    8x upsampling via 3 × pixel_shuffle(2):  2^3 = 8.
    Each sub-pixel conv expands C → 4C then shuffles back to C at 2× spatial.
    """
    def __init__(self, in_ch=1, out_ch=1, channels=CHANNELS, n_rrdb=RRDB_BLOCKS):
        super().__init__()
        self.conv_first  = nn.Conv2d(in_ch, channels, 3, 1, 1)
        self.body        = nn.Sequential(*[RRDB(channels) for _ in range(n_rrdb)])
        self.conv_body   = nn.Conv2d(channels, channels, 3, 1, 1)

        # 3 upsampling stages → 2×2×2 = 8×
        self.up1 = nn.Conv2d(channels, channels * 4, 3, 1, 1)  # → pixel_shuffle(2)
        self.up2 = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.up3 = nn.Conv2d(channels, channels * 4, 3, 1, 1)

        self.conv_hr   = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(channels, out_ch,  3, 1, 1)
        self.act       = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        fea  = self.conv_first(x)
        fea  = fea + self.conv_body(self.body(fea))

        fea  = self.act(F.pixel_shuffle(self.up1(fea), 2))   # 2×
        fea  = self.act(F.pixel_shuffle(self.up2(fea), 2))   # 4×
        fea  = self.act(F.pixel_shuffle(self.up3(fea), 2))   # 8×

        return self.conv_last(self.act(self.conv_hr(fea)))

# =========================
# DISCRIMINATOR — spectral norm + multiscale
# =========================
def _sn_conv(in_c, out_c, k, s, p):
    return nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, k, s, p))

class Discriminator(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.layers = nn.ModuleList([
            _sn_conv(in_ch, 64,  3, 1, 1),
            _sn_conv(64,    64,  4, 2, 1),
            _sn_conv(64,   128,  3, 1, 1),
            _sn_conv(128,  128,  4, 2, 1),
            _sn_conv(128,  256,  3, 1, 1),
            _sn_conv(256,  256,  4, 2, 1),
        ])
        self.act = nn.LeakyReLU(0.2, True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(256, 1)

    def forward(self, x):
        for l in self.layers:
            x = self.act(l(x))
        return self.fc(self.gap(x).flatten(1))

    def features(self, x):
        feats = []
        for l in self.layers:
            x = self.act(l(x))
            feats.append(x)
        return feats

# =========================
# LOSSES
# =========================
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:35].eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg

    def forward(self, sr, hr):
        # grayscale → 3-channel
        sr3 = sr.repeat(1, 3, 1, 1)
        hr3 = hr.repeat(1, 3, 1, 1)
        return F.l1_loss(self.vgg(sr3), self.vgg(hr3))


class DiffusiveLoss(nn.Module):
    """Gradient-domain loss to preserve edge sharpness."""
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[[[-1., 1.]]]])
        ky = torch.tensor([[[[-1.], [1.]]]])
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, sr, hr):
        sr_dx = F.conv2d(sr, self.kx, padding=(0, 1))
        sr_dy = F.conv2d(sr, self.ky, padding=(1, 0))
        hr_dx = F.conv2d(hr, self.kx, padding=(0, 1))
        hr_dy = F.conv2d(hr, self.ky, padding=(1, 0))
        return F.mse_loss(sr_dx, hr_dx) + F.mse_loss(sr_dy, hr_dy)


bce = nn.BCEWithLogitsLoss()

def ragan_g_loss(pred_fake, pred_real):
    """Relativistic average GAN loss for G."""
    return bce(pred_fake - pred_real.mean(0, keepdim=True),
               torch.ones_like(pred_fake))

def ragan_d_loss(pred_real, pred_fake):
    """Relativistic average GAN loss for D."""
    loss_real = bce(pred_real - pred_fake.mean(0, keepdim=True),
                    torch.ones_like(pred_real))
    loss_fake = bce(pred_fake - pred_real.mean(0, keepdim=True),
                    torch.zeros_like(pred_fake))
    return (loss_real + loss_fake) / 2

def feature_matching_loss(D, real, fake):
    real_feats = D.features(real.detach())
    fake_feats = D.features(fake)
    return sum(F.l1_loss(ff, rf.detach())
               for rf, ff in zip(real_feats, fake_feats))

# =========================
# METRICS
# =========================
def psnr(sr, hr, max_val=1.0):
    mse = F.mse_loss(sr, hr)
    return 10 * torch.log10(max_val ** 2 / mse)

# =========================
# DATA LOADERS
# =========================
train_dataset = SRDataset(LQ_TRAIN, GT_TRAIN)
val_dataset   = SRDataset(LQ_VAL,   GT_VAL)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    num_workers=2, pin_memory=True, persistent_workers=True
)

# =========================
# MODEL + LOSSES + OPTIMIZERS
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

G = RRDBNet().to(device)
D = Discriminator().to(device)

vgg_loss  = VGGLoss().to(device)
diff_loss = DiffusiveLoss().to(device)

ssim_metric  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_metric = lpips.LPIPS(net='alex').to(device)

opt_g = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.9, 0.99))
opt_d = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.9, 0.99))

# Cosine LR decay
sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=NUM_EPOCHS, eta_min=1e-6)
sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=NUM_EPOCHS, eta_min=1e-6)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# =========================
# VERIFY SIZE MATCH AT STARTUP
# =========================
with torch.no_grad():
    _lr, _hr = next(iter(train_loader))
    _sr = G(_lr.to(device))
    assert _sr.shape == _hr.to(device).shape, (
        f"Size mismatch: G outputs {_sr.shape}, but HR is {_hr.shape}. "
        f"Check that your HR images are exactly {SCALE}× the LQ images."
    )
    print(f"✅ Size check passed — LQ {tuple(_lr.shape)} → SR {tuple(_sr.shape)} == HR {tuple(_hr.shape)}")

# =========================
# TRAIN LOOP
# =========================
best_psnr = 0.0

for epoch in range(NUM_EPOCHS):
    G.train(); D.train()
    epoch_loss_g = epoch_loss_d = 0.0

    for lr_img, hr_img in train_loader:
        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)

        # ---------- Discriminator ----------
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            with torch.no_grad():
                sr_img = G(lr_img)

            pred_real = D(hr_img)
            pred_fake = D(sr_img.detach())
            loss_d = ragan_d_loss(pred_real, pred_fake)

        opt_d.zero_grad(set_to_none=True)
        scaler.scale(loss_d).backward()
        scaler.step(opt_d)

        # ---------- Generator ----------
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            sr_img = G(lr_img)

            l1   = F.l1_loss(sr_img, hr_img)
            vgg  = vgg_loss(sr_img, hr_img)
            diff = diff_loss(sr_img, hr_img)
            fm   = feature_matching_loss(D, hr_img, sr_img)

            pred_real_g = D(hr_img).detach()
            pred_fake_g = D(sr_img)
            gan = ragan_g_loss(pred_fake_g, pred_real_g)

            loss_g = l1 + 1.0 * vgg + 0.1 * fm + 0.005 * gan + 0.05 * diff

        opt_g.zero_grad(set_to_none=True)
        scaler.scale(loss_g).backward()
        scaler.step(opt_g)
        scaler.update()

        epoch_loss_g += loss_g.item()
        epoch_loss_d += loss_d.item()

    sched_g.step()
    sched_d.step()

    # =========================
    # VALIDATION
    # =========================
    G.eval()
    total_psnr = total_ssim = total_lpips = 0.0

    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            sr_img = G(lr_img)

            # Denormalize [-1,1] → [0,1]
            sr_01 = torch.clamp((sr_img + 1) / 2, 0, 1)
            hr_01 = torch.clamp((hr_img + 1) / 2, 0, 1)

            total_psnr  += psnr(sr_01, hr_01).item()
            total_ssim  += ssim_metric(sr_01, hr_01).item()
            total_lpips += lpips_metric(
                sr_01.repeat(1, 3, 1, 1),
                hr_01.repeat(1, 3, 1, 1)
            ).item()

    n = len(val_loader)
    avg_psnr  = total_psnr  / n
    avg_ssim  = total_ssim  / n
    avg_lpips = total_lpips / n

    n_train = len(train_loader)
    print(
        f"\nEpoch [{epoch+1:03d}/{NUM_EPOCHS}]  "
        f"Loss_G: {epoch_loss_g/n_train:.4f}  Loss_D: {epoch_loss_d/n_train:.4f}  "
        f"LR: {sched_g.get_last_lr()[0]:.2e}\n"
        f"  PSNR : {avg_psnr:.2f} dB\n"
        f"  SSIM : {avg_ssim:.4f}\n"
        f"  LPIPS: {avg_lpips:.4f}"
    )

    # Periodic checkpoint
    if (epoch + 1) % 10 == 0:
        ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1:03d}.pth")
        torch.save({"epoch": epoch+1, "G": G.state_dict(), "D": D.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                    "psnr": avg_psnr}, ckpt_path)

    # Best model
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(G.state_dict(), os.path.join(SAVE_DIR, "best_G.pth"))
        print("  🔥 Best model saved!")

# Final save
torch.save(G.state_dict(), os.path.join(SAVE_DIR, "last_G.pth"))
print("\n✅ TRAINING COMPLETE")
print(f"   Best PSNR: {best_psnr:.2f} dB")