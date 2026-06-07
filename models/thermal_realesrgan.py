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
SCALE          = 8
BATCH_SIZE     = 4
NUM_EPOCHS     = 80
LR_G           = 2e-4
LR_D           = 5e-5
RRDB_BLOCKS    = 8
CHANNELS       = 64
GAN_WARMUP     = 15
GAN_RAMP       = 10
GAN_WEIGHT     = 0.1
R1_WEIGHT      = 10.0
D_STEPS        = 2
SAVE_DIR       = "realesrgan_8x_fixed"

LQ_TRAIN = "/home/projectwork/Deep_learning/thermal/train/LQ"
GT_TRAIN = "/home/projectwork/Deep_learning/thermal/train/HR"
LQ_VAL   = "/home/projectwork/Deep_learning/thermal/val/LQ"
GT_VAL   = "/home/projectwork/Deep_learning/thermal/val/HR"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# DATASET
# =========================
class SRDataset(Dataset):
    def __init__(self, lq_dir, gt_dir):
        self.lq_dir    = lq_dir
        self.gt_dir    = gt_dir
        self.names     = sorted(os.listdir(lq_dir))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        lr = Image.open(os.path.join(self.lq_dir, name)).convert("L")
        hr = Image.open(os.path.join(self.gt_dir, name)).convert("L")
        return self.normalize(self.to_tensor(lr)), self.normalize(self.to_tensor(hr))


# =========================
# GENERATOR
# =========================
class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=CHANNELS, growth=32, scale=0.2):
        super().__init__()
        self.scale = scale
        self.c1 = nn.Conv2d(channels,            growth,   3, 1, 1)
        self.c2 = nn.Conv2d(channels + growth,   growth,   3, 1, 1)
        self.c3 = nn.Conv2d(channels + growth*2, growth,   3, 1, 1)
        self.c4 = nn.Conv2d(channels + growth*3, growth,   3, 1, 1)
        self.c5 = nn.Conv2d(channels + growth*4, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, True)
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
        return x + self.scale * self.c5(torch.cat([x, x1, x2, x3, x4], 1))


class RRDB(nn.Module):
    def __init__(self, channels=CHANNELS):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels)
        self.rdb2 = ResidualDenseBlock(channels)
        self.rdb3 = ResidualDenseBlock(channels)

    def forward(self, x):
        return x + 0.2 * self.rdb3(self.rdb2(self.rdb1(x)))


class RRDBNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, channels=CHANNELS, n_rrdb=RRDB_BLOCKS):
        super().__init__()
        self.conv_first = nn.Conv2d(in_ch, channels, 3, 1, 1)
        self.body       = nn.Sequential(*[RRDB(channels) for _ in range(n_rrdb)])
        self.conv_body  = nn.Conv2d(channels, channels, 3, 1, 1)
        self.up1 = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.up2 = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.up3 = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.conv_hr   = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(channels, out_ch,  3, 1, 1)
        self.act       = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        fea = self.conv_first(x)
        fea = fea + self.conv_body(self.body(fea))
        fea = self.act(F.pixel_shuffle(self.up1(fea), 2))
        fea = self.act(F.pixel_shuffle(self.up2(fea), 2))
        fea = self.act(F.pixel_shuffle(self.up3(fea), 2))
        return self.conv_last(self.act(self.conv_hr(fea)))


# =========================
# DISCRIMINATOR
# =========================
def _sn(layer):
    return nn.utils.spectral_norm(layer)

class Discriminator(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        def block(ic, oc, stride):
            return nn.Sequential(
                _sn(nn.Conv2d(ic, oc, 4, stride, 1, bias=False)),
                nn.LeakyReLU(0.2, True),
            )
        self.layers = nn.ModuleList([
            block(in_ch, 64,  1),
            block(64,    64,  2),
            block(64,   128,  1),
            block(128,  128,  2),
            block(128,  256,  1),
            block(256,  256,  2),
            block(256,  512,  1),
        ])
        self.head = _sn(nn.Conv2d(512, 1, 4, 1, 1))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.head(x)

    def features(self, x):
        feats = []
        for l in self.layers:
            x = l(x)
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
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, sr, hr):
        sr = ((sr + 1) / 2).repeat(1, 3, 1, 1)
        hr = ((hr + 1) / 2).repeat(1, 3, 1, 1)
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        return F.l1_loss(self.vgg(sr), self.vgg(hr))


class DiffusiveLoss(nn.Module):
    def forward(self, sr, hr):
        dx_sr = sr[:, :, :, 1:] - sr[:, :, :, :-1]
        dx_hr = hr[:, :, :, 1:] - hr[:, :, :, :-1]
        dy_sr = sr[:, :, 1:, :] - sr[:, :, :-1, :]
        dy_hr = hr[:, :, 1:, :] - hr[:, :, :-1, :]
        return F.mse_loss(dx_sr, dx_hr) + F.mse_loss(dy_sr, dy_hr)


class FrequencyLoss(nn.Module):
    """
    FFT-based frequency loss.
    Always cast to float32 before fft2 — cuFFT in half precision only supports
    power-of-2 spatial sizes, but our HR images are 448×640 (not powers of 2).
    """
    def forward(self, sr, hr):
        sr_f = sr.float()          # safe cast: no-op if already float32
        hr_f = hr.float()
        sr_mag = torch.abs(torch.fft.rfft2(sr_f))   # rfft2: real-input optimised
        hr_mag = torch.abs(torch.fft.rfft2(hr_f))   # works on ANY spatial size
        return F.l1_loss(sr_mag, hr_mag)


# =========================
# GAN LOSSES
# =========================
bce = nn.BCEWithLogitsLoss()

def ragan_d_loss(pred_real, pred_fake):
    loss_real = bce(pred_real - pred_fake.mean(), torch.ones_like(pred_real))
    loss_fake = bce(pred_fake - pred_real.mean(), torch.zeros_like(pred_fake))
    return (loss_real + loss_fake) / 2

def ragan_g_loss(pred_fake, pred_real):
    return bce(pred_fake - pred_real.mean(), torch.ones_like(pred_fake))

def r1_penalty(D, real):
    """R1 gradient penalty — keeps D Lipschitz, prevents collapse."""
    real = real.detach().float().requires_grad_(True)   # must be float32 for autograd
    pred = D(real)
    grad = torch.autograd.grad(
        outputs=pred.sum(), inputs=real,
        create_graph=True, retain_graph=True
    )[0]
    return grad.pow(2).flatten(1).sum(1).mean()

def feature_matching_loss(D, real, fake):
    with torch.no_grad():
        real_feats = D.features(real)
    fake_feats = D.features(fake)
    return sum(F.l1_loss(ff, rf) for rf, ff in zip(real_feats, fake_feats))


# =========================
# METRICS
# =========================
def psnr(sr, hr):
    mse = F.mse_loss(sr, hr).clamp(min=1e-10)
    return (10 * torch.log10(1.0 / mse)).item()

def gan_weight_schedule(epoch):
    if epoch < GAN_WARMUP:
        return 0.0
    return GAN_WEIGHT * min((epoch - GAN_WARMUP) / GAN_RAMP, 1.0)


# =========================
# DATA
# =========================
train_loader = DataLoader(
    SRDataset(LQ_TRAIN, GT_TRAIN),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True
)
val_loader = DataLoader(
    SRDataset(LQ_VAL, GT_VAL),
    batch_size=1, shuffle=False,
    num_workers=2, pin_memory=True, persistent_workers=True
)

# =========================
# SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

G = RRDBNet().to(device)
D = Discriminator().to(device)

vgg_loss  = VGGLoss().to(device)
diff_loss = DiffusiveLoss().to(device)
freq_loss = FrequencyLoss().to(device)

ssim_metric  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_metric = lpips.LPIPS(net='alex').to(device)

opt_g = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.9, 0.99))
opt_d = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.9, 0.99))

sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=NUM_EPOCHS, eta_min=1e-6)
sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=NUM_EPOCHS, eta_min=1e-6)

scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

# Startup size check
with torch.no_grad():
    _lr, _hr = next(iter(train_loader))
    _sr = G(_lr.to(device))
    assert _sr.shape == _hr.to(device).shape, (
        f"Size mismatch: G({tuple(_lr.shape)}) → {tuple(_sr.shape)} ≠ HR {tuple(_hr.shape)}"
    )
    print(f"✅ Size check: LQ{tuple(_lr.shape)} → SR{tuple(_sr.shape)} == HR{tuple(_hr.shape)}")

# =========================
# TRAINING LOOP
# =========================
best_psnr = 0.0

for epoch in range(NUM_EPOCHS):
    G.train(); D.train()

    w_gan    = gan_weight_schedule(epoch)
    use_gan  = w_gan > 0.0
    epoch_loss_g = epoch_loss_d = 0.0

    for lr_img, hr_img in train_loader:
        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)

        # ---- Discriminator (only after warmup) ----
        if use_gan:
            for _ in range(D_STEPS):
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    with torch.no_grad():
                        sr_det = G(lr_img)
                    pred_real = D(hr_img)
                    pred_fake = D(sr_det.detach())
                    loss_d    = ragan_d_loss(pred_real, pred_fake)

                # R1 computed outside autocast so grads stay float32
                r1     = r1_penalty(D, hr_img)
                loss_d = loss_d + (R1_WEIGHT / 2) * r1

                opt_d.zero_grad(set_to_none=True)
                scaler.scale(loss_d).backward()
                scaler.step(opt_d)
                scaler.update()

            epoch_loss_d += loss_d.item()

        # ---- Generator ----
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            sr_img = G(lr_img)

            l1   = F.l1_loss(sr_img, hr_img)
            vgg  = vgg_loss(sr_img, hr_img)
            diff = diff_loss(sr_img, hr_img)
            freq = freq_loss(sr_img, hr_img)   # now safe under AMP

            loss_g = l1 + 1.0 * vgg + 0.05 * diff + 0.01 * freq

            if use_gan:
                fm          = feature_matching_loss(D, hr_img, sr_img)
                pred_real_g = D(hr_img).detach()
                pred_fake_g = D(sr_img)
                gan         = ragan_g_loss(pred_fake_g, pred_real_g)
                loss_g      = loss_g + 0.1 * fm + w_gan * gan

        opt_g.zero_grad(set_to_none=True)
        scaler.scale(loss_g).backward()
        scaler.step(opt_g)
        scaler.update()

        epoch_loss_g += loss_g.item()

    sched_g.step()
    sched_d.step()

    # ---- Validation ----
    G.eval()
    total_psnr = total_ssim = total_lpips = 0.0

    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            sr_img = G(lr_img)
            sr_01  = torch.clamp((sr_img + 1) / 2, 0, 1)
            hr_01  = torch.clamp((hr_img + 1) / 2, 0, 1)

            total_psnr  += psnr(sr_01, hr_01)
            total_ssim  += ssim_metric(sr_01, hr_01).item()
            total_lpips += lpips_metric(
                sr_01.repeat(1, 3, 1, 1),
                hr_01.repeat(1, 3, 1, 1)
            ).item()

    n        = len(val_loader)
    avg_psnr = total_psnr  / n
    avg_ssim = total_ssim  / n
    avg_lpips= total_lpips / n

    phase = "pretrain" if not use_gan else f"GAN w={w_gan:.3f}"
    print(
        f"\nEpoch [{epoch+1:03d}/{NUM_EPOCHS}] [{phase}]"
        f"  LossG={epoch_loss_g/len(train_loader):.4f}"
        f"  LossD={epoch_loss_d/(len(train_loader) if use_gan else 1):.4f}"
        f"  LR={sched_g.get_last_lr()[0]:.2e}"
        f"\n  PSNR={avg_psnr:.2f} dB  SSIM={avg_ssim:.4f}  LPIPS={avg_lpips:.4f}"
    )

    if (epoch + 1) % 10 == 0:
        torch.save({
            "epoch": epoch + 1,
            "G": G.state_dict(), "D": D.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
            "psnr": avg_psnr,
        }, os.path.join(SAVE_DIR, f"ckpt_epoch_{epoch+1:03d}.pth"))

    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        torch.save(G.state_dict(), os.path.join(SAVE_DIR, "best_G.pth"))
        print("  🔥 Best model saved!")

torch.save(G.state_dict(), os.path.join(SAVE_DIR, "last_G.pth"))
print(f"\n✅ Training complete — best PSNR: {best_psnr:.2f} dB")