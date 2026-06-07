# Thermal Image Super-Resolution: An 8× Experimental Framework

A systematic experimental codebase for 8× super-resolution of thermal (infrared) images. The repository benchmarks and extends multiple SR architectures — from shallow CNN baselines to GAN-based generative models — with thermal-specific loss engineering, edge-aware modifications, frequency-domain supervision, and diffusive gradient regularization.

---

## Project Overview

Thermal imaging produces single-channel, grayscale-like imagery where pixel intensity encodes radiated heat rather than reflected light. When thermal sensors operate at low resolution (as is common in budget or embedded systems), recovering high-resolution structure from a heavily downsampled input is considerably harder than in the RGB domain.

This repository addresses **8× super-resolution of thermal images** — a challenging scale factor that requires not only accurate reconstruction of intensity values, but preservation of thermally meaningful structures: sharp gradients between warm and cold regions, coherent boundary topology, and spatially smooth isothermal zones free of hallucinatory texture.

Standard SR methods fail in several characteristic ways on thermal data. RGB-trained perceptual models (VGG, LPIPS) apply feature priors that are miscalibrated to natural image statistics. GAN discriminators trained on natural image distributions will hallucinate RGB-like texture where smooth thermal gradients should exist. Pixel-space losses (MSE) produce over-smoothed outputs at 8×. This codebase empirically explores which modifications best address these failure modes.

---

## Key Research Goals

The repository explores several directions simultaneously:

**Baseline characterization** — SRCNN with L1 vs L2 loss establishes the minimum meaningful benchmark and directly tests whether reconstruction loss choice materially affects shallow-model performance at 8×.

**Residual network scaling** — SRResNet ported to grayscale single-channel thermal input, trained with progressively richer loss combinations, establishes how much residual depth helps before GAN-style perceptual training is needed.

**Hybrid loss engineering** — Multiple variants of SRResNet combine L1 reconstruction, VGG perceptual, diffusive gradient, and FFT frequency losses. The relative weighting of these terms across variants constitutes a structured ablation on loss composition.

**Learnable edge injection** — A novel architectural modification adds a learnable scalar gate that injects Sobel-derived edge maps directly into the residual feature representation before upsampling. This is an explicit mechanism to preserve thermal boundary structure without purely relying on loss-based supervision.

**GAN training with thermal-aware stabilization** — Two separate GAN variants (SRGAN-style and Real-ESRGAN-style) are implemented, each with progressively more stable training strategies: the Real-ESRGAN variant adds spectral normalization, R1 gradient penalty, relativistic average GAN losses, GAN warmup scheduling, and AMP-safe FFT losses.

**Frequency-domain supervision** — A dedicated FFT magnitude loss is introduced in the SRResNet-Frequency and Real-ESRGAN variants to supervise spectral content reconstruction. This is particularly motivated by the observation that thermal images have a distinct low-frequency-dominated spectral signature that L1/VGG losses do not adequately constrain.

---

## Repository Structure

```
Thermal_8x_super_resolution/
│
├── models/                         # Training scripts (model + training loop combined)
│   ├── srcnn.py                    # SRCNN baseline with MSE loss
│   ├── srcnn_try.py                # SRCNN with L1 loss (loss ablation)
│   ├── hybrid_vdsr.py              # VDSR + PixelShuffle + VGG + diffusive loss
│   ├── srresnet_hybrid.py          # SRResNet + VGG + diffusive loss
│   ├── srresnet_frequency.py       # SRResNet + L1 + edge-aware + FFT loss
│   ├── edge_injection.py           # SRResNet + learnable Sobel injection + multi-loss
│   ├── srgan_offical.py            # Standard SRGAN with two-phase training
│   ├── realesrgan_diffuision.py    # Real-ESRGAN-style RRDBNet + RaGAN + diffusive loss
│   └── thermal_realesrgan.py       # Full thermal Real-ESRGAN: R1, warmup, FrequencyLoss, AMP
│
├── inference/                      # Per-model inference scripts with PSNR/SSIM evaluation
│   ├── srcnn_l1_inference.py
│   ├── srcnn_l2_inference.py
│   ├── vdslr_hybrid_infernce.py
│   ├── SRGAN_INFERENCE.py
│   ├── inference_edge_injection.py
│   └── inference_srresnet_frequency.py
│
├── output_samples/                 # Visual results on 4 validation images, per model
│   ├── srcnnL1/
│   ├── srcnnl2/
│   ├── srresnet_hybrid/
│   ├── srresnet_freq/
│   ├── vdslr_hybrid/
│   ├── edge_injection/             # PNG outputs (edge model)
│   ├── srgan/
│   ├── real_esrgan/
│   └── swinir/                     # SwinIR outputs (externally generated, no training code)
│
└── weights/                        # (Empty in repo — weights stored locally during training)
```

The `models/` directory doubles as training code — each file is a standalone script containing dataset, model definition, loss functions, optimizer setup, training loop, and validation. This monolithic structure is consistent with rapid prototyping and makes each experiment self-contained and reproducible.

---

## Implemented Architectures

### 1. SRCNN — Shallow CNN Baseline (L1 and L2 Variants)

**Files:** `srcnn.py` (MSE), `srcnn_try.py` (L1)

The classical SRCNN architecture (Dong et al., 2014): three convolutional layers with 64, 32, and 1 channels using 9×9, 5×5, and 5×5 kernels respectively. The input is bicubically upsampled to target resolution before being passed to the network — SRCNN operates in the high-resolution space rather than learning the upsampling itself.

This is standard SRCNN behavior with no architectural modification. The single meaningful difference between the two files is the loss function: `srcnn.py` uses `nn.MSELoss()` while `srcnn_try.py` uses `nn.L1Loss()`. Both train for 100 epochs, batch size 16, Adam at 1e-4.

**Thermal relevance:** SRCNN's pre-upsampling approach avoids any learned sub-pixel structure and is the weakest baseline. At 8× scale the bicubic upsampling step introduces significant blurring, and the shallow network has limited capacity to recover sharp edges. However, it establishes an important lower bound and tests loss sensitivity.

---

### 2. Hybrid VDSR — VDSR with PixelShuffle and Multi-Loss

**File:** `hybrid_vdsr.py`

This is not standard VDSR (Kim et al., 2016). The original VDSR learns residuals from bicubically-upsampled inputs and uses a very deep (20-layer) plain CNN. This implementation makes two structural changes:

- **PixelShuffle upsampling replaces the pre-upsampling strategy.** The network receives the native low-resolution input and learns to upscale internally via sub-pixel convolution (PixelShuffle with scale=8 in a single step, outputting 64 channels that are shuffled to 1-channel × 8× spatial resolution).
- **The loss is a composite:** L2 reconstruction + VGG perceptual (VGG19 features at layer 35) + a custom diffusive patch loss.

The **diffusive patch loss** is a mean-field blur regularizer: it computes the MSE between 3×3 average-pooled versions of SR and HR. This penalizes the model for producing outputs with different local mean statistics than the target, which is a soft constraint on isothermal region continuity. This is a custom, non-standard loss.

Training: 60 epochs, batch 4, Adam 1e-4, best model selected by SSIM.

**Thermal relevance:** The PixelShuffle hybrid avoids bicubic pre-processing artifacts. The diffusive patch loss is motivated by the smooth, low-frequency nature of thermal imagery — it prevents hallucination of fine texture in regions that should be thermally uniform.

---

### 3. SRResNet Hybrid — Residual Network with Perceptual and Diffusive Loss

**File:** `srresnet_hybrid.py`

A clean single-channel adaptation of SRResNet (Ledig et al., 2017): 16 residual blocks with 64 channels, 9×9 input/output convolutions, and three PixelShuffle upsampling stages (2×2×2=8×). Residual blocks follow the Conv-ReLU-Conv pattern without batch normalization (matching ESRGAN-era recommendations).

The loss is `L1 + 0.01 × VGG + 0.05 × DiffusiveLoss`.

The **DiffusiveLoss** here uses explicit first-order finite differences rather than average pooling: `kernel_x = [[-1, 1]]` and `kernel_y = [[-1], [1]]`. It penalizes mismatch in the x and y gradient fields of SR vs HR images. This is closer to a total variation constraint on the prediction error than to the patch-blur approach in hybrid VDSR.

Training: 80 epochs, batch 4, Adam 1e-4, best model selected by SSIM.

**Note on VGG:** Single-channel thermal images are `repeat`-broadcast to 3 channels before VGG feature extraction. This is a pragmatic adaptation — the VGG features are from a network trained on RGB, so the perceptual prior is at best a rough approximation to thermal structure. This is a known limitation that the codebase implicitly accepts.

---

### 4. SRResNet Frequency — Residual Network with FFT Loss

**File:** `srresnet_frequency.py`

Architecturally identical to `srresnet_hybrid.py`. The difference is the loss function: `L1 + 0.05 × EdgeAwareLoss + 0.05 × FFTLoss`.

**EdgeAwareLoss** is a weighted Sobel-based gradient loss. It computes Sobel x and y gradient maps for both SR and HR, then reweights the squared gradient error by `exp(-5 × grad_HR)`. This exponential weighting attenuates the loss in high-gradient (edge) regions of the HR image and emphasizes low-gradient (smooth) regions. The motivation is to discourage artifact production near sharp thermal boundaries while still supervising structure in flat regions.

**FFTLoss** computes the L1 distance between the magnitude spectra of SR and HR images via `torch.fft.fft2`. This supervises the frequency content of the reconstruction directly. Thermal images have a characteristic spectral profile — relatively more energy concentrated in low frequencies due to smooth spatial temperature distributions. The FFT loss provides a global signal about whether the network is reproducing the correct spatial frequency statistics.

Training: 80 epochs, batch 4, Adam 1e-4, best model by PSNR.

**Thermal motivation:** The FFT loss is well-motivated for thermal SR. RGB perceptual losses tend to drive models toward natural image frequency statistics, which for thermal data means excessive sharpening or hallucinated textures. A direct frequency loss provides a domain-agnostic spectral supervision signal.

---

### 5. SRResNet Edge Injection — Learnable Sobel Gating

**File:** `edge_injection.py`

This is the most architecturally novel model in the repository. The base remains SRResNet (16 blocks, 64 channels, PixelShuffle ×8), but a **learnable edge injection mechanism** is added to the residual path:

```python
self.edge_weight = nn.Parameter(torch.tensor(0.1))

# Inside forward():
edges = get_edges(x)  # Sobel magnitude of low-res input
edges = F.interpolate(edges, size=res.shape[-2:], ...)
res = res + self.edge_weight * edges
```

A single learned scalar `edge_weight` (initialized to 0.1) controls how strongly the Sobel-derived edge map of the LR input is added into the residual feature map before the global skip connection and upsampling. The edge map is bilinearly interpolated to match the feature resolution.

This mechanism gives the model a direct, explicit, low-level edge signal that bypasses the residual blocks. Rather than relying entirely on learned features to encode edge structure, the model can trivially boost edge responses by increasing `edge_weight`. The weight is logged each epoch, allowing observation of how the model learns to balance edge injection vs. learned features over training.

The loss function is: `L1 + 0.01 × VGG + 0.5 × EdgeLoss + 0.03 × DiffusiveLoss`, where EdgeLoss is simply `F.l1_loss(get_edges(sr), get_edges(hr))` — direct supervision on the output edge map. The high weight of 0.5 on edge loss is notable and reflects an explicit prioritization of edge accuracy.

Training: 80 epochs, batch 4, Adam 1e-4, best model by SSIM.

**Thermal motivation:** Thermal edge recovery is one of the core challenges at 8×. Warm/cool boundaries are the primary structural feature in thermal images. Providing an explicit, gradient-based spatial prior via a learnable gate lets the model trade between data-driven feature learning and direct geometric guidance, depending on what the training loss finds most useful.

---

### 6. SRGAN — Standard Adversarial Training

**File:** `srgan_offical.py`

A standard SRGAN implementation (Ledig et al., 2017) adapted for thermal data. The generator follows the SRResNet backbone with BatchNorm and PReLU activations. The discriminator is a fully-convolutional VGG-style binary classifier with adaptive average pooling and a dense head.

**Two-phase training:**
- Phase 1 (15 epochs): Pure MSE pretraining of G to avoid mode collapse at initialization.
- Phase 2 (50 epochs): Adversarial training with `loss_G = MSE + 1e-3 × adv + 2e-6 × VGG`.

The thermal imagery is loaded as RGB (3-channel) here, unlike all other models which operate in single-channel mode. This is consistent with the SRGAN paper but is technically suboptimal for grayscale thermal data — it triples the input/output channel count without adding information.

The loss weights are close to the original SRGAN paper values. No relativistic loss, no spectral normalization. This is the most standard implementation in the repository.

---

### 7. Real-ESRGAN Diffusive — RRDB with GAN and Gradient Loss

**File:** `realesrgan_diffuision.py`

An ESRGAN-style architecture (Wang et al., 2018) adapted to thermal SR. The generator is an 8-block RRDB (Residual-in-Residual Dense Block) network with 64 channels. Each RRDB contains three dense blocks, each with 5 densely-connected convolutions and residual scaling at 0.2. The 8× upsampling uses three sequential PixelShuffle stages.

Compared to standard Real-ESRGAN:
- Input and output are **single-channel** (grayscale thermal), not RGB.
- **8 RRDB blocks** (Real-ESRGAN typically uses 23).
- Discriminator uses **spectral normalization** and feature extraction for feature matching loss.
- **Relativistic average GAN** (RaGAN) losses for both G and D.
- **Feature matching loss** added alongside GAN loss.
- **DiffusiveLoss** (gradient-domain regularization) added with weight 0.05.

Generator loss: `L1 + 1.0 × VGG + 0.1 × FM + 0.005 × RaGAN + 0.05 × Diffusive`

The relatively low GAN weight (0.005) and higher FM weight (0.1) reflects a conservative adversarial strategy that prioritizes structural fidelity over perceptual sharpness — appropriate given the risk of GAN-hallucinated texture in thermal images.

Training: 80 epochs, AMP-enabled, CosineAnnealingLR, batch 4.

---

### 8. Thermal Real-ESRGAN — Full Stabilized GAN

**File:** `thermal_realesrgan.py`

The most complete and carefully engineered model in the repository. It extends `realesrgan_diffuision.py` with several additional stabilization mechanisms:

**GAN warmup scheduling:**
```python
def gan_weight_schedule(epoch):
    if epoch < GAN_WARMUP:  # 15 epochs
        return 0.0
    return GAN_WEIGHT * min((epoch - GAN_WARMUP) / GAN_RAMP, 1.0)  # 10-epoch ramp
```
The GAN loss is completely disabled for the first 15 epochs (pure reconstruction pretraining), then linearly ramped over 10 epochs to a maximum weight of 0.1. This is a deliberate stability strategy — thermal GAN training is prone to mode collapse if adversarial gradients are introduced before the generator has learned basic reconstruction.

**R1 gradient penalty:**
```python
def r1_penalty(D, real):
    real = real.detach().float().requires_grad_(True)
    pred = D(real)
    grad = torch.autograd.grad(outputs=pred.sum(), inputs=real, ...)[0]
    return grad.pow(2).flatten(1).sum(1).mean()
```
The R1 penalty (computed outside AMP to ensure float32 gradients) regularizes the discriminator to be locally Lipschitz on real data. This is a more principled stabilization than gradient clipping and prevents discriminator overpowering.

**D steps per G step:** 2 discriminator updates per generator update, improving the D/G balance.

**AMP-safe FFT loss:**
```python
class FrequencyLoss(nn.Module):
    def forward(self, sr, hr):
        sr_f = sr.float()   # safe upcast from float16
        hr_f = hr.float()
        sr_mag = torch.abs(torch.fft.rfft2(sr_f))
        hr_mag = torch.abs(torch.fft.rfft2(hr_f))
        return F.l1_loss(sr_mag, hr_mag)
```
This explicitly upcasts to float32 before FFT because cuFFT in half precision only supports power-of-2 spatial dimensions, whereas the thermal images are 448×640. The comment in the code explicitly identifies this constraint — this is careful, domain-specific engineering.

Generator loss: `L1 + 1.0 × VGG + 0.05 × Diffusive + 0.01 × Freq + 0.1 × FM + w_gan × RaGAN`

Training metrics include LPIPS in addition to PSNR and SSIM.

---

## Novel Contributions and Custom Ideas

### 1. Learnable Edge Weight Injection (edge_injection.py)

A single learned scalar gate injects bilinearly-upsampled Sobel edge maps from the LR input into the residual feature stream before upsampling. The scalar is initialized at 0.1 and learned end-to-end. The logged evolution of `edge_weight` during training provides interpretability into how strongly the model relies on LR geometric priors vs. deep features.

**Why it may help:** In thermal images, edge locations are often well-preserved at low resolution even though edge amplitudes are attenuated. Providing an explicit spatial prior of where edges are — directly in the feature space rather than indirectly through loss terms — gives the network a structural anchor that can improve edge localization without risk of the network ignoring it.

---

### 2. DiffusiveLoss (Multiple Models)

Two variants appear across models:

- **Patch-blur version** (`hybrid_vdsr.py`): MSE between 3×3 average-pooled SR and HR. A mean-field smoothness constraint.
- **Gradient-domain version** (`srresnet_hybrid.py`, `realesrgan_diffuision.py`, `thermal_realesrgan.py`): MSE between finite-difference gradient fields (`[-1, 1]` kernels in x and y). A first-order gradient matching constraint.

Neither is standard in SR literature. The gradient-domain version is conceptually closer to gradient domain image processing and to the spatial smoothness penalties used in stereo/optical flow estimation. For thermal images, it discourages the production of spurious high-frequency variations in the gradient field, which would appear as false edge artifacts.

---

### 3. FFT Frequency Loss

Introduced in `srresnet_frequency.py` and carried into `thermal_realesrgan.py`. The L1 distance between FFT magnitude spectra supervises the model to match the spatial frequency distribution of the HR image. The AMP-safe float32 upcast in the final model is a concrete engineering fix for a non-obvious CUDA limitation.

**Why it matters:** Thermal images have a characteristic low-frequency spectral bias. An L1 loss will converge to blurry outputs that satisfy pixel-level accuracy. A VGG perceptual loss applies RGB natural image priors. An FFT loss directly constrains spectral fidelity without domain-specific priors, providing a more neutral frequency supervision mechanism.

---

### 4. Edge-Aware Gradient Reweighting (srresnet_frequency.py)

The EdgeAwareLoss applies `exp(-5 × grad_HR)` as a spatial weight to the Sobel gradient error. This suppresses the loss contribution in high-gradient (edge) regions of the HR image and amplifies it in smooth regions. The rationale is that smooth (isothermal) regions need to remain smooth, while edges will be handled by other loss terms. It is an unusual inversion of the typical edge-weighting intuition (which usually upweights edges).

---

### 5. GAN Warmup and Ramp Schedule (thermal_realesrgan.py)

A structured warmup-and-ramp GAN weight schedule: 0 for epochs 0–14, then linearly increasing to 0.1 over epochs 15–24. This is more principled than the simple two-phase approach in SRGAN and prevents the adversarial loss from dominating reconstruction quality early in training. Combined with the R1 penalty and spectral normalization, this constitutes a complete GAN stabilization stack.

---

### 6. Feature Matching Loss (realesrgan_diffuision.py, thermal_realesrgan.py)

Feature matching (extracting intermediate discriminator activations for both real and fake, computing L1 distance) is borrowed from pix2pixHD-era work. It provides G with a richer, multi-scale training signal from D without the instability of pure adversarial gradients. The implementation correctly detaches real features to prevent gradient flow into D during G's update.

---

## Loss Functions and Training Strategy

### Reconstruction Losses

**MSE (L2):** Used in `srcnn.py` and SRGAN Phase 1. MSE produces PSNR-optimal outputs in expectation but at 8× it over-smooths by penalizing outliers quadratically. This is well-understood; the experiments confirm it as a baseline.

**L1:** Used in `srcnn_try.py` and as the primary reconstruction term in all hybrid models. L1 is a median estimator and produces slightly sharper outputs than MSE at the cost of a less smooth loss surface. Standard practice in modern SR.

### Perceptual Loss

**VGG19 (features[:35]):** Used in all hybrid and GAN models. Single-channel inputs are channel-replicated (`repeat(1, 3, 1, 1)`) before VGG feature extraction. The thermal_realesrgan variant additionally applies ImageNet mean/std normalization before the VGG pass, which is more correct.

The VGG prior is imperfect for thermal SR — it was trained on natural images with very different frequency and texture statistics. The codebase accepts this limitation pragmatically rather than training a thermal-specific perceptual model.

### Adversarial Losses

**Standard GAN (srgan_offical.py):** BCE on binary real/fake outputs. Simple and standard.

**Relativistic average GAN (realesrgan_diffuision.py, thermal_realesrgan.py):** The discriminator predicts the relative realism of real vs fake (and vice versa). RaGAN provides more informative training gradients than standard GAN by comparing distributions rather than classifying individual samples.

### Specialized Losses

| Loss | File | Weight | Purpose |
|------|------|--------|---------|
| DiffusiveLoss (gradient) | srresnet_hybrid, realesrgan* | 0.05 | Gradient field matching, smooth region continuity |
| DiffusiveLoss (patch blur) | hybrid_vdsr | 0.1 | Mean-field continuity |
| EdgeAwareLoss | srresnet_frequency | 0.05 | Weighted Sobel matching, emphasizes flat regions |
| EdgeLoss (direct) | edge_injection | 0.5 | Direct edge map supervision at output |
| FFT / FrequencyLoss | srresnet_frequency, thermal_realesrgan | 0.05, 0.01 | Spectral content matching |
| Feature Matching | realesrgan*, thermal_realesrgan | 0.1, 0.1 | Multi-scale D-feature consistency |
| R1 Penalty | thermal_realesrgan | 10.0 | D Lipschitz regularization |

### Optimization

All models use Adam optimizer. The two full GAN models use CosineAnnealingLR schedulers (T_max=80, eta_min=1e-6) applied to both G and D. Simpler models use a fixed learning rate.

The `thermal_realesrgan.py` uses AMP (`torch.cuda.amp.GradScaler`) with careful handling: R1 gradient penalty is computed outside the AMP context to preserve float32 gradients, and the FrequencyLoss upcasts to float32 internally.

---

## Thermal-Specific Challenges Addressed

**Smooth gradients and low texture:** The DiffusiveLoss variants directly penalize gradient field mismatch rather than pixel intensity mismatch, making them sensitive to the continuity of smooth temperature regions. The FFT loss ensures spectral reconstruction is thermally plausible.

**Edge recovery at 8×:** At this scale factor, edge localization errors compound significantly. The edge injection model directly injects Sobel priors from the LR input into the feature stream. The EdgeLoss (weight 0.5) in the edge injection model reflects that edge accuracy is the primary quality criterion.

**VGG perceptual mismatch:** All models that use VGG do so with single→3-channel replication. The `thermal_realesrgan.py` model additionally applies ImageNet normalization, which is a marginal improvement in perceptual alignment but still imperfect. This limitation is not explicitly addressed architecturally — a thermal-specific perceptual extractor would require a labeled thermal dataset.

**GAN hallucination:** The conservative GAN weights (0.005–0.1) across all GAN models reflect awareness that GANs will hallucinate texture in smooth regions if the adversarial signal is too strong. The warmup schedule in `thermal_realesrgan.py` and the feature matching loss provide a controlled introduction of adversarial supervision.

**Grayscale single-channel:** All non-SRGAN models operate in single-channel mode, correctly treating thermal images as grayscale. SRGAN processes as RGB (3-channel), which adds parameters without information gain.

**Non-power-of-2 spatial dimensions:** The thermal images appear to be 448×640 (based on SRCNN hardcoded resize and the FFT comment). This is a non-standard size that requires explicit handling for certain CUDA operations, which the FrequencyLoss correctly manages.

---

## Training

### Requirements

```
torch
torchvision
torchmetrics
pytorch_msssim
lpips
scikit-image
pillow
numpy
opencv-python
```

### Data Layout

```
thermal/
  train/
    LQ/    # Low-quality (low-resolution) images, single-channel .bmp
    HR/    # High-resolution ground truth, single-channel .bmp
  val/
    LQ/
    HR/
```

All models expect paired LQ/HR images with matching filenames. HR images should be exactly 8× the spatial dimensions of LQ images.

### Training Each Model

Each model file is a standalone training script. Edit the path constants at the top of the file before running.

```bash
# Baseline SRCNN (MSE)
python models/srcnn.py

# SRCNN with L1 loss
python models/srcnn_try.py

# Hybrid VDSR
python models/hybrid_vdsr.py

# SRResNet + VGG + Diffusive
python models/srresnet_hybrid.py

# SRResNet + FFT + Edge-Aware
python models/srresnet_frequency.py

# SRResNet + Edge Injection
python models/edge_injection.py

# SRGAN (standard)
python models/srgan_offical.py

# Real-ESRGAN + Diffusive
python models/realesrgan_diffuision.py

# Full Thermal Real-ESRGAN
python models/thermal_realesrgan.py
```

Checkpoints are saved to the working directory or to the configured `SAVE_DIR`. Best models are selected by SSIM (for hybrid models) or PSNR (for frequency and GAN models). The full thermal Real-ESRGAN also saves periodic checkpoints every 10 epochs with full optimizer state.

---

## Inference

Each model has a corresponding inference script in the `inference/` directory. Scripts load a saved checkpoint, run inference over a validation directory, compute per-image PSNR and SSIM, and save output images.

```bash
# SRGAN inference
python inference/SRGAN_INFERENCE.py

# Edge injection inference (includes boundary crop for fair metric evaluation)
python inference/inference_edge_injection.py

# SRResNet Frequency
python inference/inference_srresnet_frequency.py

# Hybrid VDSR
python inference/vdslr_hybrid_infernce.py

# SRCNN variants
python inference/srcnn_l1_inference.py
python inference/srcnn_l2_inference.py
```

**Note:** The edge injection inference script crops a border of `scale=8` pixels before metric computation. This is a standard practice to avoid boundary artifacts from convolution padding.

**Note:** The SRGAN inference script handles multiple checkpoint formats (bare state dict, `state_dict` key, `model_state_dict` key) for robustness.

---

## Metrics and Evaluation

### PSNR (Peak Signal-to-Noise Ratio)

Computed across all models. Implemented as:
```python
mse = F.mse_loss(sr, hr)
psnr = 10 * torch.log10(1.0 / mse)
```
After denormalization to [0, 1]. PSNR is the primary model selection criterion for most models.

### SSIM (Structural Similarity)

Two implementations are used:
- `torchmetrics.image.StructuralSimilarityIndexMeasure` (default in training loops)
- `pytorch_msssim.ssim` (in SRCNN scripts)
- `skimage.metrics.structural_similarity` (in frequency inference script)

SSIM is used as the model selection criterion in hybrid and edge injection models, reflecting the goal of structural fidelity over pixel-accuracy.

### LPIPS (Learned Perceptual Image Patch Similarity)

Used only in the two Real-ESRGAN variants during validation. `lpips.LPIPS(net='alex')` is applied to 3-channel-replicated outputs. LPIPS provides a perceptual quality metric that is more sensitive to structural distortions than PSNR/SSIM, and is important for evaluating GAN-based models that may achieve competitive PSNR while introducing perceptual artifacts.

### Inline SSIM (SRGAN inference)

The SRGAN inference script implements a simplified global SSIM without spatial windowing. This is a less accurate approximation and should not be compared directly against the torchmetrics/skimage implementations used elsewhere.

---

## Research Insights

The codebase reflects a clear experimental philosophy: establish a stack of increasingly capable architectures and, in parallel, explore orthogonal loss engineering strategies. The core tension being studied is between **reconstruction fidelity** (PSNR/SSIM, driven by L1 and MSE terms) and **structural plausibility** (edge sharpness, spectral accuracy, thermal continuity, driven by the custom loss terms).

The GAN branch of the experiments takes a deliberately conservative approach to adversarial training — consistently low adversarial weights, feature matching as a stabilizing surrogate signal, and increasingly sophisticated discriminator regularization. This conservatism is appropriate for thermal SR where GAN-hallucinated texture is a genuine failure mode rather than an acceptable quality trade-off.

The edge injection model represents the clearest attempt at an architectural (vs. loss-level) thermal-domain adaptation. The FFT loss represents the clearest attempt at domain-neutral spectral supervision. These two threads — geometric priors and spectral supervision — are the most conceptually interesting research directions in the repository.

---

## Future Directions

**Thermal perceptual model:** Train a feature extractor on a large thermal image corpus and use it in place of VGG for perceptual loss computation. This would eliminate the domain mismatch that currently limits all perceptual-loss models.

**Transformer-based backbone:** Replace the SRResNet residual blocks with Swin Transformer blocks (as in SwinIR). The SwinIR outputs in `output_samples/swinir/` suggest this comparison was run externally. Integrating SwinIR training would complete the benchmark.

**Learnable frequency weighting:** Rather than a fixed FFT magnitude L1 loss, learn a frequency-band weighting mask that emphasizes the spectral ranges most informative for thermal boundary reconstruction.

**Thermal-specific discriminator:** Train the GAN discriminator on thermal imagery only, so it learns priors specific to thermal image distributions rather than applying a natural-image-trained network implicitly through VGG features.

**Uncertainty-aware reconstruction:** Model the reconstruction as a distribution rather than a point estimate. Uncertainty maps would allow downstream systems to distinguish reliably recovered structure from uncertain or hallucinated regions.

**Multi-scale edge injection:** Extend the single-scale Sobel injection to a multi-scale Laplacian pyramid, providing edge signals at different spatial resolutions into different stages of the upsampling path.

**Diffusion-based SR:** Use a thermal-domain denoising diffusion model for iterative refinement, which may better model the smooth, low-frequency structure of thermal images than single-step GAN generation.
