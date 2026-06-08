import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import copy


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class Data_DINO(Dataset):
    def __init__(self, hr_dir, teacher_transform=None, student_transform=None):
        self.hr_dir = hr_dir
        self.teacher_transforms = teacher_transform
        self.student_transforms = student_transform          # fixed typo
        self.image_names = sorted(os.listdir(hr_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        hr = Image.open(os.path.join(self.hr_dir, name)).convert("L")

        hr_teacher = []
        hr_student = []

        if self.teacher_transforms is not None:
            hr_teacher.append(self.teacher_transforms(hr))
            hr_teacher.append(self.teacher_transforms(hr))

        if self.student_transforms is not None:
            # 2 global crops for student (same size as teacher)
            hr_student.append(self.teacher_transforms(hr))   # fixed: was using teacher_transforms
            hr_student.append(self.teacher_transforms(hr))
            # 6 local crops
            for _ in range(6):
                hr_student.append(self.student_transforms(hr))

        return hr_teacher, hr_student


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────
teacher_transforms = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.4, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

student_transforms = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.05, 0.4)),
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])



dataset = Data_DINO(
    hr_dir="/home/projectwork/Deep_learning/VANSH_WORK/models_thermal/dino_data_hr_only",
    teacher_transform=teacher_transforms,
    student_transform=student_transforms
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)

class MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)



class DINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet50(weights=None)
       
        self.encoder.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.encoder.fc = nn.Identity()
        self.head = MLP()

    def forward(self, x):
        features = self.encoder(x)
        projection = self.head(features)
        return F.normalize(projection, dim=-1, p=2)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

student_model = DINO().to(device)
teacher_model = copy.deepcopy(student_model).to(device)

# Freeze teacher — it is updated only via EMA, never via gradients
for p in teacher_model.parameters():
    p.requires_grad = False

student_optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)


epochs          = 80
momentum        = 0.996          
teacher_temp    = 0.04           
student_temp    = 0.1            
center_momentum = 0.9

center = torch.zeros(1, 256, device=device)

student_model.train()
teacher_model.eval()

for epoch in range(epochs):
    total_loss = 0.0

    for teacher_images, student_images in dataloader:
        student_optimizer.zero_grad()

        teacher_outputs = []
        with torch.no_grad():
            for image in teacher_images:
                image = image.to(device)
                output = teacher_model(image)
                teacher_outputs.append(output)

        student_outputs = []
        for image in student_images:
            image = image.to(device)
            output = student_model(image)
            student_outputs.append(output)

        loss  = torch.tensor(0., device=device)
        count = 0

        for i, t_out in enumerate(teacher_outputs):
            teacher_prob = torch.softmax(
                (t_out.detach() - center) / teacher_temp, dim=-1
            )
            for j, s_out in enumerate(student_outputs):
                if i == j:
                    continue                              # skip same-view pairs
                student_log_prob = torch.log_softmax(
                    s_out / student_temp, dim=-1
                )
                pair_loss = -(teacher_prob * student_log_prob).sum(dim=-1).mean()
                loss  += pair_loss
                count += 1

        loss = loss / count

        loss.backward()

        # Gradient clipping — standard DINO practice
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=3.0)

        student_optimizer.step()

        # ── EMA Teacher Update ─────────────────────────────
        with torch.no_grad():
            for t_param, s_param in zip(teacher_model.parameters(),
                                         student_model.parameters()):
                t_param.data = momentum * t_param.data + (1 - momentum) * s_param.data

        # ── Center Update (detached) ───────────────────────
        with torch.no_grad():
            batch_center = torch.cat(
                [o.detach() for o in teacher_outputs], dim=0   # detach explicitly
            ).mean(dim=0, keepdim=True)

            center = center_momentum * center + (1 - center_momentum) * batch_center

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}]  Loss: {total_loss:.4f}")

torch.save(student_model.state_dict(), "thermal_dino_student.pth")
print("Student model saved.")