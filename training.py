import os
import sys
import glob
import random
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

restormer_path = os.path.join(os.path.dirname(__file__), 'Restormer')
if restormer_path not in sys.path:
    sys.path.insert(0, restormer_path)
os.environ['PYTHONPATH'] = restormer_path
from basicsr.models.archs.restormer_arch import Restormer


device = "cuda"
SIGMA = 50
PATCH_SIZE = 256
TOTAL_ITERS = 20000


# -----------------------------
# Dataset
# -----------------------------

class TrainDataset(Dataset):
    def __init__(self, root, patch_size=256, sigma=50):
        self.paths = sorted(glob.glob(os.path.join(root, "*.png")))
        self.patch_size = patch_size
        self.sigma = sigma

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        img = Image.open(self.paths[idx]).convert("RGB")

        w, h = img.size
        ps = self.patch_size

        i = random.randint(0, h - ps)
        j = random.randint(0, w - ps)

        img = TF.crop(img, i, j, ps, ps)

        clean = TF.to_tensor(img)

        noise = torch.randn_like(clean) * (self.sigma / 255.0)

        noisy = clean + noise

        return noisy, clean


class ValDataset(Dataset):

    def __init__(self, root, num_patches=100, patch_size=256, sigma=50):

        paths = sorted(glob.glob(os.path.join(root, "*.png")))
        self.pairs = []

        rng = random.Random(42)

        for k in range(num_patches):

            path = rng.choice(paths)
            img = Image.open(path).convert("RGB")

            w, h = img.size

            i = rng.randint(0, h - patch_size)
            j = rng.randint(0, w - patch_size)

            img = TF.crop(img, i, j, patch_size, patch_size)

            clean = TF.to_tensor(img)

            torch.manual_seed(k)

            noise = torch.randn_like(clean) * (sigma / 255.0)

            noisy = torch.clamp(clean + noise, 0, 1)

            self.pairs.append((noisy, clean))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        return self.pairs[idx]


# -----------------------------
# Loss
# -----------------------------

class CharbonnierLoss(nn.Module):

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):

        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


# -----------------------------
# PSNR
# -----------------------------

def compute_psnr(pred, target):

    pred = torch.clamp(pred, 0, 1)

    mse = torch.mean((pred - target) ** 2)

    if mse == 0:
        return 100

    return 20 * math.log10(1.0 / math.sqrt(mse.item()))


# -----------------------------
# Training
# -----------------------------

def train(train_dir, val_dir):

    train_ds = TrainDataset(train_dir)
    val_ds = ValDataset(val_dir)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree'
    )

    model = model.to(device)

    criterion = CharbonnierLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TOTAL_ITERS,
        eta_min=1e-6
    )

    scaler = torch.amp.GradScaler("cuda")

    iteration = 0
    best_psnr = 0

    train_iter = iter(train_loader)

    while iteration < TOTAL_ITERS:

        try:
            noisy, clean = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            noisy, clean = next(train_iter)

        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):

            pred = model(noisy)

            loss = criterion(pred, clean)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        scheduler.step()

        iteration += 1

        if iteration % 100 == 0:
            print(f"Iter {iteration} Loss {loss.item():.6f}")

        if iteration % 500 == 0:

            model.eval()

            psnr_total = 0

            with torch.no_grad():

                for vn, vc in val_loader:

                    vn = vn.to(device)
                    vc = vc.to(device)

                    vp = model(vn)

                    psnr_total += compute_psnr(vp, vc)

            avg_psnr = psnr_total / len(val_loader)

            print("Validation PSNR:", avg_psnr)

            if avg_psnr > best_psnr:

                best_psnr = avg_psnr

                torch.save(model.state_dict(), "best_model.pth")

            model.train()

    torch.save(model.state_dict(), "final_model.pth")


if __name__ == "__main__":

    train_dir = "./datasets/DIV2K_train_HR"
    val_dir = "./datasets/DIV2K_valid_HR"

    train(train_dir, val_dir)