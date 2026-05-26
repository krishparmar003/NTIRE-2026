import os
import glob
import time
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import sys

sys.path.append("./Restormer")

from basicsr.models.archs.restormer_arch import Restormer


# -----------------------------
# Argument Parser (CLI)
# -----------------------------

parser = argparse.ArgumentParser(description="Restormer Image Denoising Inference")

parser.add_argument(
    "--input",
    type=str,
    default="test_images",
    help="Folder containing noisy input images"
)

parser.add_argument(
    "--output",
    type=str,
    default="results",
    help="Folder to save denoised outputs"
)

parser.add_argument(
    "--weights",
    type=str,
    default="best_model.pth",
    help="Path to trained model weights"
)

parser.add_argument(
    "--tile_size",
    type=int,
    default=512,
    help="Tile size for inference"
)

parser.add_argument(
    "--tile_overlap",
    type=int,
    default=32,
    help="Tile overlap to avoid boundary artifacts"
)

args = parser.parse_args()


# -----------------------------
# Device
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Model
# -----------------------------

model = Restormer(
    inp_channels=3,
    out_channels=3,
    dim=48,
    num_blocks=[4, 6, 6, 8],
    num_refinement_blocks=4,
    heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type='BiasFree',
    dual_pixel_task=False
)


# -----------------------------
# Load Checkpoint
# -----------------------------

checkpoint = torch.load(args.weights, map_location=device)

if isinstance(checkpoint, dict) and "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# -----------------------------
# Paths
# -----------------------------

input_dir = args.input
output_dir = args.output

os.makedirs(output_dir, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))

tile_size = args.tile_size
tile_overlap = args.tile_overlap


# -----------------------------
# Inference
# -----------------------------

total_time = 0.0

for path in image_paths:

    img_name = os.path.basename(path)

    img = Image.open(path).convert("RGB")

    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    _, _, H, W = img_tensor.shape

    output = torch.zeros_like(img_tensor)
    weight = torch.zeros_like(img_tensor)

    start = time.time()

    for y in range(0, H, tile_size - tile_overlap):
        for x in range(0, W, tile_size - tile_overlap):

            y0 = y
            x0 = x

            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)

            tile = img_tensor[:, :, y0:y1, x0:x1]

            _, _, h, w = tile.shape

            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8

            tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="reflect")

            with torch.no_grad():
                out_tile = model(tile)

            out_tile = out_tile[:, :, :h, :w]

            output[:, :, y0:y1, x0:x1] += out_tile
            weight[:, :, y0:y1, x0:x1] += 1

    output /= weight

    end = time.time()
    total_time += (end - start)

    output = torch.clamp(output, 0, 1)

    output_img = TF.to_pil_image(output.squeeze(0).cpu())

    output_img.save(os.path.join(output_dir, img_name), compress_level=0)

    print(f"Processed: {img_name}")


# -----------------------------
# Runtime
# -----------------------------

runtime_per_image = total_time / len(image_paths)

print("\nInference completed successfully.")
print("Runtime per image:", runtime_per_image)