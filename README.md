# Restormer Fine-Tuning for NTIRE 2026 Image Denoising (σ = 50)

This repository contains the **training and evaluation pipeline** used for our submission to the **NTIRE 2026 Image Denoising Challenge (Gaussian Noise σ = 50)**.

The approach is based on **Restormer (Efficient Transformer for High-Resolution Image Restoration)**, a transformer-based architecture designed for high-quality image restoration tasks.

The model is **fine-tuned on the DIV2K dataset** and evaluated using **tiled inference** to efficiently process high-resolution images.

---

# 1. Repository Structure

```
NTIRE-2026 (CV_SVNIT)
│
├── training.py
├── evaluation.py
├── README.md
├── requirements.txt
│
├── Restormer/              # official Restormer repository
│
├── test_images/            # noisy input images
│
└── results/                # denoised output images
```

---

# 2. Prerequisites

Recommended environment:

Python ≥ 3.8
PyTorch ≥ 2.0
CUDA-enabled GPU (recommended)

---

# 3. Install Dependencies

Install required libraries using:

```
pip install -r requirements.txt
```

or manually:

```
pip install torch torchvision
pip install einops timm lmdb pillow numpy
```

---

# 4. Clone Repository

```
git clone https://github.com/Parampandya0332/CV_SVNIT.git
cd CV_SVNIT
```

Clone the official Restormer implementation:

```
git clone https://github.com/swz30/Restormer.git
```

---

# 5. Dataset Preparation (Training)

Training uses the **DIV2K dataset**.

Download from:

https://data.vision.ee.ethz.ch/cvl/DIV2K/

Expected dataset structure:

```
datasets/

 ├── DIV2K_train_HR
 │     ├── 0001.png
 │     ├── 0002.png
 │     └── ...
 │
 └── DIV2K_valid_HR
       ├── 0801.png
       ├── 0802.png
       └── ...
```

---

# 6. Training Configuration

Noise level: σ = 50

Patch size: 256 × 256

Training iterations: 20000

Batch size: 1

Optimizer: AdamW

Learning rate: 1e-4

Weight decay: 1e-4

Learning rate scheduler: Cosine Annealing

Loss function: Charbonnier Loss

Mixed precision training: Enabled (AMP)

---

# 7. Training

Run training using:

```
python training.py
```

Training pipeline:

1. Load DIV2K training images
2. Randomly crop 256×256 patches
3. Add Gaussian noise with σ = 50
4. Train Restormer model
5. Validate every 500 iterations
6. Save best model based on PSNR

---

# 8. Evaluation / Inference

Place noisy images inside:

```
test_images/
```

Run inference using the CLI-based evaluation script:

```
python evaluation.py --input test_images --output results --weights best_model.pth
```

Arguments:

| Argument       | Description                          |
| -------------- | ------------------------------------ |
| --input        | Folder containing noisy input images |
| --output       | Folder to save denoised outputs      |
| --weights      | Path to trained model checkpoint     |
| --tile_size    | Tile size used during inference      |
| --tile_overlap | Overlap between tiles                |

Example:

```
python evaluation.py \
--input test_images \
--output results \
--weights best_model.pth
```

---

# 9. Model Checkpoint Loading

The evaluation script supports two checkpoint formats.

### Pure model weights

```
model.load_state_dict(weights)
```

### Training checkpoints

```
{
 "model": state_dict,
 "optimizer": ...,
 "scheduler": ...
}
```

The script automatically detects the checkpoint type and loads the correct model weights.

---

# 10. Evaluation Strategy

High-resolution images are processed using **tiled inference**.

Tile size: 512

Tile overlap: 32

Steps:

1. Divide image into overlapping tiles
2. Pad tiles so dimensions are divisible by 8
3. Run Restormer inference on each tile
4. Merge overlapping outputs

This approach allows processing of large images without GPU memory overflow.

---

# 11. Runtime Measurement

Runtime is measured automatically during evaluation.

The script outputs:

```
Runtime per image: X seconds
```

This metric is required for NTIRE evaluation.

---

# 12. Testing the Repository

To verify the repository works correctly:

```
git clone https://github.com/Parampandya0332/CV_SVNIT.git
cd CV_SVNIT

pip install -r requirements.txt

python evaluation.py --input test_images --output results --weights best_model.pth
```

If the script runs successfully and produces images in the `results/` folder, the repository is correctly configured.

---

# 13. Acknowledgement

This work is based on the official Restormer implementation.

Restormer: Efficient Transformer for High-Resolution Image Restoration

Official repository:

https://github.com/swz30/Restormer
