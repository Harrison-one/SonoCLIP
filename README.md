# SonoCLIP: Mask-Guided Region-Aware Vision-Language Pretraining for Fetal Ultrasound Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)]()

Official PyTorch implementation of **SonoCLIP**, a region-controllable vision-language foundation model for fetal ultrasound analysis.

Paper: coming soon  
Code: https://github.com/Harrison-one/SonoCLIP

Introduction | Method | Quick Start | Evaluation | Main Results | Repository Structure | Citation

![SonoCLIP overview](assets/readme/Methods.png)

## Introduction

Fetal ultrasound is widely used in prenatal screening, but automatic analysis remains difficult because of speckle noise, acquisition variability, view-dependent artifacts, and subtle anatomical boundaries. Generic CLIP-style vision-language models usually rely on global image-text alignment, which can miss clinically important local structures.

SonoCLIP addresses this by introducing **mask-guided region-aware contrastive pretraining** for fetal ultrasound. The model appends anatomical masks as visual prompts through a mask-channel pathway, allowing the same backbone to support both global image understanding and region-focused inference.

The pretraining data described in the paper contains **1.44M fetal ultrasound image-text pairs** spanning **24 standard fetal planes**, with global plane descriptions and mask-based regional captions.

## Method

SonoCLIP builds on CLIP ViT-L/14@336px and introduces two main components:

- **Mask-channel visual pathway**: ultrasound images and binary anatomical masks are encoded through parallel stems and fused before the transformer blocks, enabling region-controllable visual representation learning.
- **Sigmoid pairwise contrastive loss**: global and region-level image-text pairs are optimized with independent pairwise matching objectives, improving stability for large-scale mixed supervision.

At inference time, SonoCLIP supports:

- **Global inference** with an all-one mask.
- **Mask-guided inference** with a provided or generated anatomical mask.

## Datasets

![FetalP24](assets/readme/FetalP24.png)

### FetalP24

FetalP24 is the large-scale pretraining dataset used in the paper. It covers 24 fetal ultrasound planes and includes image-level plane labels, mask annotations, gestational age metadata, and normalized text descriptions.

### FetalP6 and FetalP5

FetalP6 is used for downstream fetal plane classification across six categories. FetalP5 is used for downstream segmentation across five categories.

![FetalP6 classification](assets/readme/FetalP6_cls.png)

![FetalP5 segmentation](assets/readme/FetalP5_seg.png)

Raw private clinical images are not redistributed in this repository. Public users should prepare datasets according to the expected directory layout in the scripts.

## Quick Start

### Setup

```bash
git clone https://github.com/Harrison-one/SonoCLIP.git
cd SonoCLIP

conda create -n sonoclip python=3.10
conda activate sonoclip

pip install -e . --no-build-isolation
```

The current implementation was tested with:

```text
Python 3.10
torch 1.13.1+cu117
torchvision 0.14.1+cu117
```

If you install PyTorch manually, choose the CUDA build that matches your machine first, then run `pip install -e .`.

### Checkpoints

Place checkpoints under:

```text
checkpoints/
```

Expected examples:

```text
checkpoints/ViT-L-14-336px.pt
checkpoints/sonoclip_vision.pth
checkpoints/sonoclip_cls.pth
checkpoints/sonoclip_seg.pth
```

Checkpoint files are ignored by git.

## Training

SonoCLIP pretraining:

```bash
bash train/train_sonoclip.sh
```

Downstream classification:

```bash
bash train/train_sonoclip_cls.sh
```

Downstream segmentation:

```bash
bash train/train_sonoclip_seg.sh
```

The shell launchers define dataset roots, checkpoint paths, GPU settings, and output folders near the top of each file.

## Evaluation

Zero-shot fetal ultrasound classification from anonymous per-image h5 features:

```bash
python test/test_sonoclip_ul.py \
  --features-dir /path/to/ul_features_per_image \
  --base-model checkpoints/ViT-L-14-336px.pt \
  --vision-ckpt checkpoints/sonoclip_vision.pth \
  --class-names-txt /path/to/ul_plane_ids.txt \
  --output-dir test_outputs/ul_fea_per_image
```

FetalP6 classification:

```bash
python test/test_sonoclip_cls.py \
  --base-model checkpoints/ViT-L-14-336px.pt \
  --sono-vision-ckpt checkpoints/sonoclip_vision.pth \
  --checkpoint checkpoints/sonoclip_cls.pth \
  --data-root /path/to/FetalP6 \
  --test-txt /path/to/FetalP6_split/test.txt \
  --output-dir test_outputs/cls
```

FetalP5 segmentation:

```bash
python test/test_sonoclip_seg.py \
  --base-model checkpoints/ViT-L-14-336px.pt \
  --visual-ckpt checkpoints/sonoclip_vision.pth \
  --decoder-ckpt checkpoints/sonoclip_seg.pth \
  --data-root /path/to/FetalP5 \
  --test-txt /path/to/FetalP5_split/test.txt \
  --output-root test_outputs/seg
```

## Main Results

### Zero-shot classification on FetalPT24

| Method | Top-1 | Top-5 |
| --- | ---: | ---: |
| CLIP | 10.52 | 29.59 |
| UniMed-CLIP | 16.69 | 50.34 |
| FetalCLIP | 39.78 | 83.25 |
| SonoCLIP (w/o mask) | 58.38 | 94.47 |
| SonoCLIP (w/ mask) | **85.01** | **99.01** |

### Linear-probe classification on FetalP6

| Model | Avg Acc | Avg F1 |
| --- | ---: | ---: |
| CLIP | 87.1 | 85.8 |
| UniMed-CLIP | 83.5 | 82.7 |
| FetalCLIP | 94.6 | 92.3 |
| SonoCLIP (w/o mask) | 96.3 | 95.3 |
| SonoCLIP (w/ mask) | **99.3** | **98.8** |

### Segmentation on FetalP5

| Model | Avg Dice | Avg IoU |
| --- | ---: | ---: |
| CLIP | 85.1 | 77.5 |
| UniMed-CLIP | 86.5 | 79.8 |
| FetalCLIP | 69.8 | 60.2 |
| SonoCLIP (w/o mask) | **87.2** | **80.5** |

## Repository Structure

```text
SonoCLIP/
|-- sonoclip/                 # SonoCLIP model and tokenizer code
|-- train/
|   |-- dataset/              # fetal ultrasound dataset loaders
|   |-- train/                # training entrypoints
|   |-- train_sonoclip.sh     # pretraining launcher
|   |-- train_sonoclip_cls.sh # classification launcher
|   `-- train_sonoclip_seg.sh # segmentation launcher
|-- test/                     # evaluation scripts
|-- CLIP/                     # original CLIP reference implementation
|-- assets/readme/            # README figures
|-- requirements.txt
|-- setup.py
|-- pyproject.toml
`-- README.md
```

Local-only folders such as `checkpoints/`, `test_outputs/`, `train/log/`, raw ultrasound data, caches, and internal notes are intentionally excluded from git.

## Notes on Privacy

This repository is intended to release code and lightweight README assets. Private raw ultrasound images, local test outputs, PDF drafts, and checkpoints are not committed. Feature files or checkpoints may still contain information derived from private data and should be reviewed before release.

## Citation

If you find this project useful, please cite the paper once the final citation is available:

```bibtex
@inproceedings{sonoclip2026,
  title={SonoCLIP: Mask-Guided Region-Aware Vision-Language Pretraining for Fetal Ultrasound Analysis},
  author={Anonymous},
  booktitle={To appear},
  year={2026}
}
```

## License

This repository is released under the MIT License.
