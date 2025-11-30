# FACT-CLIP: Zero-Shot Action Segmentation

This repository contains an implementation for zero-shot temporal action segmentation, combining the FACT (Frame-Action Cross-attention Temporal modeling) architecture with CLIP text embeddings for open-vocabulary recognition.

## Overview

The project extends the FACT model (CVPR 2024) to enable zero-shot action segmentation by:

1. **FACT Architecture**: Efficient temporal modeling using frame-action cross-attention
2. **CLIP Integration**: Aligning visual frame features with pre-computed CLIP text embeddings
3. **Holdout Training**: Training on seen classes while evaluating on unseen (holdout) classes

## Repository Structure

```
.
├── README.md                   # This file
├── setup.py                    # Package installation
├── requirements.txt            # Dependencies
├── LICENSE                     # License file
├── .gitignore                  # Git ignore patterns
│
├── scripts/                    # Entry point scripts
│   ├── train.py                # Training script
│   ├── eval.py                 # Evaluation script
│   ├── eval_holdout.py         # Holdout evaluation analysis
│   ├── run_eval.py             # Checkpoint evaluation
│   ├── select_holdout_classes.py  # Holdout class selection
│   └── fact_input_emb_logit_viz.py  # Visualization
│
├── fact_clip/                  # Main package
│   ├── __init__.py
│   ├── home.py                 # Path utilities
│   ├── models/                 # Model definitions
│   │   ├── blocks.py           # FACT and FACT_CLIP
│   │   ├── loss.py             # Loss functions
│   │   └── basic.py            # Building blocks
│   ├── utils/                  # Utilities
│   │   ├── dataset.py          # Dataset loading
│   │   ├── evaluate.py         # Metrics
│   │   ├── text_embeddings.py  # CLIP embeddings
│   │   └── havid_text_prompts.py  # HAViD prompts
│   └── configs/                # Configuration files
│       ├── default.py          # Default config
│       └── *.yaml              # Dataset configs
│
├── data/                       # Datasets (gitignored)
├── logs/                       # Training logs (gitignored)
├── wandb/                      # W&B logs (gitignored)
└── assets/                     # Images and figures
```

## Installation

### Prerequisites

- Python 3.8 or higher (3.10 recommended)
- CUDA-compatible GPU (check your CUDA version with `nvidia-smi`)

### Option 1: Conda Environment (Recommended)

```bash
# Create a new conda environment
conda create -n fact_clip python=3.10 -y

# Activate the environment
conda activate fact_clip

# Install PyTorch with CUDA support (adjust cuda version as needed)
# For CUDA 11.8:
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CUDA 12.1:
# conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Option 2: Python Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install PyTorch with CUDA support (adjust cuda version as needed)
# Visit https://pytorch.org/get-started/locally/ for the correct command
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Install the Package

Once your environment is set up and activated:

```bash
# Clone the repository (if not already done)
git clone https://github.com/lucas-t-t/FACT-CLIP.git
cd FACT-CLIP

# Install in development/editable mode
pip install -e .

# Or install with visualization dependencies (matplotlib, umap-learn, scikit-learn)
pip install -e ".[viz]"
```

### Verify Installation

```bash
# Check that the package is installed
python -c "import fact_clip; print('Installation successful')"
```

## Quick Start

```bash
# Train vanilla FACT on HAViD
python scripts/train.py --cfg fact_clip/configs/havid_view0_lh_pt.yaml --set aux.gpu 0

# Train FACT_CLIP with holdout classes for zero-shot evaluation
python scripts/train.py --cfg fact_clip/configs/havid_view0_lh_pt_holdout.yaml \
    --set aux.gpu 0 use_clip true

# Evaluate a checkpoint
python scripts/run_eval.py --cfg fact_clip/configs/havid_view0_lh_pt.yaml \
    --ckpt logs/havid_view0_lh_pt/split1/.../ckpts/best.net
```

## Key Features

- **Multiple Datasets**: Support for Breakfast, GTEA, HAViD, EgoProceL, Epic-Kitchens
- **FACT_CLIP**: Zero-shot capable variant using CLIP text embeddings
- **Holdout Training**: Framework for evaluating generalization to unseen action classes
- **HAViD Text Prompts**: Automatic conversion of HAViD action codes to natural language

## Documentation

See the README files in each subdirectory for detailed documentation:

- [`fact_clip/models/README.md`](fact_clip/models/README.md) - Model architecture
- [`fact_clip/utils/README.md`](fact_clip/utils/README.md) - Utilities and dataset handling
- [`fact_clip/configs/README.md`](fact_clip/configs/README.md) - Configuration system

## Citation for VanillaFACT model

```bibtex
@inproceedings{lu2024fact,
    title={{FACT}: Frame-Action Cross-Attention Temporal Modeling for Efficient Supervised Action Segmentation},
    author={Zijia Lu and Ehsan Elhamifar},
    booktitle={Conference on Computer Vision and Pattern Recognition 2024},
    year={2024},
}
```


