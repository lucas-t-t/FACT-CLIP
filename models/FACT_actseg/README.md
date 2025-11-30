# FACT: Frame-Action Cross-Attention Temporal Modeling

Implementation of FACT for efficient temporal action segmentation, extended with CLIP integration for zero-shot recognition.

## Overview

FACT performs temporal modeling on frame and action levels in parallel, using bidirectional cross-attention to iteratively refine features. This implementation adds a CLIP-based extension (`FACT_CLIP`) that enables zero-shot action segmentation by aligning frame features with text embeddings.

![FACT Architecture](src/overview.png)

## Installation

```bash
# Clone and setup
cd models/FACT_actseg

# Install dependencies
pip install -r src/requirements.txt

# For CLIP functionality, also install:
pip install transformers>=4.30.0
```

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA (recommended)
- wandb (for logging)

## Dataset Setup

### Supported Datasets

| Dataset | Config | Features |
|---------|--------|----------|
| Breakfast | `breakfast.yaml` | I3D (2048-dim) |
| GTEA | `gtea.yaml` | I3D (2048-dim) |
| HAViD | `havid_*.yaml` | I3D (2048-dim) |
| EgoProceL | `egoprocel.yaml` | I3D (2048-dim) |
| Epic-Kitchens | `epic-kitchens.yaml` | TSN (1024-dim) |

### Directory Structure

```
data/
├── breakfast/
│   ├── features/          # .npy feature files
│   ├── groundTruth/       # Frame-level annotations
│   ├── mapping.txt        # Action class mapping
│   └── splits/            # Train/test splits
├── HAViD/
│   └── ActionSegmentation/
│       └── data/
│           ├── features/
│           └── view0_lh_pt/  # View-specific data
└── ...
```

Download datasets:
- Breakfast/GTEA: [Zenodo](https://zenodo.org/records/3625992)
- HAViD: Contact dataset authors

## Training

### Vanilla FACT

```bash
# Breakfast dataset
python -m src.train --cfg src/configs/breakfast.yaml --set aux.gpu 0 split split1

# HAViD dataset
python -m src.train --cfg src/configs/havid_view0_lh_pt.yaml --set aux.gpu 0
```

### FACT_CLIP (Zero-Shot)

```bash
# Train with CLIP integration and holdout classes
python -m src.train --cfg src/configs/havid_view0_lh_pt_holdout.yaml \
    --set aux.gpu 0 use_clip true
```

### Key Training Options

```bash
--cfg CONFIG_FILE         # YAML configuration file
--set key value           # Override config values
    aux.gpu GPU_ID        # GPU to use
    aux.debug true        # Debug mode (deterministic, no wandb)
    use_clip true         # Enable FACT_CLIP
    holdout_mode true     # Enable holdout training
```

## Evaluation

```bash
# Standard evaluation
python -m src.eval --cfg CONFIG_FILE --set aux.gpu 0

# Holdout evaluation (seen/unseen class breakdown)
python -m src.eval_holdout --cfg CONFIG_FILE --set aux.gpu 0

# Evaluate specific checkpoint
python run_eval.py --checkpoint PATH_TO_CHECKPOINT
```

## FACT_CLIP: Zero-Shot Extension

FACT_CLIP extends vanilla FACT by adding a CLIP text embedding branch:

1. **Frame Feature Projection**: Projects frame features to CLIP embedding space (512-dim)
2. **Text Embeddings**: Pre-computed CLIP embeddings for action class descriptions
3. **Contrastive Loss**: InfoNCE loss aligns frame features with text embeddings

### Architecture

```
Frame Features (hid_dim) → FeatureProjection → CLIP Space (512)
                                                    ↓
                                              Cosine Similarity
                                                    ↓
Text Descriptions → CLIP Text Encoder → Text Embeddings (512)
```

### Enabling CLIP

In your config or via command line:

```yaml
use_clip: true
CLIP:
  model_name: "openai/clip-vit-base-patch32"
  temp: 0.1
  contrastive_weight: 0.5
  fact_loss_weight: 0.5
  projection_hidden_dim: 512
```

## Holdout Training

For zero-shot evaluation, hold out classes during training:

```yaml
holdout_mode: true
holdout_classes: [51, 53, 61, 67, 56]  # Class indices to hold out
```

Use `select_holdout_classes.py` to analyze and select holdout classes:

```bash
python select_holdout_classes.py --dataset havid_view0_lh_pt
```

## Configuration

All configs use YACS and inherit from `src/configs/default.py`. Key sections:

| Section | Description |
|---------|-------------|
| `FACT` | Model architecture (block types, token count) |
| `Bi`, `Bu`, `BU` | Block-specific parameters |
| `Loss` | Loss weights and matching strategy |
| `CLIP` | CLIP integration settings |
| `aux` | Training settings (GPU, logging) |

See [`src/configs/README.md`](src/configs/README.md) for detailed configuration options.

## Project Structure

```
src/
├── train.py              # Training script
├── eval.py               # Evaluation script
├── eval_holdout.py       # Holdout evaluation
├── home.py               # Path utilities
├── models/               # Model definitions
│   ├── blocks.py         # FACT and FACT_CLIP
│   ├── loss.py           # Loss functions
│   └── basic.py          # Basic building blocks
├── utils/                # Utilities
│   ├── dataset.py        # Dataset loading
│   ├── evaluate.py       # Metrics computation
│   ├── text_embeddings.py    # CLIP text embedding generation
│   └── havid_text_prompts.py # HAViD label conversion
└── configs/              # Configuration files
```

## Logging

Training logs are saved to `log/<experiment>/<run>/`:
- `args.json`: Configuration snapshot
- `ckpts/`: Model checkpoints
- `saves/`: Evaluation metrics
- `wandb/`: W&B logs

## Citation

```bibtex
@inproceedings{lu2024fact,
    title={{FACT}: Frame-Action Cross-Attention Temporal Modeling for Efficient Supervised Action Segmentation},
    author={Zijia Lu and Ehsan Elhamifar},
    booktitle={Conference on Computer Vision and Pattern Recognition 2024},
    year={2024},
}
```

## License

See [LICENSE](src/LICENSE) for details.

