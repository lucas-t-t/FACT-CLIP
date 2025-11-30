# Configuration

This directory contains YACS configuration files for training FACT models.

## Files

| File | Description |
|------|-------------|
| `default.py` | Default configuration with all parameters |
| `utils.py` | Configuration utilities |
| `generate_havid_configs.py` | Generator for HAViD config variants |
| `*.yaml` | Dataset-specific configurations |

## Configuration System

FACT uses [YACS](https://github.com/rbgirshick/yacs) for configuration management:

```python
from fact_clip.configs.utils import setup_cfg

# Load config from YAML
cfg = setup_cfg(['fact_clip/configs/breakfast.yaml'], [])

# Override via command line
cfg = setup_cfg(['fact_clip/configs/breakfast.yaml'], ['aux.gpu', '0', 'lr', '0.001'])
```

## Default Parameters (`default.py`)

### Training Settings

```yaml
batch_size: 4
optimizer: "SGD"          # SGD or Adam
epoch: 2
lr: 0.1
lr_decay: -1              # Decay every N epochs (-1 = no decay)
momentum: 0.009
weight_decay: 0.0
clip_grad_norm: 10.0
```

### Auxiliary Settings

```yaml
aux:
  gpu: 1                  # GPU ID
  mark: ""                # Experiment marker
  runid: 0                # Run ID
  debug: false            # Debug mode (deterministic, offline wandb)
  wandb_project: "FACT"
  wandb_offline: false
  resume: "max"           # Resume from checkpoint ("", path, or "max")
  eval_every: 1000        # Evaluation interval
  print_every: 200        # Logging interval
```

### Dataset Settings

```yaml
dataset: "breakfast"
split: "split1"
sr: 1                     # Temporal sample rate
eval_bg: false            # Include background in evaluation

# Optional custom paths
feature_path: null
groundTruth_path: null
split_path: null
map_fname: null
```

### FACT Model Settings

```yaml
FACT:
  ntoken: 30              # Number of action tokens
  block: "iuUU"           # Block sequence (i=Input, u=Update, U=UpdateTDU)
  trans: false            # Use transcript during training
  fpos: true              # Frame positional encoding
  cmr: 0.3                # Channel masking rate
  mwt: 0.1                # Merge weight (action vs frame branch)
```

### Block Settings

**InputBlock (Bi)**:
```yaml
Bi:
  hid_dim: 512
  dropout: 0.5
  
  # Action branch
  a: "sca"                # Self+Cross Attention
  a_nhead: 8
  a_ffdim: 2048
  a_layers: 6
  a_dim: 512
  
  # Frame branch
  f: "cnn"                # CNN (MSTCN)
  f_layers: 10
  f_ln: true              # Layer normalization
  f_dim: 512
  f_ngp: 4                # Number of groups
```

**UpdateBlock (Bu)** and **UpdateBlockTDU (BU)** inherit from Bi with overrides.

### Loss Settings

```yaml
Loss:
  pc: 1.0                 # Probability cost weight (matching)
  a2fc: 1.0               # Attention cost weight (matching)
  match: "o2o"            # Matching type (o2o=one-to-one, o2m=one-to-many)
  bgw: 1.0                # Background class weight
  nullw: -1.0             # Null class weight (-1=auto-compute)
  sw: 0.0                 # Smoothness loss weight
```

### CLIP Settings

```yaml
use_clip: false           # Enable FACT_CLIP

CLIP:
  model_name: "openai/clip-vit-base-patch32"
  text_trainable: true
  temp: 0.07              # Temperature for InfoNCE
  precompute_text: true   # Cache text embeddings
  use_prompt: true        # Use prompt engineering
  text_emb_path: null     # Custom embedding path
  contrastive_weight: 0.5
  fact_loss_weight: 0.5
  projection_hidden_dim: 512
  projection_dropout: 0.1
```

### Holdout Settings

```yaml
holdout_mode: false       # Enable holdout training
holdout_classes: []       # Class indices to hold out
```

### Time Masking (Augmentation)

```yaml
TM:
  use: false
  t: 30                   # Mask duration
  p: 0.05                 # Probability
  m: 5                    # Number of masks
  inplace: true
```

## Dataset Configurations

### Breakfast (`breakfast.yaml`)

Standard configuration for Breakfast dataset (48 action classes).

### GTEA (`gtea.yaml`)

Configuration for GTEA dataset (11 action classes).

### HAViD Variants

HAViD has multiple view/hand configurations:

| Config | View | Hand | Annotation |
|--------|------|------|------------|
| `havid_view0_lh_pt.yaml` | Front | Left | Per-task |
| `havid_view0_rh_pt.yaml` | Front | Right | Per-task |
| `havid_view1_lh_pt.yaml` | Side | Left | Per-task |
| `havid_*_aa.yaml` | * | * | All-actions |

### Holdout Configuration (`havid_view0_lh_pt_holdout.yaml`)

Pre-configured for zero-shot evaluation:
- `holdout_mode: true`
- `holdout_classes: [51, 53, 61, 67, 56]`
- CLIP settings enabled

### CLIP Training (`openvocab_havid_view0_lh_pt.yaml`)

FACT_CLIP training configuration:
- `use_clip: true`
- Optimized CLIP parameters

## Using Configurations

### Command Line Override

```bash
python scripts/train.py --cfg fact_clip/configs/breakfast.yaml \
    --set aux.gpu 0 \
         lr 0.0001 \
         batch_size 2 \
         use_clip true
```

### Multiple Config Files

```bash
# Base config + overrides
python -m src.train \
    --cfg fact_clip/configs/breakfast.yaml fact_clip/configs/clip_settings.yaml
```

### Programmatic Access

```python
from fact_clip.configs.default import get_cfg_defaults
from yacs.config import CfgNode

cfg = get_cfg_defaults()
cfg.merge_from_file('fact_clip/configs/breakfast.yaml')
cfg.merge_from_list(['aux.gpu', '0'])
cfg.freeze()
```

## Generating HAViD Configs

Use `generate_havid_configs.py` to create configs for all HAViD variants:

```bash
python -m fact_clip.configs.generate_havid_configs
```

This generates configs for all view/hand/annotation combinations.

