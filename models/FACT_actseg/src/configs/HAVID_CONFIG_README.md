# HAViD Dataset Configuration Guide

This guide explains how to use the HAViD dataset configurations with the FACT model.

## HAViD Dataset Structure

The HAViD (Human Action Video Dataset) contains multiple variants organized by:

- **Views**: 
  - `view0` = side view (master camera 0)
  - `view1` = front view (subordinate camera 1) 
  - `view2` = top view (subordinate camera 2)

- **Hands**:
  - `lh` = left hand
  - `rh` = right hand

- **Annotation Types**:
  - `pt` = primitive task (higher-level actions)
  - `aa` = atomic action (more granular actions)

## Available Configurations

### 1. Generic HAViD Config
- **File**: `havid.yaml`
- **Usage**: Base configuration that can be customized for any HAViD variant
- **Recommended for**: Custom experiments or when you want to specify paths manually

### 2. Specific Variant Configs
- **File**: `havid_view0_lh_pt.yaml` - Side view, left hand, primitive tasks
- **File**: `havid_view0_lh_aa.yaml` - Side view, left hand, atomic actions

## Key Configuration Differences

### Model Parameters
- **Primitive Tasks (pt)**: `ntoken: 40` - Fewer tokens for higher-level actions
- **Atomic Actions (aa)**: `ntoken: 60` - More tokens for granular actions

### Training Parameters
- **Batch Size**: `2` (reduced from breakfast's 4) due to longer video sequences
- **Learning Rate**: `0.0001` (same as breakfast)
- **Epochs**: `150` (same as breakfast)

## Usage Instructions

### Method 1: Using gen_config.py (Recommended)

```bash
# For primitive tasks (view0, left hand)
python utils/gen_config.py \
  --dataset_path /path/to/HAViD/ActionSegmentation/view0_lh_pt \
  --dataset_name havid_view0_lh_pt \
  --output_config havid_view0_lh_pt_final.yaml \
  --base_config configs/havid_view0_lh_pt.yaml \
  --batch_size 2 \
  --token_ratio 1.5

# For atomic actions (view0, left hand)
python utils/gen_config.py \
  --dataset_path /path/to/HAViD/ActionSegmentation/view0_lh_aa \
  --dataset_name havid_view0_lh_aa \
  --output_config havid_view0_lh_aa_final.yaml \
  --base_config configs/havid_view0_lh_aa.yaml \
  --batch_size 2 \
  --token_ratio 1.5
```

### Method 2: Manual Path Configuration

Edit the configuration files to set the correct paths:

```yaml
feature_path: "/path/to/HAViD/ActionSegmentation/features"
groundTruth_path: "/path/to/HAViD/ActionSegmentation/view0_lh_pt/groundTruth"
split_path: "/path/to/HAViD/ActionSegmentation/view0_lh_pt/splits"
map_fname: "/path/to/HAViD/ActionSegmentation/view0_lh_pt/mapping.txt"
```

## Training Commands

Once the configuration is ready, train the model:

```bash
# For primitive tasks
python3 -m src.train --cfg configs/havid_view0_lh_pt_final.yaml --set aux.gpu 0 split "split1"

# For atomic actions  
python3 -m src.train --cfg configs/havid_view0_lh_aa_final.yaml --set aux.gpu 0 split "split1"
```

## Creating Configs for Other Variants

To create configurations for other HAViD variants (different views, hands, or annotation types):

1. Copy an existing config file
2. Update the `dataset` name and `aux.mark` fields
3. Adjust `ntoken` based on annotation complexity:
   - Primitive tasks: 30-40 tokens
   - Atomic actions: 50-70 tokens
4. Use `gen_config.py` to set the correct paths

## Dataset Requirements

Ensure your HAViD dataset has the following structure:

```
ActionSegmentation/
├── features/                    # I3D features (.npy files)
│   ├── S01A01I01M0.npy
│   ├── S01A01I01S1.npy
│   └── ...
├── view0_lh_pt/                # Side view, left hand, primitive tasks
│   ├── groundTruth/
│   │   ├── S01A01I01M0.txt
│   │   └── ...
│   ├── splits/
│   │   ├── training.bundle
│   │   └── testing.bundle
│   └── mapping.txt
├── view0_lh_aa/                # Side view, left hand, atomic actions
│   ├── groundTruth/
│   ├── splits/
│   └── mapping.txt
└── ... (other variants)
```

## HAViD Integration Changes

To make HAViD work with the FACT model, several modifications were required:

### 1. Dataset Loading Support (`src/utils/dataset.py`)

Added HAViD dataset support to the `create_dataset()` function:

```python
elif cfg.dataset.startswith("havid"):
    # HAViD dataset variants (e.g., "havid_view0_lh_pt", "havid_view1_rh_aa")
    variant = cfg.dataset.replace("havid_", "")  # e.g., "view0_lh_pt"
    havid_base = BASE + 'data/HAViD/ActionSegmentation/data'
    
    map_fname = f'{havid_base}/{variant}/mapping.txt'
    dataset_path = f'{havid_base}/{variant}/'
    feature_path = f'{havid_base}/features'
    train_split_fname = f'{havid_base}/{variant}/splits/train.{cfg.split}.bundle'
    test_split_fname = f'{havid_base}/{variant}/splits/test.{cfg.split}.bundle'
    
    feature_transpose = True  # HAViD features are (D, T), need (T, D)
    bg_class = [0]
    
    # Set average_transcript_len based on annotation type
    if variant.endswith('_pt'):  # primitive tasks
        average_transcript_len = 8.0
    elif variant.endswith('_aa'):  # atomic actions
        average_transcript_len = 15.0
    else:
        average_transcript_len = 10.0
```

### 2. Video Name Processing

Fixed video name handling for HAViD split files:

```python
# For test dataset
if cfg.dataset in ['breakfast', '50salads', 'gtea']: 
    test_video_list = [ v[:-4] for v in test_video_list ] 
elif cfg.dataset.startswith('havid'):
    test_video_list = [ v[:-4] for v in test_video_list if v.endswith('.txt') ]

# For train dataset
if cfg.dataset in ['breakfast', '50salads', 'gtea']: 
    video_list = [ v[:-4] for v in video_list ] 
elif cfg.dataset.startswith('havid'):
    video_list = [ v[:-4] for v in video_list if v.endswith('.txt') ]
```

**Why needed**: HAViD split files contain video names with `.txt` extension (e.g., `S02A04I01M0.txt`) but feature files have `.npy` extension (e.g., `S02A04I01M0.npy`).

### 3. Feature Dimension Handling

**Key insight**: HAViD I3D features have shape `(2048, T)` but FACT expects `(T, 2048)`.

- **Solution**: Set `feature_transpose = True` for HAViD
- **Raw HAViD features**: `(2048, 1445)` → 2048 feature dims, 1445 time steps
- **After transpose**: `(1445, 2048)` → 1445 time steps, 2048 feature dims

### 4. Model Configuration Adjustments

Optimal HAViD configuration differs from breakfast/gtea:

```yaml
Bi:
  a_dim: 256          # Reduced from 512
  f_dim: 256          # Reduced from 512  
  f: m                # Use MSTCN (not m2/MSTCN2)
  f_ln: false         # Match other datasets
  f_ngp: 1            # Match other datasets
  hid_dim: 512        # Keep standard
  dropout: 0.0        # Low dropout for HAViD

FACT:
  ntoken: 43          # Computed from HAViD statistics

batch_size: 2         # Reduced due to longer sequences
dataset: havid_view0_lh_pt  # Specific variant name
```

**Why these changes**:
- **Smaller dimensions**: HAViD features are high-dimensional (2048), smaller processing dims improve efficiency
- **MSTCN over MSTCN2**: MSTCN (`f: m`) adapts better to varying input dimensions than MSTCN2 (`f: m2`)
- **Reduced batch size**: HAViD videos are longer than breakfast/gtea

### 5. Generator Script Enhancements (`src/utils/gen_config.py`)

Added robust encoding and explicit features path support:

```python
# Robust text reading with encoding fallback
def _read_text_with_fallback(path: Path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, 'r', encoding='latin-1') as f:
            return f.read()

# Support for shared features folder
parser.add_argument(
    '--features_path',
    help='Optional explicit path to features folder for shared features'
)
```

**Usage example**:
```bash
python -m src.utils.gen_config \
  --dataset_path /path/to/HAViD/ActionSegmentation/data/view0_lh_pt \
  --features_path /path/to/HAViD/ActionSegmentation/data/features \
  --dataset_name havid_view0_lh_pt \
  --base_config src/configs/breakfast.yaml \
  --output_config havid_view0_lh_pt_final.yaml
```

## Troubleshooting

### Common Issues and Solutions

#### 1. **Dimension Mismatch Errors**
```
RuntimeError: expected input[X, Y, Z] to have A channels, but got B channels
```
**Solution**: Ensure `feature_transpose = True` in dataset.py for HAViD

#### 2. **File Not Found Errors**
```
FileNotFoundError: S02A04I01M0.txt.npy
```
**Solution**: The video name processing fixes in dataset.py strip `.txt` extensions

#### 3. **Unicode Decode Errors**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```
**Solution**: Use updated gen_config.py with encoding fallback

#### 4. **WANDB Permission Errors**
```
wandb.errors.CommError: Error 403: Forbidden
```
**Solution**: Disable wandb:
```bash
export WANDB_MODE=disabled
# or use aux.debug True
```

### Validation Steps

1. **Check feature dimensions**:
```bash
python -c "import numpy as np; f=np.load('data/HAViD/ActionSegmentation/data/features/S01A04I01M0.npy'); print(f.shape)"
# Should show: (2048, T) where T is number of frames
```

2. **Verify dataset loading**:
```bash
python -c "from src.utils.dataset import create_dataset; import yaml; 
with open('src/configs/havid_view0_lh_pt_clean.yaml') as f: cfg=yaml.safe_load(f)"
```

3. **Test training start**:
```bash
python -m src.train --cfg src/configs/havid_view0_lh_pt_clean.yaml --set aux.debug True aux.gpu 0
```

## Notes

- The HAViD dataset uses I3D features, which are compatible with the FACT model
- Video naming follows the pattern: `S--A--I----` (Subject-Attempt-Instruction-Camera)
- The `gen_config.py` script will automatically compute the optimal number of action tokens based on your dataset statistics
- Consider using different batch sizes based on your GPU memory and video sequence lengths
- **Feature transpose is critical**: HAViD features must be transposed from `(D,T)` to `(T,D)` format
- **Use MSTCN (`f: m`)** instead of MSTCN2 (`f: m2`) for better dimension flexibility
