# Utilities

This directory contains dataset handling, evaluation, and helper utilities.

## Files

| File | Description |
|------|-------------|
| `dataset.py` | Dataset loading and DataLoader implementation |
| `evaluate.py` | Evaluation metrics and Checkpoint management |
| `train_tools.py` | Training utilities (checkpoint resume, null weight computation) |
| `text_embeddings.py` | CLIP text embedding generation |
| `havid_text_prompts.py` | HAViD action code to natural language conversion |
| `analyze_holdout_classes.py` | Tool for selecting holdout classes |
| `utils.py` | General utilities (numpy conversion, label parsing) |
| `extract_epic_kitchens.py` | Epic-Kitchens feature extraction |

## Dataset Loading (`dataset.py`)

### Dataset Class

```python
class Dataset:
    """
    Lazy-loading dataset for action segmentation.
    
    Attributes:
        video_list: List of video names
        nclasses: Number of action classes
        bg_class: Background class indices
        input_dimension: Feature dimension
    """
```

### DataLoader

Custom DataLoader that handles variable-length sequences:

```python
loader = DataLoader(dataset, batch_size=4, shuffle=True)
for vnames, seq_list, train_labels, eval_labels in loader:
    # seq_list: List of (T, feature_dim) tensors
    # train_labels: List of (T,) label tensors
```

### create_dataset()

Factory function for creating train/test datasets:

```python
dataset, test_dataset = create_dataset(cfg)
```

Supports:
- **Automatic holdout filtering**: Removes videos containing holdout classes from training
- **Multiple datasets**: Breakfast, GTEA, HAViD variants, EgoProceL, Epic-Kitchens
- **Custom paths**: Override default paths via config

## Text Embeddings (`text_embeddings.py`)

Generate and cache CLIP text embeddings for action classes:

```python
from fact_clip.utils.text_embeddings import get_or_compute_text_embeddings

# Returns (n_classes, 512) tensor
text_embeddings = get_or_compute_text_embeddings(cfg, label2index, index2label, device)
```

### Functions

| Function | Description |
|----------|-------------|
| `generate_text_descriptions()` | Convert class labels to natural language |
| `precompute_text_embeddings()` | Compute CLIP embeddings for descriptions |
| `load_text_embeddings()` | Load cached embeddings from file |
| `get_or_compute_text_embeddings()` | Load or compute embeddings |

### Caching

Embeddings are cached to `data/<dataset>_text_embeddings.pt` by default.

## HAViD Text Prompts (`havid_text_prompts.py`)

Converts HAViD action codes to natural language for CLIP:

```python
from fact_clip.utils.havid_text_prompts import generate_action_prompt

prompt = generate_action_prompt("sshc1dh")
# "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver"
```

### HAViD Label Format

HAViD uses compact action codes: `[verb][object][target][tool]`

| Code | Meaning |
|------|---------|
| `s` | screw |
| `g` | grasp |
| `i` | insert |
| `p` | place |
| `sh` | hex screw |
| `sp` | phillips screw |
| `c1-c4` | cylinder holes 1-4 |
| `dh` | hex screwdriver |
| `dp` | phillips screwdriver |

### Example Conversions

| HAViD Code | Natural Language |
|------------|------------------|
| `sshc1dh` | "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver" |
| `gsh` | "a person grasps a hex screw" |
| `pglg1` | "a person places a large gear onto gear plate hole 1" |

## Evaluation (`evaluate.py`)

### Checkpoint Class

Manages evaluation results and metrics:

```python
ckpt = Checkpoint(
    iteration=10000,
    bg_class=[0],           # Background classes to exclude
    holdout_classes=[5,10], # Unseen classes (for zero-shot)
    seen_classes=[...]      # Seen classes
)

# Accumulate predictions
ckpt.add_prediction(video_name, gt_labels, predictions)

# Compute metrics
ckpt.compute_metrics()
print(ckpt.metrics)
# {'Acc': 75.3, 'F1@0.50': 65.8, 'Acc-seen': 78.2, 'Acc-unseen': 45.1, ...}
```

### Metrics

| Metric | Description |
|--------|-------------|
| `Acc` | Frame-wise accuracy |
| `AccFG` | Accuracy excluding background |
| `Edit` | Edit distance score |
| `F1@IoU` | F1 score at IoU thresholds (0.10, 0.25, 0.50) |
| `*-seen` | Metrics on seen classes only |
| `*-unseen` | Metrics on holdout classes only |

## Holdout Class Analysis (`analyze_holdout_classes.py`)

Analyze dataset to select appropriate holdout classes:

```bash
python -m src.utils.analyze_holdout_classes \
    --dataset havid_view0_lh_pt \
    --n_frequent 5 \
    --n_medium 5
```

Outputs:
- Class frequency statistics
- Recommended holdout classes
- Training data impact analysis

## Training Tools (`train_tools.py`)

### resume_ckpt()

Resume training from checkpoint:

```python
global_step, ckpt_file = resume_ckpt(cfg, logdir)
if ckpt_file:
    model.load_state_dict(torch.load(ckpt_file))
```

### compute_null_weight()

Automatically compute loss weight for null class based on frequency:

```python
compute_null_weight(cfg, dataset)
# Sets cfg.Loss.nullw based on action segment statistics
```

### save_results()

Save predictions to checkpoint:

```python
save_results(checkpoint, video_names, labels, model_outputs)
```

