# Models

This directory contains the FACT model architecture and its CLIP-extended variant.

## Files

| File | Description |
|------|-------------|
| `blocks.py` | Main FACT and FACT_CLIP model definitions |
| `loss.py` | Loss functions including MatchCriterion and InfoNCE |
| `basic.py` | Building blocks (MSTCN, attention layers, positional encoding) |
| `blocks_SepVerbNoun.py` | Variant for Epic-Kitchens with separate verb/noun prediction |

## FACT Architecture

The FACT model uses a sequence of blocks that process frame and action features in parallel:

```
Input → [InputBlock] → [UpdateBlock]* → [UpdateBlockTDU]* → Output
```

### Block Types

| Block | Config | Description |
|-------|--------|-------------|
| `InputBlock` | `i` | Initial processing of frame features, creates action queries |
| `UpdateBlock` | `u` | Bidirectional cross-attention refinement |
| `UpdateBlockTDU` | `U` | UpdateBlock with temporal downsampling/upsampling |

Configure block sequence via `FACT.block`, e.g., `"iuUU"` = InputBlock + UpdateBlock + 2x UpdateBlockTDU

### Key Components

**Frame Branch**:
- MSTCN or MSTCN2 for temporal convolution
- Outputs frame features + class logits

**Action Branch**:
- Learnable action queries (when transcript unavailable)
- Self-attention + cross-attention layers
- Outputs action features + class logits

**Cross-Attention**:
- `f2a`: Frames attend to actions
- `a2f`: Actions attend to frames
- Enables bidirectional information flow

## FACT_CLIP

Extends FACT for zero-shot action segmentation:

```python
class FACT_CLIP(nn.Module):
    """
    Adds CLIP text embedding alignment to vanilla FACT.
    Projects frame features to CLIP space and uses contrastive loss.
    """
```

### Key Additions

1. **FeatureProjection**: Projects frame features to 512-dim CLIP space
   ```python
   frame_features → Linear → LayerNorm → ReLU → Dropout → Linear → Normalize
   ```

2. **Text Embeddings**: Pre-computed CLIP embeddings stored as buffer
   ```python
   self.register_buffer('text_embeddings', text_embeddings)  # (n_classes, 512)
   ```

3. **Contrastive Loss**: InfoNCE loss aligns frames with text
   ```python
   contrastive_loss = infonce_contrastive_loss(
       projected_frame_embeddings,  # (T, B, 512)
       text_embeddings,             # (n_classes, 512)
       labels,                      # (T,)
       temperature=0.07
   )
   ```

4. **Zero-Shot Evaluation**: `eval_with_clip()` uses cosine similarity for prediction

## Loss Functions (`loss.py`)

### MatchCriterion

Handles bipartite matching between action tokens and ground truth segments:

- `match()`: Hungarian matching based on class probability and attention IoU
- `frame_loss()`: Per-frame classification loss
- `action_token_loss()`: Per-token classification loss
- `cross_attn_loss()`: Attention alignment loss

### InfoNCE Contrastive Loss

```python
def infonce_contrastive_loss(projected_embeddings, text_embeddings, labels, temperature):
    """
    Symmetric contrastive loss:
    - Video-to-text: frames should match their class text embedding
    - Text-to-video: text embeddings should match their class frames
    """
```

### Smooth Loss

Temporal smoothness regularization on predictions.

## Usage

```python
from src.models.blocks import FACT, FACT_CLIP
from src.models.loss import MatchCriterion

# Vanilla FACT
model = FACT(cfg, input_dim=2048, n_classes=75)
model.mcriterion = MatchCriterion(cfg, n_classes=75, bg_class=[0])

# FACT_CLIP
text_embeddings = load_text_embeddings(...)  # (n_classes, 512)
model = FACT_CLIP(cfg, input_dim=2048, n_classes=75, text_embeddings=text_embeddings)
```

## Configuration

Key model configs in `default.py`:

```yaml
FACT:
  ntoken: 30          # Number of action tokens
  block: "iuUU"       # Block sequence
  trans: false        # Use transcript (if available)
  fpos: true          # Use frame positional encoding
  cmr: 0.3            # Channel masking rate
  mwt: 0.1            # Merge weight for action/frame branches

Bi:                   # InputBlock config
  hid_dim: 512
  a_dim: 512
  f_layers: 10
  a_layers: 6

Bu:                   # UpdateBlock config (inherits from Bi)
  f_layers: 5
  a_layers: 1

BU:                   # UpdateBlockTDU config
  s_layers: 1         # GRU layers for segment refinement
```

