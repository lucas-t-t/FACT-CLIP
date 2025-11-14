# FACT Open-Vocabulary Action Segmentation

This directory contains the implementation of **FACT_OpenVocab**, an open-vocabulary extension of the FACT model that enables zero-shot action segmentation using CLIP embeddings.

## Overview

FACT_OpenVocab adapts the original FACT model to work in an open-vocabulary setting by:
1. **Projecting existing I3D features** (2048-dim) to CLIP-compatible embedding space (512-dim)
2. **Using CLIP text encoder** to generate embeddings for action descriptions
3. **Computing similarity scores** instead of classification logits
4. **Training with contrastive loss** for visual-text alignment

This approach enables **zero-shot inference** on unseen action classes without retraining visual features.

## Architecture

```
I3D Features (2048-dim)  ──► Visual Projection ──► CLIP Space (512-dim)
                                                         │
                                                         ▼
                                                    Align with
                                                         │
                                                         ▼
Action Text ──► CLIP Text Encoder (trainable) ──► Text Embeddings (512-dim)
```

### Key Components

1. **VisualFeatureProjection**: Multi-layer projection (I3D → CLIP space)
2. **CLIPTextEncoder**: Trainable CLIP text encoder for action descriptions
3. **FACT_OpenVocab**: Main model integrating both encoders with FACT's temporal modeling
4. **Modified Blocks**: InputBlock_OV, UpdateBlock_OV, UpdateBlockTDU_OV compute similarities instead of logits
5. **Contrastive Losses**: Frame and action token losses based on cosine similarity

## Files Created

### Models
- `src/models/blocks_OpenVocab.py` - Open-vocabulary FACT model and blocks
- `src/models/loss_OpenVocab.py` - Contrastive loss functions

### Utilities
- `src/utils/dataset_OpenVocab.py` - Dataset utilities with text description support

### Configurations
- `src/configs/default.py` - Updated with CLIP configuration section
- `src/configs/openvocab_havid_view0_lh_pt.yaml` - Example configuration for HAViD

### Scripts
- `src/train_openvocab.py` - Training script with different LRs per component
- `src/eval_openvocab.py` - Evaluation script with zero-shot support

## Installation

Install additional dependencies:

```bash
pip install transformers>=4.30.0
```

## Configuration

The CLIP configuration section has been added to `src/configs/default.py`:

```python
_C.CLIP = CLIP = CN()
CLIP.model_name = "openai/clip-vit-b-32"
CLIP.text_trainable = True  # Fine-tune CLIP text encoder
CLIP.temp = 0.07  # Initial temperature (learnable)
CLIP.precompute_text = True  # Pre-compute text embeddings
CLIP.use_prompt = True  # Use prompt engineering
CLIP.projection_hidden_dim = 1024  # Hidden layer in projection
CLIP.projection_dropout = 0.1  # Dropout in projection
```

## Usage

### 1. Training on Seen Classes

Train the model on a dataset with standard action classes:

```bash
python -m src.train_openvocab \
    --cfg src/configs/openvocab_havid_view0_lh_pt.yaml \
    --set aux.gpu 0
```

**Key features:**
- Visual projection learns to map I3D features to CLIP space
- CLIP text encoder (trainable) adapts to video domain
- Different learning rates: CLIP text (lr × 0.1), projection & FACT blocks (lr × 1.0)

### 2. Standard Evaluation (Seen Classes)

Evaluate on the same classes used during training:

```bash
python -m src.eval_openvocab \
    --cfg src/configs/openvocab_havid_view0_lh_pt.yaml \
    --ckpt log/havid_openvocab/ckpts/network.iter-50000.net \
    --set aux.gpu 0
```

### 3. Zero-Shot Evaluation (Unseen Classes)

Evaluate on completely new action classes not seen during training:

```bash
# Option 1: Provide actions via command line
python -m src.eval_openvocab \
    --cfg src/configs/openvocab_havid_view0_lh_pt.yaml \
    --ckpt log/havid_openvocab/ckpts/network.iter-50000.net \
    --zero_shot \
    --unseen_actions "open_door,close_window,pick_up_phone" \
    --set aux.gpu 0

# Option 2: Provide actions via file
echo "open_door" > unseen_actions.txt
echo "close_window" >> unseen_actions.txt
echo "pick_up_phone" >> unseen_actions.txt

python -m src.eval_openvocab \
    --cfg src/configs/openvocab_havid_view0_lh_pt.yaml \
    --ckpt log/havid_openvocab/ckpts/network.iter-50000.net \
    --zero_shot \
    --unseen_actions_file unseen_actions.txt \
    --set aux.gpu 0
```

## Text Description Formatting

Action class names are automatically converted to natural language descriptions using prompt engineering:

```python
"crack_egg" → "a video of a person cracking an egg"
"pour_milk" → "a video of a person pouring milk"
"S01" → "a video of a person performing action S01"  # For HAViD codes
```

You can disable prompt engineering by setting `CLIP.use_prompt = False`.

## How It Works

### Training Phase

1. **Load existing I3D features** from HAViD/Breakfast datasets (no re-extraction needed)
2. **Project I3D features** to CLIP space via learnable projection layer
3. **Generate text embeddings** using CLIP text encoder (pre-computed for efficiency)
4. **Forward through FACT blocks** maintaining temporal modeling
5. **Compute similarities** between visual and text embeddings (cosine similarity)
6. **Optimize with contrastive loss** to align visual and text representations

### Inference Phase (Zero-Shot)

1. **Load trained model** (with learned I3D → CLIP projection)
2. **Encode new action descriptions** dynamically using CLIP text encoder
3. **Compute similarities** between projected visual features and new text embeddings
4. **Predict actions** based on highest similarity scores

### Key Advantages

- ✅ **No visual re-extraction**: Uses existing I3D features
- ✅ **Fast training**: Pre-computed text embeddings cached
- ✅ **Zero-shot capable**: Can recognize unseen actions via text descriptions
- ✅ **Maintains FACT architecture**: Keeps temporal modeling strengths
- ✅ **Trainable alignment**: Learns optimal I3D → CLIP mapping

## Optimizer Configuration

The training script uses different learning rates for different components:

```python
param_groups = [
    {'params': clip_text.parameters(), 'lr': cfg.lr * 0.1},      # Lower for CLIP
    {'params': visual_projection.parameters(), 'lr': cfg.lr},    # Full LR
    {'params': fact_blocks.parameters(), 'lr': cfg.lr},          # Full LR
]
```

**Rationale**: CLIP text encoder is pre-trained and should adapt slowly, while projection and FACT blocks need to learn from scratch.

## Expected Performance

- **Seen classes**: Comparable to original FACT (≈1-2% drop due to contrastive loss)
- **Unseen classes**: Depends on:
  - Quality of text descriptions
  - Semantic similarity between seen and unseen actions
  - Visual-text alignment learned during training

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in config (HAViD uses 2, Breakfast uses 4)
- Use smaller CLIP model: `CLIP.model_name = "openai/clip-vit-b-16"`

### Poor Zero-Shot Performance

- Check text descriptions quality (use descriptive names)
- Enable prompt engineering: `CLIP.use_prompt = True`
- Train longer for better visual-text alignment
- Try different CLIP models (ViT-L/14 is more powerful but slower)

### CLIP Model Download Issues

```python
# Pre-download CLIP model
from transformers import CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-b-32")
```

## Comparison with Original FACT

| Aspect | Original FACT | FACT_OpenVocab |
|--------|---------------|----------------|
| Input features | I3D (2048-dim) | I3D (2048-dim) → Projected |
| Action representation | Learned embeddings | CLIP text embeddings |
| Classification | Linear layer logits | Cosine similarity |
| Loss | Cross-entropy | Contrastive |
| Inference | Fixed classes | Dynamic text descriptions |
| Zero-shot | ❌ | ✅ |

## Citation

If you use this code, please cite both the original FACT paper and CLIP:

```bibtex
@inproceedings{lu2024fact,
  title={FACT: Frame-Action Cross-Attention Temporal Modeling for Efficient Action Segmentation},
  author={Lu, Zijia and Elhamifar, Ehsan},
  booktitle={CVPR},
  year={2024}
}

@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and others},
  booktitle={ICML},
  year={2021}
}
```

## Future Improvements

1. **Multi-modal fusion**: Combine I3D and CLIP visual features
2. **Learnable prompts**: Optimize text prompts end-to-end
3. **Temporal text grounding**: Use action descriptions with temporal markers
4. **Cross-dataset zero-shot**: Train on one dataset, test on another
5. **Few-shot learning**: Adapt to new classes with few examples

## Contact

For questions or issues, please refer to the original FACT project documentation and GitHub repository.



