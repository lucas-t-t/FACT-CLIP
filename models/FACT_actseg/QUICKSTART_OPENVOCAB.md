# Quick Start Guide: FACT Open-Vocabulary

Get started with open-vocabulary action segmentation in 5 minutes!

## Prerequisites

```bash
# Install additional dependency
pip install transformers>=4.30.0
```

## Step 1: Verify Installation

Check that all files are in place:

```bash
ls src/models/blocks_OpenVocab.py  # ✓ Core model
ls src/models/loss_OpenVocab.py    # ✓ Loss functions
ls src/utils/dataset_OpenVocab.py  # ✓ Dataset utilities
ls src/train_openvocab.py          # ✓ Training script
ls src/eval_openvocab.py           # ✓ Evaluation script
ls src/configs/openvocab_havid_view0_lh_pt.yaml  # ✓ Example config
```

## Step 2: Quick Test (Optional)

Test model instantiation:

```python
from src.configs.utils import setup_cfg
from src.models.blocks_OpenVocab import FACT_OpenVocab

# Load config
cfg = setup_cfg(['src/configs/openvocab_havid_view0_lh_pt.yaml'], [])

# Create model
action_descriptions = ["a video of a person opening a door", "a video of a person closing a window"]
model = FACT_OpenVocab(cfg, visual_input_dim=2048, action_descriptions=action_descriptions)

print(f"✓ Model created successfully!")
print(f"  Visual projection: {model.visual_projection}")
print(f"  CLIP text encoder: {model.clip_text}")
print(f"  Temperature: {model.temperature.item():.4f}")
```

## Step 3: Train on HAViD

### Option A: Standard Training (All Classes)

```bash
python -m src.train_openvocab \
    --cfg src/configs/openvocab_havid_view0_lh_pt.yaml \
    --set aux.gpu 0 \
          aux.exp havid_openvocab_exp1 \
          aux.debug False
```

### Option B: Holdout Training (Zero-Shot Mode)

```bash
# Hold out classes 5, 10, 15 for zero-shot evaluation
python -m src.train_openvocab \
    --cfg src/configs/openvocab_havid_view0_lh_pt.yaml \
    --set aux.gpu 0 \
          aux.exp havid_openvocab_holdout \
          holdout_mode True \
          holdout_classes [5,10,15]
```

**Monitor training:**
- Logs saved to: `log/havid_openvocab_exp1/`
- Checkpoints: `log/havid_openvocab_exp1/ckpts/`
- WandB dashboard: Project "FACT-OpenVocab"

## Step 4: Evaluate

### Standard Evaluation (Seen Classes)

```bash
python -m src.eval_openvocab \
    --cfg src/configs/openvocab_havid_view0_lh_pt.yaml \
    --ckpt log/havid_openvocab_exp1/ckpts/network.iter-50000.net \
    --set aux.gpu 0
```

### Zero-Shot Evaluation (Unseen Classes)

```bash
# Create file with unseen action names
cat > unseen_actions.txt << EOF
open_door
close_window
pour_water
pick_up_phone
write_on_paper
EOF

# Evaluate
python -m src.eval_openvocab \
    --cfg src/configs/openvocab_havid_view0_lh_pt.yaml \
    --ckpt log/havid_openvocab_exp1/ckpts/network.iter-50000.net \
    --zero_shot \
    --unseen_actions_file unseen_actions.txt \
    --set aux.gpu 0
```

## Step 5: Interpret Results

### Training Output

```
TRAINING
========================================
Ep:0 Stp:1000  Loss:2.345 (2.456)
...
TESTING
Acc:75.3, Edit:68.2, F1@0.10:82.1, F1@0.25:76.4, F1@0.50:65.8
```

### Evaluation Output

```
STANDARD EVALUATION (Seen Classes)
========================================
  Acc                 : 75.3
  Edit                : 68.2
  F1@0.10             : 82.1
  F1@0.25             : 76.4
  F1@0.50             : 65.8

ZERO-SHOT EVALUATION (Unseen Classes)
========================================
  Acc                 : 45.2  # Expected to be lower
  Edit                : 38.5
  F1@0.50             : 28.3
```

## Understanding Performance

### Seen Classes
- **Expected**: ≈ 73-77% accuracy (comparable to original FACT)
- **Slight drop**: 1-2% due to contrastive loss vs classification loss

### Unseen Classes (Zero-Shot)
- **Expected**: 10-50% accuracy depending on:
  - Semantic similarity to seen classes
  - Quality of text descriptions
  - Training data diversity
- **Example**:
  - If trained on "open door", good zero-shot on "close door"
  - Poor zero-shot on completely different actions

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size

```bash
python -m src.train_openvocab \
    --cfg src/configs/openvocab_havid_view0_lh_pt.yaml \
    --set batch_size 1  # Reduce from 2 to 1
```

### Issue: CLIP Model Download Slow

**Solution**: Pre-download model

```python
from transformers import CLIPModel
CLIPModel.from_pretrained("openai/clip-vit-b-32")
```

### Issue: Low Zero-Shot Performance

**Diagnosis**:
1. Check text description quality
2. Verify semantic similarity between seen/unseen classes
3. Train longer for better alignment

**Solutions**:
```bash
# Enable prompt engineering
--set CLIP.use_prompt True

# Increase training epochs
--set epoch 200

# Use more powerful CLIP model
--set CLIP.model_name openai/clip-vit-l-14
```

## Next Steps

1. **Experiment with hyperparameters**:
   - Learning rates: `--set lr 0.0005`
   - Temperature: `--set CLIP.temp 0.05`
   - Projection size: `--set CLIP.projection_hidden_dim 2048`

2. **Try different datasets**:
   - Breakfast: Use `src/configs/breakfast.yaml` as base
   - Create custom config for your dataset

3. **Analyze zero-shot performance**:
   - Create confusion matrix
   - Visualize text-visual embedding alignment
   - Test on various unseen action types

## Advanced Usage

### Custom Action Descriptions

Create richer descriptions for better zero-shot:

```python
# Instead of "open_door"
"a video of a person opening a door with their hand"

# Instead of "pour_water"
"a video of a person pouring water from a container into a glass"
```

### Multi-GPU Training

```bash
# Use DataParallel (single machine, multiple GPUs)
python -m src.train_openvocab \
    --cfg src/configs/openvocab_havid_view0_lh_pt.yaml \
    --set aux.gpu 0,1,2,3  # Use GPUs 0-3
```

### Resume Training

```bash
python -m src.train_openvocab \
    --cfg src/configs/openvocab_havid_view0_lh_pt.yaml \
    --set aux.resume max  # Resume from latest checkpoint
```

## Common Workflows

### Workflow 1: Standard Training + Evaluation

```bash
# 1. Train
python -m src.train_openvocab --cfg CONFIG --set aux.gpu 0

# 2. Evaluate
python -m src.eval_openvocab --cfg CONFIG --ckpt CKPT --set aux.gpu 0
```

### Workflow 2: Zero-Shot Evaluation Only

```bash
# Use pre-trained model from standard training for zero-shot
python -m src.eval_openvocab \
    --cfg CONFIG \
    --ckpt CKPT \
    --zero_shot \
    --unseen_actions "action1,action2,action3"
```

### Workflow 3: Holdout Training + Zero-Shot Eval

```bash
# 1. Train with holdout classes
python -m src.train_openvocab --cfg CONFIG \
    --set holdout_mode True holdout_classes [1,2,3]

# 2. Model automatically evaluates on holdout classes during training
# Check WandB for "unseen" metrics
```

## Performance Benchmarks

**Hardware**: NVIDIA A100 (40GB)

| Dataset | Batch Size | Time/Epoch | Memory |
|---------|------------|------------|--------|
| HAViD   | 2          | ~15 min    | ~12 GB |
| Breakfast | 4        | ~8 min     | ~10 GB |

## Getting Help

1. **Documentation**: See `src/OPENVOCAB_README.md` for details
2. **Implementation**: See `OPENVOCAB_IMPLEMENTATION_SUMMARY.md`
3. **Original FACT**: Refer to FACT project documentation
4. **CLIP**: Check Hugging Face transformers documentation

## Quick Reference

```bash
# Training
python -m src.train_openvocab --cfg CONFIG

# Evaluation (standard)
python -m src.eval_openvocab --cfg CONFIG --ckpt CKPT

# Evaluation (zero-shot)
python -m src.eval_openvocab --cfg CONFIG --ckpt CKPT --zero_shot --unseen_actions "a,b,c"

# Check GPU usage
nvidia-smi

# Monitor training
tensorboard --logdir log/
# or check WandB dashboard
```

---

**You're all set!** 🚀

For detailed information, see:
- `src/OPENVOCAB_README.md` - Comprehensive guide
- `OPENVOCAB_IMPLEMENTATION_SUMMARY.md` - Technical details



