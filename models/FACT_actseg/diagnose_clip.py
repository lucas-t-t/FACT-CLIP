#!/usr/bin/env python3
"""Diagnostic script to check CLIP implementation in FACT_CLIP."""

import torch
import sys
sys.path.insert(0, 'src')

from configs.utils import setup_cfg
from utils.dataset import create_dataset, load_action_mapping
from models.blocks import FACT_CLIP
from models.loss import MatchCriterion
from home import get_project_base
import os

print("="*80)
print("DIAGNOSING FACT_CLIP IMPLEMENTATION")
print("="*80)

# Load config
cfg = setup_cfg(['src/configs/havid_view0_lh_pt_holdout.yaml'], None)
BASE = get_project_base()

print(f"\n1. Configuration:")
print(f"   - use_clip: {cfg.use_clip}")
print(f"   - dataset: {cfg.dataset}")
print(f"   - ntoken: {cfg.FACT.ntoken}")
print(f"   - holdout_mode: {cfg.holdout_mode}")
print(f"   - holdout_classes: {cfg.holdout_classes}")

# Load dataset
print(f"\n2. Loading dataset...")
dataset, test_dataset = create_dataset(cfg)
print(f"   - Training videos: {len(dataset.video_list)}")
print(f"   - Test videos: {len(test_dataset.video_list)}")
print(f"   - Number of classes: {dataset.nclasses}")

# Check text embeddings
print(f"\n3. Checking text embeddings...")
variant = cfg.dataset.replace("havid_", "")
map_fname = os.path.join(BASE, 'data', 'HAViD', 'ActionSegmentation', 'data', variant, 'mapping.txt')

if os.path.exists(map_fname):
    print(f"   ✓ Mapping file found: {map_fname}")
    label2index, index2label = load_action_mapping(map_fname)
    print(f"   ✓ Loaded {len(index2label)} class mappings")
    
    # Get text embeddings
    from utils.text_embeddings import get_or_compute_text_embeddings
    text_embeddings = get_or_compute_text_embeddings(
        cfg, label2index, index2label, device='cpu'
    )
    print(f"   ✓ Text embeddings shape: {text_embeddings.shape}")
    print(f"   ✓ Text embeddings normalized: {torch.allclose(text_embeddings.norm(dim=1), torch.ones(text_embeddings.shape[0]), atol=0.01)}")
else:
    print(f"   ✗ Mapping file NOT found: {map_fname}")
    text_embeddings = None

# Create model
print(f"\n4. Creating FACT_CLIP model...")
model = FACT_CLIP(cfg, dataset.input_dimension, dataset.nclasses, text_embeddings=text_embeddings)
print(f"   ✓ Model created successfully")

# Check if text embeddings are in model
if hasattr(model, 'text_embeddings') and model.text_embeddings is not None:
    print(f"   ✓ Model has text_embeddings: {model.text_embeddings.shape}")
else:
    print(f"   ✗ Model does NOT have text_embeddings loaded!")

# Check action projection
if hasattr(model, 'action_projection'):
    print(f"   ✓ Model has action_projection")
    # Check input/output dims
    first_layer = model.action_projection.projection[0]
    print(f"   ✓ Projection input dim: {first_layer.in_features}")
    print(f"   ✓ Projection output will be: 512")
else:
    print(f"   ✗ Model does NOT have action_projection!")

# Test forward pass
print(f"\n5. Testing forward pass...")
try:
    # Get a sample from dataset
    vname, seq, train_label, eval_label = dataset[0]
    seq = seq.unsqueeze(0).cuda()  # Add batch dim
    train_label = train_label.unsqueeze(0).cuda()
    
    # Create criterion
    model.mcriterion = MatchCriterion(cfg, dataset.nclasses, bg_ids=[])
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        video_saves = model([seq], [train_label[0]], compute_loss=False)
    
    print(f"   ✓ Forward pass successful!")
    
    # Check if projected embeddings were created
    if hasattr(model, 'projected_action_embeddings'):
        print(f"   ✓ Projected action embeddings created: {model.projected_action_embeddings.shape}")
    else:
        print(f"   ✗ Projected action embeddings NOT created!")
    
except Exception as e:
    print(f"   ✗ Forward pass FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test loss computation
print(f"\n6. Testing loss computation...")
try:
    model.train()
    with torch.enable_grad():
        loss, video_saves = model([seq], [train_label[0]], compute_loss=True)
    
    print(f"   ✓ Loss computation successful: {loss.item():.4f}")
    
    if hasattr(model, 'fact_loss'):
        print(f"   ✓ FACT loss: {model.fact_loss.item():.4f}")
    if hasattr(model, 'contrastive_loss'):
        print(f"   ✓ Contrastive loss: {model.contrastive_loss.item():.4f}")
    else:
        print(f"   ✗ Contrastive loss NOT computed!")
    
except Exception as e:
    print(f"   ✗ Loss computation FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)

