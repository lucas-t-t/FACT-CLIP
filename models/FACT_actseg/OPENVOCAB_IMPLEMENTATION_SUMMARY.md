# Open-Vocabulary FACT Implementation Summary

## ✅ Implementation Complete

All planned components have been successfully implemented following the feature projection approach (Option 2).

## Files Created

### 1. Core Model (`src/models/blocks_OpenVocab.py`)
- **CLIPTextEncoder**: Trainable CLIP text encoder (512-dim embeddings)
- **VisualFeatureProjection**: Projects I3D features (2048-dim) → CLIP space (512-dim)
- **FACT_OpenVocab**: Main model class integrating:
  - Visual projection module
  - CLIP text encoder
  - FACT temporal blocks (InputBlock_OV, UpdateBlock_OV, UpdateBlockTDU_OV)
  - Learnable temperature parameter
  - Text embedding caching for training efficiency
- **Block_OV classes**: Modified blocks computing similarity instead of logits

### 2. Loss Functions (`src/models/loss_OpenVocab.py`)
- **contrastive_frame_loss**: Frame-level contrastive loss
- **contrastive_action_token_loss**: Action token contrastive loss
- **MatchCriterion_OV**: Inherits matching logic, replaces classification losses with contrastive versions

### 3. Dataset Utilities (`src/utils/dataset_OpenVocab.py`)
- **format_action_description**: Converts class names to natural language
  - "crack_egg" → "a video of a person cracking an egg"
  - Handles HAViD codes, breakfast actions, etc.
- **precompute_text_embeddings**: Pre-computes CLIP text embeddings for efficiency
- **create_dataset_openvocab**: Extended dataset loader with text description support

### 4. Configuration Files
- **`src/configs/default.py`**: Added CLIP configuration section
  ```python
  CLIP.model_name = "openai/clip-vit-b-32"
  CLIP.text_trainable = True
  CLIP.temp = 0.07
  CLIP.precompute_text = True
  CLIP.use_prompt = True
  CLIP.projection_hidden_dim = 1024
  CLIP.projection_dropout = 0.1
  ```
- **`src/configs/openvocab_havid_view0_lh_pt.yaml`**: Example HAViD configuration

### 5. Training Script (`src/train_openvocab.py`)
- Loads dataset with text descriptions
- Creates FACT_OpenVocab model
- Registers pre-computed text embeddings
- **Different learning rates per component**:
  - CLIP text encoder: lr × 0.1 (lower for pre-trained model)
  - Visual projection: lr × 1.0 (full learning rate)
  - FACT blocks: lr × 1.0 (full learning rate)
- Training loop with contrastive loss optimization
- Supports holdout/zero-shot training mode

### 6. Evaluation Script (`src/eval_openvocab.py`)
- **Standard evaluation**: On seen classes with cached embeddings
- **Zero-shot evaluation**: On unseen classes with dynamic text encoding
- Support for providing unseen actions via:
  - Command line: `--unseen_actions "action1,action2"`
  - File: `--unseen_actions_file path/to/actions.txt`

### 7. Documentation (`src/OPENVOCAB_README.md`)
- Comprehensive guide covering:
  - Architecture overview
  - Installation and dependencies
  - Usage examples (training, evaluation, zero-shot)
  - Configuration options
  - Troubleshooting
  - Comparison with original FACT

## Architecture Highlights

### Feature Projection Approach (Why Not Re-extract?)

**Decision**: Use existing I3D features + learnable projection instead of re-extracting CLIP visual features

**Rationale**:
1. HAViD already provides pre-extracted I3D features (2048-dim)
2. Re-extraction would require:
   - Processing all videos again (days of computation)
   - Significant storage overhead
   - Access to raw video frames
3. Projection approach:
   - Uses existing features immediately
   - Learnable alignment (I3D → CLIP space)
   - More flexible (works with any pre-computed features)

### Dual Embedding Spaces

The model operates in two embedding spaces:

1. **CLIP Space (512-dim)**:
   - Visual features projected from I3D
   - Text embeddings from CLIP
   - Similarity computation happens here

2. **FACT Space (hid_dim)**:
   - Visual features for temporal modeling
   - Action features for cross-attention
   - Maintains FACT's temporal dependencies

### Contrastive Learning

Replaces classification with similarity-based prediction:

```python
# Original FACT
logits = linear_layer(features)  # (T, num_classes)
loss = cross_entropy(logits, labels)

# FACT_OpenVocab
visual_emb = project(features)  # (T, 512)
text_emb = clip_text(descriptions)  # (C, 512)
similarity = cosine_sim(visual_emb, text_emb)  # (T, C)
loss = cross_entropy(similarity / temperature, labels)
```

## Usage Examples

### 1. Training on HAViD

```bash
python -m src.train_openvocab \
    --cfg src/configs/openvocab_havid_view0_lh_pt.yaml \
    --set aux.gpu 0 aux.exp havid_openvocab
```

### 2. Standard Evaluation

```bash
python -m src.eval_openvocab \
    --cfg src/configs/openvocab_havid_view0_lh_pt.yaml \
    --ckpt log/havid_openvocab/ckpts/network.iter-50000.net \
    --set aux.gpu 0
```

### 3. Zero-Shot Evaluation

```bash
python -m src.eval_openvocab \
    --cfg src/configs/openvocab_havid_view0_lh_pt.yaml \
    --ckpt log/havid_openvocab/ckpts/network.iter-50000.net \
    --zero_shot \
    --unseen_actions "open_door,close_window,pour_water" \
    --set aux.gpu 0
```

## Key Design Decisions

### 1. Trainable vs Frozen CLIP

**Decision**: Make CLIP text encoder trainable

**Rationale**:
- Adapts CLIP to video action domain
- Video actions differ from image captions CLIP was trained on
- User requirement: "allow CLIP encoders to be trainable" (requirement 2.b)

### 2. Pre-computed vs Dynamic Text Embeddings

**Decision**: Support both

**Implementation**:
- Training: Pre-compute and cache text embeddings (efficiency)
- Zero-shot inference: Dynamically encode new descriptions (flexibility)

### 3. Temperature Parameter

**Decision**: Make temperature learnable

**Rationale**:
- Standard CLIP uses τ = 0.07
- Video actions may need different scaling
- Let model learn optimal value during training

### 4. Similarity Computation Location

**Decision**: Compute in CLIP space, not FACT space

**Rationale**:
- CLIP embeddings are trained to be comparable
- Better alignment for zero-shot generalization
- Maintains semantic structure from CLIP

## Testing Recommendations

### Unit Tests

1. **Model instantiation**: 
   ```python
   model = FACT_OpenVocab(cfg, visual_dim=2048, actions=descriptions)
   assert model.visual_projection is not None
   ```

2. **Forward pass shapes**:
   ```python
   output = model(seq_list, label_list)
   assert output[0]['pred'].shape == label_list[0].shape
   ```

3. **Similarity computation**:
   ```python
   sim = model.block_list[-1].frame_sim
   assert sim.shape[-1] == num_classes  # (T, 1, C)
   ```

### Integration Tests

1. **Training for 1 epoch** on small subset
2. **Standard evaluation** on test set
3. **Zero-shot evaluation** with 2-3 unseen actions

### Expected Behavior

- Loss should decrease during training
- Accuracy on seen classes: ≈ original FACT (±2%)
- Zero-shot accuracy: 10-40% depending on semantic similarity

## Dependencies

Minimum versions:
```
torch>=1.12.0
transformers>=4.30.0
yacs
numpy
scipy
wandb
tqdm
```

## Known Limitations

1. **Computational overhead**: Additional projection and CLIP text encoding
2. **Memory**: Caching text embeddings for all classes
3. **Zero-shot performance**: Depends heavily on:
   - Quality of text descriptions
   - Semantic similarity between seen/unseen classes
   - Training data diversity

## Future Work

1. **Multi-modal fusion**: Combine I3D + CLIP visual features
2. **Learnable prompts**: Optimize text templates end-to-end
3. **Cross-dataset zero-shot**: Train on Breakfast, test on HAViD
4. **Temporal grounding**: Use richer text descriptions with temporal info
5. **Few-shot adaptation**: Quick adaptation to new classes with few examples

## Verification Checklist

- ✅ Visual projection module implemented
- ✅ CLIP text encoder integrated
- ✅ FACT_OpenVocab main class complete
- ✅ All block variants (InputBlock_OV, UpdateBlock_OV, UpdateBlockTDU_OV)
- ✅ Contrastive loss functions
- ✅ MatchCriterion_OV with matching logic
- ✅ Dataset utilities with text formatting
- ✅ Configuration files updated
- ✅ Training script with multi-LR optimizer
- ✅ Evaluation script with zero-shot support
- ✅ Comprehensive documentation
- ✅ No linter errors

## Credits

Implementation based on:
- **FACT paper**: Lu & Elhamifar, CVPR 2024
- **CLIP paper**: Radford et al., ICML 2021
- **ActionCLIP**: Inspiration for video-text alignment approach

## Contact & Support

For issues or questions:
1. Check `src/OPENVOCAB_README.md` for detailed documentation
2. Refer to original FACT project for base model questions
3. CLIP documentation for text encoder questions

---

**Status**: ✅ **Implementation Complete and Ready for Testing**

All components have been implemented according to the plan. The codebase is ready for:
- Training experiments on HAViD/Breakfast datasets
- Standard evaluation on seen classes
- Zero-shot evaluation on unseen action descriptions



