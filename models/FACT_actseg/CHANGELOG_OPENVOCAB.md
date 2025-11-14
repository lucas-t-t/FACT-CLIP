# Changelog: Open-Vocabulary FACT Implementation

## [1.0.0] - 2025-11-03

### Added - Core Model Components

#### `src/models/blocks_OpenVocab.py`
- **CLIPTextEncoder** class
  - Wraps CLIP text encoder from transformers
  - Trainable text model and projection layers
  - Returns 512-dim embeddings in CLIP space
  
- **VisualFeatureProjection** class
  - Multi-layer projection: I3D (2048-dim) → CLIP space (512-dim)
  - Architecture: Linear(2048, 1024) → ReLU → Dropout → Linear(1024, 512) → LayerNorm
  - Xavier initialization for weight preservation

- **FACT_OpenVocab** class
  - Main model integrating visual projection and CLIP text encoder
  - Dual embedding spaces: CLIP (512) for similarity, FACT (hid_dim) for temporal modeling
  - Text embedding caching for training efficiency
  - Learnable temperature parameter (initialized to 0.07)
  - Support for both pre-computed and dynamic text embeddings

- **Block_OV** base class
  - compute_similarity(): Cosine similarity in CLIP space
  - Inherits frame/action branch creation from original Block
  - Modified eval methods using similarity scores

- **InputBlock_OV** class
  - Frame and action branches in FACT space
  - Projection layers: FACT features → CLIP space (512-dim)
  - Similarity computation for loss and evaluation
  
- **UpdateBlock_OV** class
  - Cross-attention between frame and action branches
  - Projection to CLIP space for similarity computation
  - Stores attention maps for loss computation

- **UpdateBlockTDU_OV** class
  - Temporal downsampling/upsampling support
  - Segment-level processing with GRU
  - Projection layers for frames, actions, and segments

### Added - Loss Functions

#### `src/models/loss_OpenVocab.py`
- **contrastive_frame_loss()**
  - Frame-level contrastive loss
  - Treats similarity scores as logits for cross-entropy

- **contrastive_action_token_loss()**
  - Action token contrastive loss
  - Handles matched and unmatched tokens (null class)

- **MatchCriterion_OV** class
  - Inherits matching logic from MatchCriterion
  - Overrides frame_loss() and action_token_loss() with contrastive versions
  - Overrides frame_loss_tdu() for temporal downsampling
  - Reuses cross-attention losses (temporal alignment)

### Added - Dataset Utilities

#### `src/utils/dataset_OpenVocab.py`
- **format_action_description()**
  - Converts class names to natural language
  - Supports prompt engineering: "a video of a person {action}"
  - Handles HAViD codes, breakfast actions, etc.

- **precompute_text_embeddings()**
  - Pre-computes CLIP text embeddings for efficiency
  - Caches embeddings (C, 512) for all classes
  - Optional: can skip for dynamic encoding only

- **get_mapping_file_path()**
  - Helper to get mapping.txt path for different datasets
  - Supports: breakfast, gtea, ego, epic, havid variants

- **create_dataset_openvocab()**
  - Extended dataset creation with text descriptions
  - Generates action descriptions for all classes
  - Pre-computes text embeddings if configured
  - Returns: dataset, test_dataset, text_embeddings, action_descriptions

### Added - Configuration

#### `src/configs/default.py`
- New CLIP configuration section:
  ```python
  _C.CLIP = CLIP = CN()
  CLIP.model_name = "openai/clip-vit-b-32"
  CLIP.text_trainable = True
  CLIP.temp = 0.07
  CLIP.precompute_text = True
  CLIP.use_prompt = True
  CLIP.projection_hidden_dim = 1024
  CLIP.projection_dropout = 0.1
  ```

#### `src/configs/openvocab_havid_view0_lh_pt.yaml`
- Example configuration for HAViD dataset
- Extends standard HAViD config with CLIP settings
- WandB project: "FACT-OpenVocab"
- Batch size: 2 (for HAViD's longer sequences)

### Added - Training Script

#### `src/train_openvocab.py`
- Imports open-vocabulary components
- Loads dataset with text descriptions
- Creates FACT_OpenVocab model
- Registers pre-computed text embeddings
- **Multi-component optimizer**:
  - CLIP text encoder: lr × 0.1
  - Visual projection: lr × 1.0
  - FACT blocks: lr × 1.0
  - Temperature parameter: lr × 1.0
- Training loop with contrastive loss
- Supports holdout/zero-shot training mode
- Enhanced logging for open-vocabulary metrics

### Added - Evaluation Script

#### `src/eval_openvocab.py`
- **evaluate_standard()**:
  - Standard evaluation on seen classes
  - Uses cached text embeddings

- **evaluate_zero_shot()**:
  - Zero-shot evaluation on unseen classes
  - Dynamically encodes new action descriptions
  - Disables embedding cache for fresh encoding

- Command-line arguments:
  - `--zero_shot`: Enable zero-shot mode
  - `--unseen_actions`: Comma-separated action names
  - `--unseen_actions_file`: Path to file with action names

### Added - Documentation

#### `src/OPENVOCAB_README.md`
- Comprehensive documentation:
  - Architecture overview with diagrams
  - Installation instructions
  - Usage examples (training, evaluation, zero-shot)
  - Configuration options explained
  - Text description formatting
  - Troubleshooting guide
  - Performance expectations
  - Comparison with original FACT
  - Citation information

#### `OPENVOCAB_IMPLEMENTATION_SUMMARY.md`
- Implementation summary:
  - Complete file listing
  - Architecture highlights
  - Design decisions rationale
  - Usage examples
  - Testing recommendations
  - Known limitations
  - Future work suggestions
  - Verification checklist

#### `QUICKSTART_OPENVOCAB.md`
- Quick start guide:
  - 5-minute getting started
  - Step-by-step instructions
  - Common workflows
  - Troubleshooting tips
  - Performance benchmarks
  - Quick reference commands

## Design Decisions

### Why Feature Projection (Option 2)?
- **Rationale**: HAViD provides pre-extracted I3D features (2048-dim)
- **Benefits**:
  - No re-extraction needed (saves days of computation)
  - Works with existing features immediately
  - More flexible (any pre-computed features)
  - Learnable I3D → CLIP alignment

### Why Trainable CLIP Text Encoder?
- **Rationale**: Video actions differ from image captions
- **Benefits**:
  - Adapts CLIP to video action domain
  - Better alignment for action descriptions
  - User requirement: "allow CLIP encoders to be trainable"

### Why Two Embedding Spaces?
- **CLIP Space (512-dim)**:
  - For similarity computation
  - Maintains CLIP's semantic structure
  - Better zero-shot generalization

- **FACT Space (hid_dim)**:
  - For temporal modeling
  - Maintains FACT's architectural strengths
  - Preserves frame-action cross-attention

### Why Different Learning Rates?
- **CLIP text encoder** (lr × 0.1): Pre-trained, needs gentle adaptation
- **Visual projection** (lr × 1.0): Learning from scratch
- **FACT blocks** (lr × 1.0): Learning from scratch

## Dependencies Added

```
transformers>=4.30.0
```

All other dependencies remain the same as original FACT.

## Backward Compatibility

- ✅ Original FACT model unchanged
- ✅ Original training/evaluation scripts unchanged
- ✅ Original configs unchanged
- ✅ New components isolated in separate files

## File Structure

```
models/FACT_actseg/
├── src/
│   ├── models/
│   │   ├── blocks.py                      # Original (unchanged)
│   │   ├── blocks_SepVerbNoun.py          # Original (unchanged)
│   │   ├── blocks_OpenVocab.py            # NEW
│   │   ├── loss.py                        # Original (unchanged)
│   │   ├── loss_OpenVocab.py              # NEW
│   │   └── basic.py                       # Original (unchanged)
│   ├── utils/
│   │   ├── dataset.py                     # Original (unchanged)
│   │   └── dataset_OpenVocab.py           # NEW
│   ├── configs/
│   │   ├── default.py                     # MODIFIED (added CLIP section)
│   │   └── openvocab_havid_view0_lh_pt.yaml  # NEW
│   ├── train.py                           # Original (unchanged)
│   ├── train_openvocab.py                 # NEW
│   ├── eval.py                            # Original (unchanged)
│   ├── eval_openvocab.py                  # NEW
│   └── OPENVOCAB_README.md                # NEW
├── OPENVOCAB_IMPLEMENTATION_SUMMARY.md    # NEW
├── QUICKSTART_OPENVOCAB.md                # NEW
└── CHANGELOG_OPENVOCAB.md                 # NEW (this file)
```

## Testing Status

- ✅ No linter errors in new files
- ✅ Model instantiation tested
- ⏳ Training pending (requires dataset)
- ⏳ Standard evaluation pending
- ⏳ Zero-shot evaluation pending

## Known Issues

None at implementation time. Issues will be tracked as they arise during testing.

## Future Enhancements

1. **Multi-modal fusion**: Combine I3D + CLIP visual features
2. **Learnable prompts**: Optimize text templates end-to-end
3. **Cross-dataset zero-shot**: Train on one dataset, test on another
4. **Temporal text grounding**: Use richer action descriptions
5. **Few-shot adaptation**: Quick adaptation with few examples
6. **Uncertainty estimation**: Confidence scores for predictions
7. **Attention visualization**: Visualize visual-text alignment
8. **Action proposal**: Generate action descriptions from videos

## Credits

- **Implementation**: Based on FACT architecture by Lu & Elhamifar (CVPR 2024)
- **Inspiration**: ActionCLIP (video-text alignment), CLIP (multimodal learning)
- **Date**: November 3, 2025
- **Status**: ✅ Complete and ready for testing

---

For questions or contributions, please refer to the project documentation.



