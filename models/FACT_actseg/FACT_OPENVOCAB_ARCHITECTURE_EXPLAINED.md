# FACT Open Vocabulary Model: Complete Architecture Explanation

This document provides a detailed explanation of how the FACT Open Vocabulary model processes inputs through to the final contrastive loss.

---

## Table of Contents
1. [Overview](#overview)
2. [Input Stage](#input-stage)
3. [Visual Feature Projection](#visual-feature-projection)
4. [Text Embedding Processing](#text-embedding-processing)
5. [FACT Block Processing](#fact-block-processing)
6. [Similarity Computation](#similarity-computation)
7. [Loss Computation](#loss-computation)
8. [Complete Flow Diagram](#complete-flow-diagram)

---

## Overview

The FACT Open Vocabulary model adapts the original FACT architecture to work with CLIP embeddings, enabling zero-shot action segmentation. The key innovation is **projecting pre-computed I3D features into CLIP's embedding space** and using **cosine similarity with text embeddings** instead of classification logits.

**Key Components:**
- **VisualFeatureProjection**: Maps I3D features (2048D) → CLIP space (512D)
- **CLIPTextEncoder**: Encodes action descriptions into CLIP text embeddings (512D)
- **FACT Blocks**: Process temporal relationships (unchanged architecture)
- **Similarity Computation**: Replaces classification heads with cosine similarity
- **Contrastive Loss**: Replaces cross-entropy with contrastive learning

---

## Input Stage

### 1. Video Features (Pre-computed I3D)
```
Input: I3D features from video frames
Shape: (T, B, 2048)
  - T: Number of frames (temporal dimension)
  - B: Batch size (typically 1 for action segmentation)
  - 2048: I3D feature dimension
```

**Example for HAViD dataset:**
- Video with 1000 frames
- Each frame has a 2048-dimensional I3D feature vector
- Shape: `(1000, 1, 2048)`

### 2. Ground Truth Labels
```
Labels: Frame-level action class indices
Shape: (T,)
  - Each frame has a class label (0 to C-1, where C includes background)
```

### 3. Action Descriptions (Text)
```
Text: Natural language descriptions of actions
Example: ["A person is opening the refrigerator door",
          "A person is closing the refrigerator door",
          ...]
Number: C classes (e.g., 74 action classes + 1 background = 75 total)
```

---

## Visual Feature Projection

### Purpose
Convert pre-computed I3D features into CLIP-compatible embedding space without re-extracting video features.

### Architecture
```python
VisualFeatureProjection:
  Input: (T, B, 2048)  # I3D features
  ↓
  Linear(2048 → 1024)
  ↓
  ReLU
  ↓
  Dropout(0.1)
  ↓
  Linear(1024 → 512)
  ↓
  LayerNorm(512)
  ↓
  Output: (T, B, 512)  # CLIP-compatible features
```

### Mathematical Operation
```
For each frame feature x_t ∈ ℝ^2048:
  h = ReLU(W₁x_t + b₁)           # h ∈ ℝ^1024
  h = Dropout(h, p=0.1)
  z_t = W₂h + b₂                 # z_t ∈ ℝ^512
  z_t = LayerNorm(z_t)
  
Result: z = [z_1, z_2, ..., z_T] ∈ ℝ^(T×512)
```

### Key Points
- **Trainable**: These projection layers learn to align I3D features with CLIP space
- **Preserves temporal structure**: Processes each frame independently
- **Bidirectional alignment**: Learns to map I3D → CLIP during training

---

## Text Embedding Processing

### Two Modes of Operation

#### Mode 1: Pre-computed (Training)
Used during training for efficiency:

```python
# Done once before training starts
CLIPTextEncoder.encode_text(action_descriptions)
  Input: ["A person is opening...", "A person is closing...", ...]
  ↓
  CLIP Tokenizer → token_ids (max 77 tokens)
  ↓
  CLIP Text Encoder (Transformer)
  ↓
  CLIP Text Projection
  ↓
  Output: (C, 512)  # C text embeddings in CLIP space
```

**Stored as model buffer:**
```python
net.register_text_embeddings(text_embeddings)
# Shape: (75, 512) for 74 actions + 1 background
```

#### Mode 2: Dynamic (Inference/Zero-shot)
Used for zero-shot evaluation with new action classes:

```python
# At inference time for unseen actions
new_descriptions = ["A person is stirring soup", ...]
text_embeddings = net.clip_text_encoder.encode_text(new_descriptions)
# Shape: (N_new, 512)
```

### Text Embedding Properties
- **Normalized**: L2-normalized for cosine similarity
- **Trainable**: Text encoder can be fine-tuned (per user requirement)
- **Shared space**: Same 512D CLIP embedding space as visual features

---

## FACT Block Processing

The FACT blocks maintain the original temporal modeling architecture but work with CLIP-space features.

### Block Structure

```
FACT_OpenVocab has N blocks (e.g., 3 blocks):
  - Block 0: InputBlock_OV
  - Block 1: UpdateBlock_OV (or UpdateBlockTDU_OV)
  - Block 2: UpdateBlock_OV (or UpdateBlockTDU_OV)
```

### InputBlock_OV (First Block)

**Input:**
```
visual_features: (T, B, 512)  # Projected I3D features in CLIP space
```

**Processing:**

1. **Frame Branch (F-branch):**
```python
frame_features = fbranch(visual_features)
# Shape: (T, B, hid_dim)
# hid_dim = 256 (configurable)
```

2. **Action Branch (A-branch):**
```python
# Learnable action queries
action_queries = nn.Parameter(torch.randn(M, B, hid_dim))
# M = number of action tokens (e.g., 30)

action_features = abranch(action_queries)
# Shape: (M, B, hid_dim)
```

3. **Cross-Attention (Frame ↔ Action):**
```python
# Action queries attend to frame features
a2f_attn = cross_attention(
    query=action_features,    # (M, B, hid_dim)
    key=frame_features,       # (T, B, hid_dim)
    value=frame_features      # (T, B, hid_dim)
)
# Output: (M, B, T) - attention weights showing which frames each action attends to

# Frame features attend to action features
f2a_attn = cross_attention(
    query=frame_features,     # (T, B, hid_dim)
    key=action_features,      # (M, B, hid_dim)
    value=action_features     # (M, B, hid_dim)
)
# Output: (T, B, M) - attention weights showing which actions each frame attends to
```

4. **Similarity Computation:**
```python
# Frame similarities (direct comparison with text embeddings)
frame_sim = compute_similarity(
    frame_features,      # (T, B, hid_dim)
    text_embeddings      # (C, 512)
)
# Shape: (T, B, C)

# Action similarities
action_sim = compute_similarity(
    action_features,     # (M, B, hid_dim)
    text_embeddings      # (C, 512)
)
# Shape: (M, B, C)
```

**Output:**
```python
{
    'frame_sim': (T, B, C),      # Frame-to-text similarities
    'action_sim': (M, B, C),     # Action-to-text similarities
    'a2f_attn': (M, B, T),       # Action-to-frame attention
    'f2a_attn': (T, B, M),       # Frame-to-action attention
    'frame_feat': (T, B, hid_dim),
    'action_feat': (M, B, hid_dim)
}
```

### UpdateBlock_OV (Subsequent Blocks)

**Input:**
```
Previous block's outputs + original visual_features
```

**Processing:**

1. **Refine Frame Features:**
```python
# Use previous action information to refine frames
frame_features_new = fbranch(
    visual_features,           # Original input
    prev_action_features,      # From previous block
    prev_f2a_attn             # Previous attention
)
```

2. **Refine Action Features:**
```python
# Use previous frame information to refine actions
action_features_new = abranch(
    prev_action_features,      # From previous block
    prev_frame_features,       # From previous block
    prev_a2f_attn             # Previous attention
)
```

3. **Update Cross-Attention:**
```python
# Recompute attention with refined features
a2f_attn_new = cross_attention(action_features_new, frame_features_new)
f2a_attn_new = cross_attention(frame_features_new, action_features_new)
```

4. **Recompute Similarities:**
```python
frame_sim_new = compute_similarity(frame_features_new, text_embeddings)
action_sim_new = compute_similarity(action_features_new, text_embeddings)
```

**Key Insight:** Each block refines the frame and action representations by incorporating information from the other branch, improving temporal coherence.

---

## Similarity Computation

This is where the open-vocabulary magic happens!

### compute_similarity Function

```python
def compute_similarity(features, text_embeddings, temperature=0.07):
    """
    Compute cosine similarity between visual features and text embeddings
    
    Args:
        features: (T, B, hid_dim) or (M, B, hid_dim)
        text_embeddings: (C, 512)
        temperature: Scaling factor for similarities
    
    Returns:
        similarities: (T, B, C) or (M, B, C)
    """
```

### Step-by-Step Process

1. **Project to CLIP dimension:**
```python
# If features are in hid_dim (256), project to CLIP dim (512)
if features.shape[-1] != text_embeddings.shape[-1]:
    features = projection_layer(features)
# Now features: (T, B, 512) or (M, B, 512)
```

2. **Normalize features:**
```python
features_norm = F.normalize(features, dim=-1)
text_norm = F.normalize(text_embeddings, dim=-1)
# Both have unit L2 norm
```

3. **Compute cosine similarity:**
```python
# Matrix multiplication gives cosine similarity
similarities = torch.matmul(features_norm, text_norm.T)
# Shape: (T, B, C) or (M, B, C)
# Values in range [-1, 1]
```

4. **Temperature scaling:**
```python
similarities = similarities / temperature
# temperature = 0.07 (makes similarities sharper)
# Higher similarity → higher score
```

### Mathematical Formula

For a frame feature **f** and text embedding **t**:

```
similarity(f, t) = (f · t) / (||f|| ||t|| τ)
                 = cos(θ) / τ

where:
  · = dot product
  ||·|| = L2 norm
  θ = angle between vectors
  τ = temperature (0.07)
```

### Interpretation

- **High similarity (> 0)**: Visual feature aligns well with text description
- **Low similarity (< 0)**: Visual feature doesn't match text description
- **Temperature effect**: Dividing by 0.07 amplifies differences (0.7 → 10.0)

---

## Loss Computation

The model uses **contrastive losses** instead of cross-entropy classification losses.

### Loss Components

#### 1. Frame-Level Contrastive Loss

**Purpose:** Align frame features with correct action text embeddings

```python
def contrastive_frame_loss(frame_sim, labels):
    """
    Args:
        frame_sim: (T, C) - similarity scores
        labels: (T,) - ground truth class indices
    """
    # Treat similarities as logits
    loss = F.cross_entropy(frame_sim, labels)
    return loss
```

**What it does:**
- For each frame, we have similarities to all C classes
- Ground truth label tells us which class is correct
- Cross-entropy pushes the correct class similarity higher
- Pushes incorrect class similarities lower

**Mathematical form:**
```
L_frame = -log(exp(sim[t, y_t]) / Σ_c exp(sim[t, c]))

where:
  sim[t, c] = similarity of frame t to class c
  y_t = ground truth class for frame t
```

#### 2. Action Token Contrastive Loss

**Purpose:** Align action tokens with their matched segments

```python
def contrastive_action_token_loss(action_sim, match, transcript):
    """
    Args:
        action_sim: (M, C) - action token similarities
        match: (action_indices, segment_indices) - bipartite matching
        transcript: (S,) - segment-level labels
    """
    aind, sind = match  # Matched pairs
    M, C = action_sim.shape
    
    # Create target: null class for unmatched, transcript label for matched
    target = torch.zeros(M, dtype=torch.long) + (C - 1)  # All null initially
    target[aind] = transcript[sind]  # Matched tokens get segment labels
    
    loss = F.cross_entropy(action_sim, target)
    return loss
```

**What it does:**
- Uses Hungarian matching to pair action tokens with ground truth segments
- Matched tokens: pushed toward their segment's action class
- Unmatched tokens: pushed toward null/background class

**Example:**
```
Video has 3 ground truth segments: [open_fridge, take_item, close_fridge]
Model has 30 action tokens

Hungarian matching finds best alignment:
  token_5 ↔ segment_0 (open_fridge)
  token_12 ↔ segment_1 (take_item)
  token_23 ↔ segment_2 (close_fridge)
  tokens [0,1,2,3,4,6,...,22,24,...,29] ↔ null (unmatched)

Loss encourages:
  - token_5 similarity to "open_fridge" text → high
  - token_12 similarity to "take_item" text → high
  - token_23 similarity to "close_fridge" text → high
  - Other tokens similarity to null class → high
```

#### 3. Cross-Attention Losses (Temporal Alignment)

**Purpose:** Ensure action tokens attend to correct temporal regions

These losses are **kept from the original FACT** as they don't involve classification:

```python
def cross_attn_loss(a2f_attn, match, seg_mask):
    """
    Args:
        a2f_attn: (M, T) - action-to-frame attention weights
        match: Bipartite matching
        seg_mask: (S, T) - binary mask for each segment
    """
    # For matched action tokens, encourage attention to their segment's frames
    aind, sind = match
    matched_attn = a2f_attn[aind]  # (|match|, T)
    target_masks = seg_mask[sind]   # (|match|, T)
    
    # Binary cross-entropy: attention should match segment mask
    loss = F.binary_cross_entropy_with_logits(matched_attn, target_masks)
    return loss
```

**What it does:**
- Ensures action tokens attend to the correct temporal windows
- Example: If segment_0 spans frames [0:100], token_5 should attend to frames [0:100]

#### 4. Total Loss

```python
total_loss = (
    λ_frame * frame_loss +           # Frame-text alignment
    λ_action * action_token_loss +   # Action-text alignment
    λ_attn * cross_attn_loss +       # Temporal alignment
    λ_smooth * smoothness_loss       # Temporal smoothness
)
```

**Default weights (from config):**
- λ_frame = 1.0
- λ_action = 1.0
- λ_attn = 1.0
- λ_smooth = 0.15

---

## Complete Flow Diagram

### Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT STAGE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Video I3D Features          Action Descriptions    GT Labels   │
│  (T, 1, 2048)               ["open fridge", ...]    (T,)        │
│       │                            │                  │          │
│       │                            │                  │          │
│       ▼                            ▼                  │          │
│  ┌──────────────┐          ┌──────────────┐          │          │
│  │   Visual     │          │  CLIP Text   │          │          │
│  │  Projection  │          │   Encoder    │          │          │
│  │ 2048→512     │          │  (trainable) │          │          │
│  └──────────────┘          └──────────────┘          │          │
│       │                            │                  │          │
│       ▼                            ▼                  │          │
│  Visual Features           Text Embeddings            │          │
│  (T, 1, 512)               (C, 512)                   │          │
│       │                            │                  │          │
└───────┼────────────────────────────┼──────────────────┼──────────┘
        │                            │                  │
        │                            │                  │
┌───────┼────────────────────────────┼──────────────────┼──────────┐
│       │         FACT BLOCK 0 (InputBlock_OV)          │          │
├───────┼────────────────────────────┼──────────────────┼──────────┤
│       │                            │                  │          │
│       ▼                            │                  │          │
│  ┌─────────┐                       │                  │          │
│  │ F-branch│────────┐              │                  │          │
│  └─────────┘        │              │                  │          │
│       │             │              │                  │          │
│       │    ┌────────▼────────┐     │                  │          │
│       │    │  Cross-Attention │    │                  │          │
│       │    │   (Frame ↔ Act)  │    │                  │          │
│       │    └────────┬────────┘     │                  │          │
│       │             │              │                  │          │
│  ┌─────────┐        │              │                  │          │
│  │ A-branch│────────┘              │                  │          │
│  └─────────┘                       │                  │          │
│       │                            │                  │          │
│       ├──────────┬─────────────────┘                  │          │
│       │          │                                    │          │
│       ▼          ▼                                    │          │
│  ┌─────────┐ ┌─────────┐                             │          │
│  │ Frame   │ │ Action  │                             │          │
│  │Features │ │Features │                             │          │
│  │(T,1,256)│ │(M,1,256)│                             │          │
│  └────┬────┘ └────┬────┘                             │          │
│       │           │                                   │          │
│       ▼           ▼                                   │          │
│  ┌─────────────────────┐                             │          │
│  │ Compute Similarity  │◄─── Text Embeddings         │          │
│  │  (cosine + temp)    │     (C, 512)                │          │
│  └─────────────────────┘                             │          │
│       │           │                                   │          │
│       ▼           ▼                                   │          │
│  frame_sim    action_sim                             │          │
│  (T, 1, C)    (M, 1, C)                              │          │
│       │           │                                   │          │
└───────┼───────────┼───────────────────────────────────┼──────────┘
        │           │                                   │
        │           │                                   │
┌───────┼───────────┼───────────────────────────────────┼──────────┐
│       │      FACT BLOCKS 1, 2, ... (UpdateBlock_OV)  │          │
├───────┼───────────┼───────────────────────────────────┼──────────┤
│       │           │                                   │          │
│       │  [Iterative refinement of features]          │          │
│       │  [Recompute cross-attention]                 │          │
│       │  [Recompute similarities]                    │          │
│       │           │                                   │          │
│       ▼           ▼                                   │          │
│  frame_sim    action_sim                             │          │
│  (refined)    (refined)                              │          │
│       │           │                                   │          │
└───────┼───────────┼───────────────────────────────────┼──────────┘
        │           │                                   │
        │           │                                   │
┌───────┼───────────┼───────────────────────────────────┼──────────┐
│       │           │         LOSS COMPUTATION          │          │
├───────┼───────────┼───────────────────────────────────┼──────────┤
│       │           │                                   │          │
│       ▼           │                                   ▼          │
│  ┌─────────────┐  │                          ┌──────────────┐   │
│  │  Frame Loss │  │                          │ Hungarian    │   │
│  │ (contrastive│  │                          │ Matching     │   │
│  │  per frame) │  │                          │ (action↔seg) │   │
│  └─────────────┘  │                          └──────┬───────┘   │
│                   │                                 │           │
│                   ▼                                 ▼           │
│            ┌──────────────┐                ┌──────────────┐    │
│            │ Action Token │                │Cross-Attention│    │
│            │     Loss     │                │     Loss      │    │
│            │ (contrastive)│                │  (temporal)   │    │
│            └──────────────┘                └──────────────┘    │
│                   │                                 │           │
│                   └────────────┬────────────────────┘           │
│                                ▼                                │
│                         ┌─────────────┐                         │
│                         │  Total Loss │                         │
│                         └─────────────┘                         │
│                                │                                │
└────────────────────────────────┼────────────────────────────────┘
                                 │
                                 ▼
                          Backpropagation
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Update Parameters:    │
                    │  - Visual Projection   │
                    │  - CLIP Text Encoder   │
                    │  - FACT Blocks         │
                    └────────────────────────┘
```

### Inference Flow (Zero-Shot)

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEW ACTION CLASSES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  New Descriptions: ["stir soup", "pour water", ...]             │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │  CLIP Text   │                                               │
│  │   Encoder    │                                               │
│  │  (frozen)    │                                               │
│  └──────────────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  New Text Embeddings (C_new, 512)                               │
│         │                                                        │
└─────────┼───────────────────────────────────────────────────────┘
          │
          ▼
    [Same FACT processing as training]
          │
          ▼
    Similarities with new text embeddings
          │
          ▼
    Predictions for unseen actions!
```

---

## Key Insights

### 1. **Why Feature Projection Works**
- I3D features already capture visual semantics
- Projection learns to align these semantics with CLIP's text-image space
- More efficient than re-extracting CLIP visual features from raw video

### 2. **Contrastive Learning Advantage**
- No fixed classification head → open vocabulary
- Text embeddings can be swapped at inference time
- Model learns general visual-semantic alignment

### 3. **FACT's Temporal Modeling**
- Frame branch: Direct frame-level predictions
- Action branch: Segment-level reasoning via learnable queries
- Cross-attention: Links frames to actions temporally
- **All preserved in open-vocabulary version!**

### 4. **Temperature Scaling**
- τ = 0.07 is crucial for contrastive learning
- Makes similarity distributions sharper
- Helps model distinguish between similar actions

### 5. **Training Dynamics**
- Visual projection learns I3D → CLIP alignment
- CLIP text encoder fine-tunes for action descriptions
- FACT blocks learn temporal patterns
- All trained end-to-end with contrastive losses

---

## Dimensions Summary

| Component | Shape | Description |
|-----------|-------|-------------|
| **Input I3D** | (T, 1, 2048) | Pre-computed video features |
| **Projected Visual** | (T, 1, 512) | CLIP-compatible visual features |
| **Text Embeddings** | (C, 512) | CLIP text embeddings for actions |
| **Frame Features** | (T, 1, 256) | FACT frame branch output |
| **Action Features** | (M, 1, 256) | FACT action branch output (M tokens) |
| **Frame Similarities** | (T, 1, C) | Frame-to-text cosine similarities |
| **Action Similarities** | (M, 1, C) | Action-to-text cosine similarities |
| **A2F Attention** | (M, 1, T) | Action-to-frame attention weights |
| **F2A Attention** | (T, 1, M) | Frame-to-action attention weights |

**Typical values:**
- T = 1000-3000 (video frames)
- C = 75 (74 actions + 1 background for HAViD)
- M = 30 (learnable action tokens)
- hid_dim = 256 (FACT hidden dimension)

---

## Comparison: Original FACT vs Open Vocabulary FACT

| Aspect | Original FACT | Open Vocabulary FACT |
|--------|---------------|----------------------|
| **Input** | I3D features (2048D) | I3D features (2048D) |
| **Feature Processing** | Direct use | Project to CLIP space (512D) |
| **Action Representation** | Fixed classification head | CLIP text embeddings |
| **Prediction** | Softmax over logits | Cosine similarity with text |
| **Loss** | Cross-entropy | Contrastive (cross-entropy on similarities) |
| **Vocabulary** | Closed (fixed classes) | Open (can add new classes) |
| **Zero-shot** | ❌ No | ✅ Yes |
| **Temporal Modeling** | Frame-Action Cross-Attention | **Same** (preserved) |

---

## Example Walkthrough

Let's trace a single frame through the model:

### Frame 500 of a video showing "opening refrigerator"

**1. Input:**
```
I3D feature: x₅₀₀ ∈ ℝ^2048
```

**2. Visual Projection:**
```
z₅₀₀ = VisualProjection(x₅₀₀) ∈ ℝ^512
```

**3. FACT Processing:**
```
f₅₀₀ = FBranch(z₅₀₀) ∈ ℝ^256  (frame feature)
```

**4. Similarity Computation:**
```
For each action class c:
  sim₅₀₀,c = cos(f₅₀₀, text_emb_c) / 0.07
  
Example similarities:
  sim₅₀₀,open_fridge = 8.5   (high - correct action!)
  sim₅₀₀,close_fridge = 3.2  (medium - related)
  sim₅₀₀,stir_soup = -1.4    (low - unrelated)
  sim₅₀₀,null = -2.1         (low - not background)
```

**5. Loss (if ground truth is "open_fridge"):**
```
L = -log(exp(8.5) / (exp(8.5) + exp(3.2) + exp(-1.4) + ... ))
  = -log(0.95)  (high probability to correct class)
  ≈ 0.05        (low loss - good prediction!)
```

**6. Backpropagation:**
```
Gradients flow to:
  - Visual projection (improve I3D→CLIP alignment)
  - CLIP text encoder (refine "open_fridge" embedding)
  - FACT frame branch (better temporal features)
```

---

## Conclusion

The FACT Open Vocabulary model elegantly extends FACT's temporal modeling to open-vocabulary scenarios by:

1. **Projecting** existing I3D features to CLIP space (efficient!)
2. **Replacing** classification heads with cosine similarity
3. **Using** contrastive losses for visual-semantic alignment
4. **Preserving** FACT's powerful frame-action cross-attention mechanism

This enables zero-shot action segmentation while maintaining FACT's state-of-the-art temporal modeling capabilities!

