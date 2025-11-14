# HAViD Text Processing Pipeline

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HAViD Dataset                                     │
│  ┌──────────────────┐              ┌────────────────────┐              │
│  │  Video Features  │              │   Ground Truth     │              │
│  │   (I3D, 2048)    │              │   (label strings)  │              │
│  └──────────────────┘              └────────────────────┘              │
│         ↓                                    ↓                           │
│    feature.npy                          transcript.txt                  │
│         ↓                                    ↓                           │
└─────────┼────────────────────────────────────┼───────────────────────────┘
          ↓                                    ↓
          │                          ┌─────────────────────┐
          │                          │  mapping.txt        │
          │                          │  ┌───────────────┐  │
          │                          │  │ sshc1dh → 0   │  │
          │                          │  │ pglg1   → 1   │  │
          │                          │  │ gsh     → 2   │  │
          │                          │  └───────────────┘  │
          │                          └─────────────────────┘
          │                                    ↓
          │                          ┌─────────────────────────────────────┐
          │                          │  HAViD Text Prompts                 │
          │                          │  (havid_text_prompts.py)            │
          │                          │                                     │
          │                          │  parse_havid_label()                │
          │                          │    ↓                                │
          │                          │  sshc1dh →                          │
          │                          │    verb: 'screws'                   │
          │                          │    object: 'hex screw'              │
          │                          │    target: 'cylinder plate hole 1'  │
          │                          │    tool: 'hex screwdriver'          │
          │                          │    ↓                                │
          │                          │  generate_action_prompt()           │
          │                          └─────────────────────────────────────┘
          │                                    ↓
          │                          "a person screws a hex screw into
          │                           cylinder plate hole 1 with a 
          │                           hex screwdriver"
          │                                    ↓
┌─────────┼────────────────────────────────────┼───────────────────────────┐
│         │                                    │                           │
│         │         Open-Vocabulary FACT Model                             │
│         │                                    │                           │
│         ↓                                    ↓                           │
│  ┌──────────────────┐           ┌────────────────────────┐             │
│  │  Visual Branch   │           │     Text Branch         │             │
│  ├──────────────────┤           ├────────────────────────┤             │
│  │ I3D Features     │           │ Text Descriptions      │             │
│  │ (2048, T)        │           │ (list of strings)      │             │
│  │      ↓           │           │      ↓                 │             │
│  │ Projection       │           │ CLIP Text Encoder      │             │
│  │ (2048 → 512)     │           │ (string → 512)         │             │
│  │      ↓           │           │      ↓                 │             │
│  │ Visual Features  │           │ Text Embeddings        │             │
│  │ (512, T)         │           │ (C, 512)               │             │
│  └──────────────────┘           └────────────────────────┘             │
│         ↓                                    ↓                           │
│         └────────────┬───────────────────────┘                           │
│                      ↓                                                   │
│         ┌─────────────────────────┐                                     │
│         │  Cosine Similarity      │                                     │
│         │  (visual × text)        │                                     │
│         │  → (C, T)               │                                     │
│         └─────────────────────────┘                                     │
│                      ↓                                                   │
│         ┌─────────────────────────┐                                     │
│         │  Contrastive Loss       │                                     │
│         │  (KL divergence)        │                                     │
│         └─────────────────────────┘                                     │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Process

### 1. Dataset Loading
```python
# Load HAViD dataset (existing FACT dataloader)
dataset, test_dataset = create_dataset(cfg)
# Returns:
#   - I3D features: np.array (2048, T) per video
#   - Labels: list of acronyms per frame ['sshc1dh', 'sshc1dh', 'gsh', ...]
```

### 2. Label Mapping
```python
# Load action mapping (existing FACT)
label2index, index2label = load_action_mapping('mapping.txt')
# Returns:
#   label2index: {'sshc1dh': 0, 'pglg1': 1, 'gsh': 2, ...}
#   index2label: {0: 'sshc1dh', 1: 'pglg1', 2: 'gsh', ...}
```

### 3. Text Generation (NEW)
```python
# Convert labels to natural language
from utils.havid_text_prompts import generate_action_prompt

action_descriptions = [
    generate_action_prompt(index2label[i]) 
    for i in range(num_classes)
]
# Returns:
#   [
#     "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver",
#     "a person places a large gear onto gear plate hole 1",
#     "a person grasps a hex screw",
#     ...
#   ]
```

### 4. Text Encoding (NEW)
```python
# Pre-compute CLIP text embeddings
import clip
model, preprocess = clip.load("ViT-B/32")

text_tokens = clip.tokenize(action_descriptions).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)  # (C, 512)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
```

### 5. Visual Projection (NEW)
```python
# In FACT_OpenVocab model
class VisualProjection(nn.Module):
    def __init__(self, in_dim=2048, out_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim * 2, out_dim)
        )
    
    def forward(self, x):
        # x: (B, 2048, T)
        x = x.permute(0, 2, 1)  # (B, T, 2048)
        x = self.proj(x)         # (B, T, 512)
        x = x.permute(0, 2, 1)  # (B, 512, T)
        return F.normalize(x, dim=1)  # L2 normalize
```

### 6. Similarity Computation (NEW)
```python
# In forward pass
visual_features = self.visual_projection(i3d_features)  # (B, 512, T)
text_features = self.text_features  # (C, 512) - pre-computed or dynamic

# Compute similarity
similarity = torch.einsum('bdt,cd->bct', visual_features, text_features)  # (B, C, T)
# similarity[b, c, t] = cosine similarity between frame t and class c
```

### 7. Loss Computation (NEW)
```python
# Contrastive loss (from ActionCLIP paper)
def contrastive_loss(similarity, targets):
    """
    similarity: (B, C, T) - logits
    targets: (B, T) - ground truth class indices
    """
    B, C, T = similarity.shape
    
    # Video-to-text
    v2t_logits = similarity / temperature  # (B, C, T)
    v2t_probs = F.softmax(v2t_logits, dim=1)  # (B, C, T)
    
    # Text-to-video
    t2v_logits = similarity.permute(1, 0, 2) / temperature  # (C, B, T)
    t2v_probs = F.softmax(t2v_logits, dim=1)  # (C, B, T)
    
    # Ground truth (one-hot)
    gt_v2t = F.one_hot(targets, num_classes=C).permute(0, 2, 1).float()  # (B, C, T)
    gt_t2v = gt_v2t.permute(1, 0, 2)  # (C, B, T)
    
    # KL divergence
    loss_v2t = F.kl_div(v2t_probs.log(), gt_v2t, reduction='batchmean')
    loss_t2v = F.kl_div(t2v_probs.log(), gt_t2v, reduction='batchmean')
    
    return (loss_v2t + loss_t2v) / 2
```

## Data Flow Example

### Input
```
Video: P01_part1.npy
Labels: P01_part1.txt
Content of P01_part1.txt:
  0-500: sshc1dh
  501-800: sshc1dh
  801-1200: gsh
  1201-1500: pglg1
  ...
```

### Processing
```
Frame 0-500: "sshc1dh"
  ↓ (parse)
  verb='screws', object='hex screw', target='cylinder plate hole 1', tool='hex screwdriver'
  ↓ (generate)
  "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver"
  ↓ (CLIP encode)
  [0.12, -0.34, 0.56, ...] (512-dim vector)
  ↓ (compare with visual)
  similarity = 0.89 (high - correct match!)

Frame 801-1200: "gsh"
  ↓ (parse)
  verb='grasps', object='hex screw', target=None, tool=None
  ↓ (generate)
  "a person grasps a hex screw"
  ↓ (CLIP encode)
  [0.45, 0.23, -0.12, ...] (512-dim vector)
  ↓ (compare with visual)
  similarity = 0.92 (high - correct match!)
```

### Training
```
Loss = KL_divergence(similarity_scores, ground_truth)
     + smooth_loss(predictions)
     + other_regularization
```

### Inference
```
Given: new video with I3D features
  ↓
Project to CLIP space (512-dim)
  ↓
Compute similarity with ALL class text embeddings
  ↓
argmax(similarity) → predicted class
  ↓
Map back to HAViD label: class_idx → "sshc1dh"
  ↓
Display: "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver"
```

## Key Components

### Files Involved

1. **Text Generation**
   - `havid_text_prompts.py` - Label parsing and text generation
   
2. **Dataset**
   - `dataset_OpenVocab.py` - Loads data and generates text descriptions
   
3. **Model**
   - `blocks_OpenVocab.py` - Visual projection, CLIP integration
   
4. **Loss**
   - `loss_OpenVocab.py` - Contrastive loss implementation
   
5. **Training**
   - `train_openvocab.py` - Training loop
   
6. **Config**
   - `openvocab_havid_*.yaml` - Configuration files

### Key Classes

```python
# Text generation
generate_action_prompt(label: str) -> str

# Dataset
create_dataset_openvocab(cfg) -> (dataset, test_dataset, text_embeddings, descriptions)

# Model
FACT_OpenVocab(cfg, num_classes, text_embeddings, action_descriptions)
  ├── visual_projection: Linear(2048 -> 512)
  ├── clip_text_encoder: CLIPTextEncoder (frozen or trainable)
  ├── forward() -> similarity_scores (B, C, T)

# Loss
ContrastiveLoss(similarity, targets) -> loss
```

## Benefits

1. **Semantic Understanding**: Text captures action meaning
2. **Zero-Shot Transfer**: Can recognize novel action combinations
3. **Compositional Learning**: Understands verb-object-target-tool structure
4. **Better Generalization**: CLIP's pre-training helps with unseen scenarios
5. **Interpretability**: Natural language predictions are human-readable

---

For complete implementation details, see:
- **Quick Reference**: `HAVID_QUICK_REFERENCE.md`
- **Full Documentation**: `src/utils/HAVID_TEXT_PROMPTS_README.md`
- **Implementation Summary**: `HAVID_TEXT_INTEGRATION_SUMMARY.md`

