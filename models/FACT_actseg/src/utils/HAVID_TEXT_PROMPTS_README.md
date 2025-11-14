# HAViD Text Prompts for Open-Vocabulary FACT

This document explains how HAViD's compact action labels are converted to natural language descriptions for CLIP-based open-vocabulary temporal action segmentation.

## Overview

HAViD (Human Assembly Video Dataset) uses a structured labeling system where each action is encoded as a compact string combining:
- **Verb** (1 character): The action type (e.g., 's' = screw, 'g' = grasp)
- **Manipulated Object** (2 characters): The object being manipulated (e.g., 'sh' = hex screw)
- **Target Object** (2 characters): The location/target (e.g., 'c1' = cylinder plate hole 1)
- **Tool** (2 characters, optional): The tool used (e.g., 'dh' = hex screwdriver)

Example: `sshc1dh` → "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver"

## Files

### Core Implementation
- **`havid_text_prompts.py`**: Main implementation of label parsing and text generation
- **`test_havid_prompts.py`**: Comprehensive test suite
- **`demo_havid_integration.py`**: Demo showing integration with FACT model
- **`dataset_OpenVocab.py`**: Dataset utilities that use HAViD prompts

### Integration with FACT
The text prompts are used in:
- **`blocks_OpenVocab.py`**: CLIP text encoder integration
- **`loss_OpenVocab.py`**: Contrastive loss with text embeddings
- **`train_openvocab.py`**: Training script

## Label Structure

### Format Options

1. **Full specification** (7 characters): `[verb][object][target][tool]`
   - Example: `sshc1dh` → screws hex screw into cylinder plate hole 1 with hex screwdriver

2. **No tool** (5 characters): `[verb][object][target]`
   - Example: `ishc1` → inserts hex screw into cylinder plate hole 1

3. **No target** (3 characters): `[verb][object]`
   - Example: `gsh` → grasps a hex screw

4. **Verb only** (1 character): `[verb]`
   - Example: `s` → screws

5. **Noise labels**: `null`, `w` (wrong)

### Vocabulary

#### Action Verbs (10 total)
| Code | Verb | Preposition | Example |
|------|------|-------------|---------|
| `a` | approaches | to | approaches to cylinder base |
| `d` | disassembles | from | disassembles from cylinder plate |
| `g` | grasps | (default: to) | grasps a hex screw |
| `h` | holds | (default: to) | holds a gear |
| `i` | inserts | into | inserts into hole 1 |
| `l` | slides | onto | slides onto plate |
| `m` | moves | to | moves to hole 2 |
| `p` | places | onto | places onto gear plate |
| `r` | rotates | on | rotates on shaft |
| `s` | screws | into | screws into hole 3 |

#### Objects (~50+ types)
**Screws & Fasteners:**
- `sh`: hex screw
- `sp`: phillips screw
- `sb`: bolt
- `nt`: nut

**Gears:**
- `gl`: large gear
- `gs`: small gear
- `gw`: worm gear
- `g1-g3`: gear plate holes

**Cylinder Parts:**
- `cb`: cylinder base
- `cc`: cylinder cap
- `ck`: cylinder bracket
- `cs`: cylinder subassembly
- `c1-c4`: cylinder plate holes

**Tools:**
- `dh`: hex screwdriver
- `dp`: phillips screwdriver
- `wn`: nut wrench
- `ws`: shaft wrench

**Other:**
- `ba`: ball, `bs`: ball seat, `bx`: box
- `ft`: gear shaft, `hd`: dial, `hw`: hand-wheel
- `ib`: bar, `ir`: rod, `lb`: linear bearing
- `pl`: large spacer, `ps`: small spacer
- `n1-n6`: general plate holes/components

## Usage

### Basic Usage

```python
from havid_text_prompts import generate_action_prompt, parse_havid_label

# Generate text description
label = "sshc1dh"
prompt = generate_action_prompt(label)
# Returns: "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver"

# Parse label components
parsed = parse_havid_label(label)
# Returns: {
#   'verb': 'screws',
#   'manipulated_object': 'hex screw',
#   'target_object': 'cylinder plate hole 1',
#   'tool': 'hex screwdriver'
# }
```

### Integration with Dataset

```python
from dataset_OpenVocab import create_dataset_openvocab

# Load dataset with text descriptions
dataset, test_dataset, text_embeddings, action_descriptions = create_dataset_openvocab(cfg)

# action_descriptions will be automatically generated using HAViD prompts
# Example output:
# ['a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver',
#  'a person places a large gear onto gear plate hole 1',
#  'a person grasps a hex screw',
#  ...]
```

### Custom Templates

```python
# Use a custom template
template = "video showing {verb} {manipulated_object} {prep} {target_object}"
prompt = generate_action_prompt("sshc1", template=template)
```

## Implementation Details

### Preposition Handling

Each verb has an associated preposition for more natural English:
- `inserts` → "into"
- `places` → "onto"
- `screws` → "into"
- `moves` → "to"
- `grasps` → "" (no preposition, defaults to "to" if target exists)

### Verb Conjugation

All verbs are conjugated in third-person singular present tense:
- `screw` → `screws`
- `grasp` → `grasps`
- `insert` → `inserts`

This matches the template "a person [verb]s..."

### Dataset Detection

The system automatically detects HAViD labels and applies the appropriate formatting:

```python
from dataset_OpenVocab import format_action_description

# HAViD label - uses structured parsing
desc = format_action_description("sshc1dh", dataset_name="havid_view0_lh_pt")
# Returns: "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver"

# Non-HAViD label - uses simple formatting
desc = format_action_description("crack_egg", dataset_name="breakfast")
# Returns: "a video of a person crack egg"
```

## Testing

Run the comprehensive test suite:

```bash
cd models/FACT_actseg/src/utils
python test_havid_prompts.py
```

Run the integration demo:

```bash
python demo_havid_integration.py
```

## Examples

### Common Action Patterns

```python
# Assembly actions
"sshc1dh" → "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver"
"pglg1"   → "a person places a large gear onto gear plate hole 1"
"ishc3"   → "a person inserts a hex screw into cylinder plate hole 3"

# Manipulation actions
"gsh"     → "a person grasps a hex screw"
"mglg2"   → "a person moves a large gear to gear plate hole 2"
"rcs"     → "a person rotates a cylinder subassembly"

# Disassembly actions
"dshc1dh" → "a person disassembles a hex screw from cylinder plate hole 1 with a hex screwdriver"

# Tool approach
"adh"     → "a person approaches a hex screwdriver"

# Noise
"null"    → "noise: null"
```

### Real Dataset Statistics

Based on HAViD annotation specification:
- **Number of verbs**: 10
- **Number of objects**: ~50
- **Number of targets**: ~10
- **Number of tools**: 4
- **Estimated vocabulary size**: 100-500 unique action combinations
- **Typical sequence length**: 1000-5000 frames per video

## Open-Vocabulary Benefits

Converting HAViD labels to natural language enables:

1. **Zero-shot recognition**: The model can recognize unseen action combinations
2. **Transfer learning**: Leverage CLIP's pre-trained vision-language knowledge
3. **Compositional understanding**: The model learns to understand verb-object-target-tool compositions
4. **Richer semantics**: Text descriptions capture meaning lost in numeric labels

## Architecture Integration

The text prompts feed into the open-vocabulary FACT pipeline:

```
HAViD Label → Text Prompt → CLIP Text Encoder → Text Features (512,)
                                                        ↓
                                                  Cosine Similarity
                                                        ↓
I3D Features (2048, T) → Projection Layer → Visual Features (512, T)
```

## Future Extensions

Potential improvements:
1. **Multi-template ensemble**: Generate multiple prompts per action and average embeddings
2. **Augmentation**: Add contextual information (e.g., "in a factory assembly task")
3. **Hierarchy**: Leverage object hierarchies (e.g., "hex screw" is a type of "screw")
4. **Language models**: Use LLMs to generate more diverse descriptions

## References

- HAViD Dataset: https://arxiv.org/pdf/2307.05721
- CLIP: https://arxiv.org/abs/2103.00020
- ActionCLIP: https://arxiv.org/pdf/2109.08472
- FACT: https://openaccess.thecvf.com/content/CVPR2024/papers/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.pdf

