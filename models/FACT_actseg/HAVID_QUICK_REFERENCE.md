# HAViD Label → Text Conversion Quick Reference

## Quick Start

```python
from utils.havid_text_prompts import generate_action_prompt

# Convert HAViD label to text
label = "sshc1dh"
text = generate_action_prompt(label)
# Result: "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver"
```

## Label Format

```
[verb][object][target][tool]
  1     2        2      2     = 7 characters (full)
  
Examples:
  sshc1dh  (7 chars) - screw + hex screw + cylinder hole 1 + hex screwdriver
  ishc3    (5 chars) - insert + hex screw + cylinder hole 3
  gsh      (3 chars) - grasp + hex screw
  s        (1 char)  - screw
  null              - noise
```

## Verb Codes

| Code | Verb | Example Output |
|------|------|----------------|
| `a` | approaches | a person approaches ... |
| `d` | disassembles | a person disassembles ... from ... |
| `g` | grasps | a person grasps ... |
| `h` | holds | a person holds ... |
| `i` | inserts | a person inserts ... into ... |
| `l` | slides | a person slides ... onto ... |
| `m` | moves | a person moves ... to ... |
| `p` | places | a person places ... onto ... |
| `r` | rotates | a person rotates ... |
| `s` | screws | a person screws ... into ... |

## Common Object Codes

### Screws & Fasteners
- `sh` - hex screw
- `sp` - phillips screw
- `sb` - bolt
- `nt` - nut

### Gears
- `gl` - large gear
- `gs` - small gear
- `gw` - worm gear

### Cylinder Parts
- `cb` - cylinder base
- `cc` - cylinder cap
- `ck` - cylinder bracket
- `cs` - cylinder subassembly
- `c1`-`c4` - cylinder plate holes 1-4

### Gear Plate
- `g1`-`g3` - gear plate holes 1-3

### General Plate
- `n1`-`n4` - general plate holes 1-4

### Tools
- `dh` - hex screwdriver
- `dp` - phillips screwdriver
- `wn` - nut wrench
- `ws` - shaft wrench

### Other
- `ba` - ball
- `bs` - ball seat
- `bx` - box
- `ft` - gear shaft
- `hw` - hand-wheel
- `ib` - bar
- `ir` - rod
- `pl` - large spacer
- `ps` - small spacer

## Example Conversions

```
# Assembly
sshc1dh → "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver"
sspc2dp → "a person screws a phillips screw into cylinder plate hole 2 with a phillips screwdriver"
pglg1   → "a person places a large gear onto gear plate hole 1"
ishc3   → "a person inserts a hex screw into cylinder plate hole 3"

# Manipulation
gsh     → "a person grasps a hex screw"
ggl     → "a person grasps a large gear"
mglg2   → "a person moves a large gear to gear plate hole 2"
rcs     → "a person rotates a cylinder subassembly"

# Disassembly
dshc1dh → "a person disassembles a hex screw from cylinder plate hole 1 with a hex screwdriver"
dspc2   → "a person disassembles a phillips screw from cylinder plate hole 2"

# Tool approach
adh     → "a person approaches a hex screwdriver"
awndh   → "a person approaches a nut wrench to dh"

# Noise
null    → "noise: null"
w       → "noise: wrong"
```

## Integration with Dataset

```python
from utils.dataset_OpenVocab import create_dataset_openvocab

# Automatically converts HAViD labels
dataset, test_dataset, text_embeddings, descriptions = create_dataset_openvocab(cfg)

# descriptions[i] contains natural language for class i
print(descriptions[0])
# "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver"
```

## Testing

```bash
# Run tests
cd models/FACT_actseg/src/utils
python test_havid_prompts.py

# Run demo
python demo_havid_integration.py
```

## Files

- **Implementation**: `src/utils/havid_text_prompts.py`
- **Tests**: `src/utils/test_havid_prompts.py`
- **Demo**: `src/utils/demo_havid_integration.py`
- **Full Documentation**: `src/utils/HAVID_TEXT_PROMPTS_README.md`

## Advanced Usage

### Custom Templates

```python
template = "video of person {verb} {manipulated_object}"
text = generate_action_prompt("gsh", template=template)
```

### Parse Label Components

```python
from utils.havid_text_prompts import parse_havid_label

parsed = parse_havid_label("sshc1dh")
# {
#   'verb': 'screws',
#   'manipulated_object': 'hex screw',
#   'target_object': 'cylinder plate hole 1',
#   'tool': 'hex screwdriver'
# }
```

### Batch Processing

```python
from utils.havid_text_prompts import get_prompts_for_labels

labels = ["sshc1dh", "gsh", "pglg1"]
prompts = get_prompts_for_labels(labels)
```

## Troubleshooting

### Label not recognized?
- Check label format (1, 3, 5, or 7 characters)
- Verify verb code is valid (a, d, g, h, i, l, m, p, r, s)
- Unknown objects will be passed through as-is

### Using non-HAViD dataset?
- System auto-detects and falls back to simple formatting
- Example: `"crack_egg"` → `"a video of a person crack egg"`

### Need more objects?
- Add to `MANIPULATED_OBJECT_MAP` in `havid_text_prompts.py`
- Follow format: `'code': 'description'`

---

**Need help?** See full documentation: `src/utils/HAVID_TEXT_PROMPTS_README.md`

