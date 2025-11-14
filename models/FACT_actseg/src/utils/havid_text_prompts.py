#!/usr/bin/python3
"""
Text prompt generation utilities for HA-ViD dataset labels.

HA-ViD labels encode actions in a compact format:
- Action verb: a single character (a=approach, d=disassemble, g=grasp, h=hold, i=insert,
  l=slide, m=move, p=place, r=rotate, s=screw)
- Manipulated object: two characters (e.g., sh=hex screw, sp=phillips screw, ... )
- Target object: two characters (e.g., c1=cylinder plate hole 1, g1=gear plate hole 1, ... )
- Tool: two characters (dh=hex screwdriver, dp=phillips screwdriver, wn=nut wrench, ws=shaft wrench)

Example: "sshc1dh" → "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver"
"""

from typing import Dict, List, Optional


# ----------------------------
# Vocabulary from the figure
# ----------------------------

VERB_MAP = {
    'a': 'approaches',
    'd': 'disassembles',
    'g': 'grasps',
    'h': 'holds',
    'i': 'inserts',
    'l': 'slides',
    'm': 'moves',
    'p': 'places',
    'r': 'rotates',
    's': 'screws',
}

# For slightly nicer English ("insert into", "place onto", etc.)
VERB_PREP = {
    'approaches': 'to',
    'disassembles': 'from',
    'grasps': '',          # no preposition by default
    'holds': '',           # "
    'inserts': 'into',
    'slides': 'onto',
    'moves': 'to',
    'places': 'onto',
    'rotates': 'on',
    'screws': 'into',
}

# Manipulated objects (subset shown in the figure; add more as needed)
MANIPULATED_OBJECT_MAP = {
    # screws, bolts, nuts
    'sh': 'hex screw',
    'sp': 'phillips screw',
    'sb': 'bolt',
    'nt': 'nut',

    # gears & gear plates
    'gl': 'large gear',
    'gs': 'small gear',
    'gw': 'worm gear',
    'g1': 'gear plate hole 1',
    'g2': 'gear plate hole 2',
    'g3': 'gear plate hole 3',

    # cylinder assembly parts
    'c1': 'cylinder plate hole 1',
    'c2': 'cylinder plate hole 2',
    'c3': 'cylinder plate hole 3',
    'c4': 'cylinder plate hole 4',
    'cb': 'cylinder base',
    'cc': 'cylinder cap',
    'ck': 'cylinder bracket',
    'cs': 'cylinder subassembly',

    # general plate holes / studs
    'n1': 'general plate hole 1',
    'n2': 'general plate hole 2',
    'n3': 'general plate hole 3',
    'n4': 'general plate hole 4',
    'n5': 'general plate stud',
    'n6': 'general plate usb female',

    # spacers
    'pl': 'large spacer',
    'ps': 'small spacer',

    # tools can sometimes be manipulated (appear in object list in figure)
    'dh': 'hex screwdriver',
    'dp': 'phillips screwdriver',
    'wn': 'nut wrench',
    'ws': 'shaft wrench',

    # misc parts shown
    'ba': 'ball',
    'bs': 'ball seat',
    'bx': 'box',
    'ft': 'gear shaft',
    'hd': 'dial',
    'hw': 'hand-wheel',
    'hq': 'quarter-turn handle',
    'ib': 'bar',
    'ir': 'rod',
    'lb': 'linear bearing',
    'us': 'usb male',
}

# Targets (the same IDs can be targets; most common ones from the figure)
TARGET_OBJECT_MAP = {
    'c1': 'cylinder plate hole 1',
    'c2': 'cylinder plate hole 2',
    'c3': 'cylinder plate hole 3',
    'c4': 'cylinder plate hole 4',
    'g1': 'gear plate hole 1',
    'g2': 'gear plate hole 2',
    'g3': 'gear plate hole 3',
    'n1': 'general plate hole 1',
    'n2': 'general plate hole 2',
    'n3': 'general plate hole 3',
    'n4': 'general plate hole 4',
}

# Tools
TOOL_MAP = {
    'dh': 'hex screwdriver',
    'dp': 'phillips screwdriver',
    'wn': 'nut wrench',
    'ws': 'shaft wrench',
}

# Noise (from the figure)
NOISE_MAP = {
    'null': 'null',
    'w': 'wrong',
}


# ----------------------------
# Parsing & prompt generation
# ----------------------------

def parse_havid_label(label: str) -> Dict[str, Optional[str]]:
    """
    Parse HA-ViD label into components.
    Returns keys: 'verb', 'manipulated_object', 'target_object', 'tool'
    """
    if not label:
        return {'verb': None, 'manipulated_object': None, 'target_object': None, 'tool': None}

    lab = label.strip().lower()

    # Noise labels (e.g., "null", "w")
    if lab in NOISE_MAP:
        return {'verb': NOISE_MAP[lab], 'manipulated_object': None, 'target_object': None, 'tool': None}

    parsed = {'verb': None, 'manipulated_object': None, 'target_object': None, 'tool': None}

    # Expected: 1 + 2 + 2 + 2 = 7 chars (e.g., s sh c1 dh)
    if len(lab) >= 7:
        verb_char = lab[0]
        manipulated = lab[1:3]
        target = lab[3:5]
        tool = lab[5:7]

        verb = VERB_MAP.get(verb_char)
        parsed['verb'] = verb if verb else verb_char
        parsed['manipulated_object'] = MANIPULATED_OBJECT_MAP.get(manipulated, manipulated)
        parsed['target_object'] = TARGET_OBJECT_MAP.get(target, target)
        parsed['tool'] = TOOL_MAP.get(tool, tool)
        return parsed

    # 1 + 2 + 2 (no tool)
    if len(lab) >= 5:
        verb_char = lab[0]
        manipulated = lab[1:3]
        target = lab[3:5]

        verb = VERB_MAP.get(verb_char)
        parsed['verb'] = verb if verb else verb_char
        parsed['manipulated_object'] = MANIPULATED_OBJECT_MAP.get(manipulated, manipulated)
        parsed['target_object'] = TARGET_OBJECT_MAP.get(target, target)
        return parsed

    # 1 + 2 (verb + object only)
    if len(lab) >= 3:
        verb_char = lab[0]
        manipulated = lab[1:3]
        verb = VERB_MAP.get(verb_char)
        parsed['verb'] = verb if verb else verb_char
        parsed['manipulated_object'] = MANIPULATED_OBJECT_MAP.get(manipulated, manipulated)
        return parsed

    # Only verb
    verb = VERB_MAP.get(lab[0], lab[0])
    parsed['verb'] = verb
    return parsed


def _choose_prep(verb_text: Optional[str]) -> str:
    """Get appropriate preposition for a verb."""
    if not verb_text:
        return ''
    return VERB_PREP.get(verb_text, '')


def generate_action_prompt(label: str, template: Optional[str] = None) -> str:
    """
    Generate a natural-language prompt from a HA-ViD label.
    
    Args:
        label: HA-ViD label (e.g., "sshc1dh")
        template: Optional custom template string
    
    Returns:
        Natural language description (e.g., "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver")
    """
    parsed = parse_havid_label(label)

    verb = parsed.get('verb')
    manipulated = parsed.get('manipulated_object')
    target = parsed.get('target_object')
    tool = parsed.get('tool')

    # Noise handling
    if verb in ('null', 'wrong'):
        return f'noise: {verb}'

    prep = _choose_prep(verb)

    # Default templates with nicer prepositions
    if template is None:
        if tool and target and manipulated:
            # add "with a" article for tools and "a" for manipulated
            if prep:
                template = "a person {verb} a {manipulated_object} {prep} {target_object} with a {tool}"
            else:
                template = "a person {verb} a {manipulated_object} to {target_object} with a {tool}"
        elif target and manipulated:
            if prep:
                template = "a person {verb} a {manipulated_object} {prep} {target_object}"
            else:
                template = "a person {verb} a {manipulated_object} to {target_object}"
        elif manipulated:
            template = "a person {verb} a {manipulated_object}"
        else:
            template = "a person {verb}"

    # Fill template
    prompt = template.format(
        verb=verb if verb else "perform action",
        manipulated_object=manipulated if manipulated else "object",
        target_object=target if target else "target",
        tool=tool if tool else "tool",
        prep=(f"{prep}" if prep else "").strip()
    ).replace("  ", " ").strip()

    # Clean possible trailing/extra spaces if prep was empty
    prompt = prompt.replace("  ", " ").replace(" a a ", " a ")

    return prompt


def get_all_prompts(label2index: Dict[str, int],
                    index2label: Dict[int, str],
                    template: Optional[str] = None) -> List[str]:
    """
    Generate prompts ordered by class index.
    
    Args:
        label2index: Mapping from label string to class index
        index2label: Mapping from class index to label string
        template: Optional custom template
    
    Returns:
        List of prompts ordered by class index
    """
    n_classes = len(index2label)
    prompts: List[str] = []
    for i in range(n_classes):
        lbl = index2label.get(i)
        if lbl is None:
            prompts.append(f"a person performs action {i}")
        else:
            prompts.append(generate_action_prompt(lbl, template))
    return prompts


def get_prompts_for_labels(labels: List[str], template: Optional[str] = None) -> List[str]:
    """
    Generate prompts for a list of labels (e.g., a transcript).
    
    Args:
        labels: List of HA-ViD labels
        template: Optional custom template
    
    Returns:
        List of natural language descriptions
    """
    return [generate_action_prompt(l, template) for l in labels]


def normalize_label_for_parsing(label: str) -> str:
    """Normalize label string for parsing (trim + lowercase)."""
    return label.strip().lower()


# Simple fallback for non-HA-ViD datasets
def generate_simple_prompt(label: str, template: str = "a person {action}") -> str:
    """
    Generate simple prompt for non-HAViD datasets.
    
    Args:
        label: Action label (e.g., "crack_egg")
        template: Template string with {action} placeholder
    
    Returns:
        Natural language description
    """
    return template.format(action=label.replace('_', ' '))


# ----------------------------
# Integration helper
# ----------------------------

def is_havid_label(label: str) -> bool:
    """
    Check if a label follows HAViD format.
    
    Args:
        label: Label string
    
    Returns:
        True if label appears to be HAViD format
    """
    if not label:
        return False
    
    lab = label.strip().lower()
    
    # Noise labels
    if lab in NOISE_MAP:
        return True
    
    # Check if first character is a known verb
    if len(lab) >= 1 and lab[0] in VERB_MAP:
        return True
    
    return False

