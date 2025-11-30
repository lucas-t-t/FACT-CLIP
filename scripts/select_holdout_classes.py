#!/usr/bin/python3
"""
Script to select optimal holdout classes for Zero-Shot learning on HA-ViD.
Strategy: Select classes whose components (Verbs/Nouns) are well-represented 
in the remaining training set, enabling Compositional Zero-Shot transfer.
"""

import os
import sys
from collections import Counter
from pathlib import Path

from fact_clip.utils.havid_text_prompts import parse_havid_label
from fact_clip.home import get_project_base

def main():
    # Configuration
    dataset_variant = 'view0_lh_pt'
    base_path = os.path.join(get_project_base(), 'data', 'HAViD', 'ActionSegmentation', 'data', dataset_variant)
    map_path = os.path.join(base_path, 'mapping.txt')
    train_split = os.path.join(base_path, 'splits', 'train.split1.bundle')
    
    print(f"Analyzing dataset: {dataset_variant}")
    print(f"Path: {base_path}")
    
    # 1. Load Mapping
    label2index = {}
    index2label = {}
    try:
        with open(map_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                idx, label = line.strip().split(' ', 1)
                idx = int(idx)
                label2index[label] = idx
                index2label[idx] = label
    except FileNotFoundError:
        print(f"Error: Could not find mapping file at {map_path}")
        return

    # 2. Load Training Data Counts
    print("Loading training data stats...")
    train_counts = Counter()
    
    try:
        with open(train_split) as f:
            train_videos = [ln.strip() for ln in f if ln.strip()]
        train_videos = [v[:-4] if v.endswith('.txt') else v for v in train_videos]
        
        for v in train_videos:
            path = os.path.join(base_path, 'groundTruth', v + '.txt')
            with open(path, 'rb') as f:
                content = f.read().replace(b'\r\n', b'\n')
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    text = content.decode('latin-1')
            
            labels = []
            for line in text.split('\n'):
                line = line.strip()
                if not line or line not in label2index: continue
                labels.append(label2index[line])
            
            train_counts.update(labels)
            
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    print(f"Total training frames: {sum(train_counts.values())}")
    
    # 3. Parse all classes into components
    class_components = {} # idx -> {'verb': v, 'noun': n}
    
    # Vocabulary stats across the whole dataset
    verb_counts = Counter()
    noun_counts = Counter()
    
    for idx, label in index2label.items():
        if label == 'background' or idx == 27: # 27 is typically bg in havid or null
            continue
            
        parsed = parse_havid_label(label)
        
        # simplify: noun = manipulated object + target object
        # We want to see if the visual objects appear elsewhere
        
        verb = parsed.get('verb')
        
        # Create a composite "visual noun" signature
        # e.g. "screw" + "hole1"
        objs = []
        if parsed.get('manipulated_object'): objs.append(parsed['manipulated_object'])
        if parsed.get('target_object'): objs.append(parsed['target_object'])
        if parsed.get('tool'): objs.append(parsed['tool'])
        
        class_components[idx] = {
            'label': label,
            'verb': verb,
            'objects': objs,
            'frame_count': train_counts[idx]
        }

    # 4. Score Candidates
    # A good holdout candidate:
    # - Is NOT background
    # - Has sufficient frames (so removing it actually tests something meaningful)
    # - Its Verb appears frequently in OTHER classes
    # - Its Objects appear frequently in OTHER classes
    
    candidates = []
    
    for target_idx in class_components:
        target = class_components[target_idx]
        
        # Calculate support in the rest of the dataset (if we remove this class)
        verb_support = 0
        object_support = 0
        
        for other_idx, other in class_components.items():
            if other_idx == target_idx: continue
            
            # Add frames from this other class to support scores
            frames = other['frame_count']
            if frames == 0: continue
            
            # Check verb match
            if other['verb'] == target['verb']:
                verb_support += frames
            
            # Check object match (fuzzy: does OTHER share ANY object with TARGET?)
            # Ideally we want strong support for ALL objects in target
            target_objs = set(target['objects'])
            other_objs = set(other['objects'])
            
            common = target_objs.intersection(other_objs)
            if common:
                # Add support proportional to how many objects match
                # e.g. if target has 2 objects and other has 1 matching, adding support
                object_support += frames * (len(common) / max(1, len(target_objs)))

        # Score = Geometric mean of verb and object support
        # We need BOTH to be supported for compositional zero-shot
        score = (verb_support * object_support) ** 0.5
        
        candidates.append({
            'id': target_idx,
            'label': target['label'],
            'frames': target['frame_count'],
            'verb': target['verb'],
            'objects': target['objects'],
            'score': score,
            'verb_support': verb_support,
            'obj_support': object_support
        })

    # Sort by score (best candidates first)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "="*100)
    print(f"{'ID':<4} {'Label':<15} {'Frames':<8} {'Score':<10} {'Verb Support':<12} {'Obj Support':<12} {'Components'}")
    print("="*100)
    
    top_candidates = []
    
    # Filter: must have > 500 frames (to be significant) and good support
    for c in candidates:
        if c['frames'] > 500: # Only consider classes with decent data amount
            print(f"{c['id']:<4} {c['label']:<15} {c['frames']:<8} {int(c['score']):<10} {c['verb_support']:<12} {int(c['obj_support']):<12} {c['verb']} | {c['objects']}")
            top_candidates.append(c['id'])
            
    print("\n" + "="*80)
    print("RECOMMENDED HOLDOUT CONFIGURATION")
    print("="*80)
    
    # Select top 5 distinct actions
    selected = top_candidates[:5]
    print(f"Top 5 Compositional Zero-Shot Candidates: {selected}")
    print("These classes have high frequency but their components (verbs/objects) are well-seen in other classes.")
    print("Removing them forces the model to recombine known concepts (e.g. known 'screw' + known 'gear').")
    
    print("\nYAML Config snippet:")
    print(f"holdout_classes: {selected}")

if __name__ == "__main__":
    main()

