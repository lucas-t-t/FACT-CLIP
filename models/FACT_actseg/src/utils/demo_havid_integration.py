#!/usr/bin/python3
"""
Demo script showing how HAViD labels are converted to text prompts
for the open-vocabulary FACT model.
"""

from havid_text_prompts import generate_action_prompt, parse_havid_label


def demo_label_conversion():
    """Show examples of HAViD label to text conversion."""
    
    print("="*80)
    print("HAViD Label → Natural Language Conversion Demo")
    print("="*80)
    print("\nThis demonstrates how HAViD's compact action labels are converted to")
    print("natural language descriptions for CLIP text encoding in the open-vocabulary")
    print("FACT model.\n")
    
    # Example labels (based on HAViD annotation specification)
    example_labels = [
        # Full specification (verb + object + target + tool)
        "sshc1dh",  # screw + hex screw + cylinder plate hole 1 + hex screwdriver
        "sspc2dp",  # screw + phillips screw + cylinder plate hole 2 + phillips screwdriver
        
        # Placement actions
        "pglg1",    # place + large gear + gear plate hole 1
        "pgsg2",    # place + small gear + gear plate hole 2
        
        # Insertion actions
        "ishc1",    # insert + hex screw + cylinder plate hole 1
        "ispc3",    # insert + phillips screw + cylinder plate hole 3
        
        # Grasping actions
        "gsh",      # grasp + hex screw
        "gsp",      # grasp + phillips screw
        "ggl",      # grasp + large gear
        
        # Disassembly actions
        "dshc1dh",  # disassemble + hex screw + cylinder plate hole 1 + hex screwdriver
        "dspc2dp",  # disassemble + phillips screw + cylinder plate hole 2 + phillips screwdriver
        
        # Rotation and movement
        "rcs",      # rotate + cylinder subassembly
        "mglg1",    # move + large gear + gear plate hole 1
        
        # Approach actions
        "adh",      # approach + hex screwdriver
        "awndh",    # approach + nut wrench + hex screwdriver (?)
        
        # Noise labels
        "null",     # no action / null
    ]
    
    print("-"*80)
    print(f"{'Label':<15} | {'Parsed Components':<35} | {'Natural Language'}")
    print("-"*80)
    
    for label in example_labels:
        parsed = parse_havid_label(label)
        prompt = generate_action_prompt(label)
        
        # Construct component string
        components = []
        if parsed['verb'] and parsed['verb'] not in ['null', 'wrong']:
            components.append(f"v:{parsed['verb'][:4]}")
        if parsed['manipulated_object']:
            components.append(f"o:{parsed['manipulated_object'][:8]}")
        if parsed['target_object']:
            components.append(f"t:{parsed['target_object'][:8]}")
        if parsed['tool']:
            components.append(f"tool:{parsed['tool'][:8]}")
        
        component_str = " | ".join(components) if components else parsed['verb']
        
        print(f"{label:<15} | {component_str:<35} | {prompt}")
    
    print("-"*80)


def demo_dataset_usage():
    """Show how this integrates with the dataset."""
    
    print("\n" + "="*80)
    print("Integration with FACT Open-Vocabulary Dataset")
    print("="*80)
    print("\nExample workflow:\n")
    
    print("1. Load HAViD dataset with standard FACT dataloader")
    print("   - Returns I3D features: (2048, T) per video")
    print("   - Returns ground truth labels: list of acronyms per frame")
    print()
    
    print("2. Convert label acronyms to class indices (standard FACT)")
    print("   - 'sshc1dh' → class_idx: 42 (example)")
    print()
    
    print("3. Generate natural language descriptions (NEW for open-vocabulary)")
    print("   - class_idx: 42 → 'sshc1dh' → 'a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver'")
    print()
    
    print("4. Encode text with CLIP text encoder")
    print("   - text → CLIP text features: (512,) per class")
    print()
    
    print("5. Project I3D features to CLIP space")
    print("   - I3D: (2048, T) → projection → (512, T)")
    print()
    
    print("6. Compute similarity between visual and text features")
    print("   - cosine_similarity(visual_proj, text_features) → (num_classes, T)")
    print()
    
    print("7. Apply contrastive loss")
    print("   - Positive pairs: ground truth class")
    print("   - Negative pairs: all other classes")
    print()
    
    print("-"*80)


def show_statistics():
    """Show some statistics about HAViD labels."""
    
    print("\n" + "="*80)
    print("HAViD Label Statistics")
    print("="*80)
    
    # Based on the annotation specification
    verbs = ['a', 'd', 'g', 'h', 'i', 'l', 'm', 'p', 'r', 's']
    print(f"\nNumber of action verbs: {len(verbs)}")
    print(f"Verbs: {', '.join(verbs)}")
    
    # Approximate counts from the figure
    print(f"\nApproximate object types:")
    print(f"  - Screws/fasteners: ~10 types")
    print(f"  - Gears: ~6 types")
    print(f"  - Cylinder parts: ~8 types")
    print(f"  - Plates/holes: ~12 types")
    print(f"  - Tools: 4 types (dh, dp, wn, ws)")
    print(f"  - Misc parts: ~15 types")
    
    print(f"\nLabel structure:")
    print(f"  - Full: [verb(1)] + [object(2)] + [target(2)] + [tool(2)] = 7 chars")
    print(f"  - No tool: [verb(1)] + [object(2)] + [target(2)] = 5 chars")
    print(f"  - No target: [verb(1)] + [object(2)] = 3 chars")
    print(f"  - Verb only: [verb(1)] = 1 char")
    
    print(f"\nEstimated vocabulary size:")
    print(f"  - Theoretical max: 10 verbs × ~50 objects × ~10 targets × ~4 tools")
    print(f"  - Practical: likely 100-500 unique action combinations in dataset")
    
    print("\n" + "-"*80)


if __name__ == "__main__":
    demo_label_conversion()
    demo_dataset_usage()
    show_statistics()
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Pre-compute or dynamically encode all class descriptions with CLIP text encoder")
    print("  2. Add projection layer to map I3D features (2048,) → CLIP space (512,)")
    print("  3. Replace classification head with cosine similarity computation")
    print("  4. Replace cross-entropy loss with contrastive loss (KL divergence)")
    print("\nFor full implementation, see:")
    print("  - models/blocks_OpenVocab.py")
    print("  - models/loss_OpenVocab.py")
    print("  - utils/dataset_OpenVocab.py")
    print("="*80 + "\n")

