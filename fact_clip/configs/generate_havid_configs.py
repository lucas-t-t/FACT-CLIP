#!/usr/bin/env python3
"""
Script to generate HAViD configuration files for different variants.
This script helps automate the creation of HAViD-specific configurations.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

def generate_havid_config(havid_base_path, variant, output_dir=None):
    """
    Generate a HAViD configuration file for a specific variant.
    
    Args:
        havid_base_path (str): Path to the HAViD ActionSegmentation directory
        variant (str): HAViD variant (e.g., 'view0_lh_pt', 'view0_lh_aa')
        output_dir (str): Output directory for the config file
    """
    
    havid_path = Path(havid_base_path)
    variant_path = havid_path / variant
    
    if not variant_path.exists():
        print(f"Error: Variant path {variant_path} does not exist")
        return False
    
    # Check required files
    required_files = ['groundTruth', 'splits', 'mapping.txt']
    for req_file in required_files:
        if not (variant_path / req_file).exists():
            print(f"Error: Required file/folder {req_file} not found in {variant_path}")
            return False
    
    # Check features folder
    features_path = havid_path / 'features'
    if not features_path.exists():
        print(f"Error: Features folder not found at {features_path}")
        return False
    
    # Determine annotation type and adjust ntoken
    if variant.endswith('_pt'):
        ntoken = 40  # Primitive tasks
        annotation_type = "primitive tasks"
    elif variant.endswith('_aa'):
        ntoken = 60  # Atomic actions
        annotation_type = "atomic actions"
    else:
        ntoken = 50  # Default
        annotation_type = "unknown"
    
    # Generate config content
    config_content = f"""BU:
  a: sa
  a_dim: null
  a_ffdim: null
  a_layers: 1
  a_nhead: 8
  dropout: null
  f: null
  f_dim: null
  f_layers: 10
  f_ln: null
  f_ngp: null
  hid_dim: null
  s_layers: 1
Bi:
  a: sca
  a_dim: 512
  a_ffdim: 512
  a_layers: 6
  a_nhead: 8
  dropout: 0.0
  f: m2
  f_dim: 512
  f_layers: 10
  f_ln: false
  f_ngp: 1
  hid_dim: 512
Bu:
  a: sa
  a_dim: null
  a_ffdim: null
  a_layers: 1
  a_nhead: 8
  dropout: null
  f: null
  f_dim: null
  f_layers: 10
  f_ln: null
  f_ngp: null
  hid_dim: null
FACT:
  block: iuUU
  cmr: 0.3
  fpos: false
  ntoken: {ntoken}  # Adjusted for HAViD {annotation_type}
  trans: false
  mwt: 0.1
Loss:
  a2fc: 1.0
  bgw: 1.0
  match: o2o
  nullw: -1.0
  pc: 0.2
  sw: 5.0
TM:
  inplace: true
  m: 5
  p: 0.05
  t: 30
  use: true
aux:
  debug: false
  eval_every: 2000
  gpu: 0
  mark: "{variant}"
  print_every: 1000
  resume: max
  runid: 0
batch_size: 2  # Reduced for HAViD due to longer sequences
clip_grad_norm: 10.0
dataset: havid_{variant}
eval_bg: true
epoch: 150
lr: 0.0001
lr_decay: 80
momentum: 0.0
optimizer: Adam
split: split1
sr: 1
weight_decay: 0.0

# HAViD-specific dataset paths for {variant} variant
feature_path: "{features_path}"
groundTruth_path: "{variant_path / 'groundTruth'}"
split_path: "{variant_path / 'splits'}"
map_fname: "{variant_path / 'mapping.txt'}"
feature_transpose: false
bg_class: 0
average_transcript_len: 0.0  # Will be computed by gen_config.py
"""
    
    # Write config file
    if output_dir is None:
        output_dir = current_dir
    
    output_path = Path(output_dir) / f"havid_{variant}.yaml"
    
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Generated HAViD config: {output_path}")
    print(f"  Variant: {variant}")
    print(f"  Annotation type: {annotation_type}")
    print(f"  Action tokens: {ntoken}")
    print(f"  Features path: {features_path}")
    print(f"  Labels path: {variant_path / 'groundTruth'}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Generate HAViD configuration files for different variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate config for view0_lh_pt variant
  python generate_havid_configs.py --havid_path /path/to/HAViD/ActionSegmentation --variant view0_lh_pt
  
  # Generate config for view0_lh_aa variant
  python generate_havid_configs.py --havid_path /path/to/HAViD/ActionSegmentation --variant view0_lh_aa
  
  # Generate all available variants
  python generate_havid_configs.py --havid_path /path/to/HAViD/ActionSegmentation --all
        """
    )
    
    parser.add_argument(
        '--havid_path',
        type=str,
        required=True,
        help='Path to the HAViD ActionSegmentation directory'
    )
    
    parser.add_argument(
        '--variant',
        type=str,
        help='HAViD variant to generate config for (e.g., view0_lh_pt)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate configs for all available variants'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for config files (default: current directory)'
    )
    
    args = parser.parse_args()
    
    havid_path = Path(args.havid_path)
    
    if not havid_path.exists():
        print(f"Error: HAViD path {havid_path} does not exist")
        return 1
    
    if args.all:
        # Find all variant directories
        variants = []
        for item in havid_path.iterdir():
            if item.is_dir() and item.name.startswith('view'):
                variants.append(item.name)
        
        if not variants:
            print(f"No HAViD variants found in {havid_path}")
            return 1
        
        print(f"Found {len(variants)} HAViD variants: {variants}")
        
        success_count = 0
        for variant in variants:
            if generate_havid_config(havid_path, variant, args.output_dir):
                success_count += 1
        
        print(f"\n✓ Successfully generated {success_count}/{len(variants)} config files")
        
    elif args.variant:
        if generate_havid_config(havid_path, args.variant, args.output_dir):
            print("✓ Config generation completed successfully")
            return 0
        else:
            return 1
    else:
        print("Error: Please specify either --variant or --all")
        return 1

if __name__ == "__main__":
    sys.exit(main())


