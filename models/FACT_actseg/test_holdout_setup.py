#!/usr/bin/env python3
"""
Quick validation test for holdout training setup.
This script verifies that the holdout configuration loads correctly
and filtering works as expected WITHOUT running actual training.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.configs.utils import setup_cfg
from src.utils.dataset import create_dataset
from src.home import get_project_base

def test_holdout_setup():
    """Test that holdout configuration and filtering works correctly."""
    
    print("="*80)
    print("HOLDOUT SETUP VALIDATION TEST")
    print("="*80)
    
    # Test 1: Load holdout configuration
    print("\n[Test 1] Loading holdout configuration...")
    try:
        cfg = setup_cfg(['src/configs/havid_view0_lh_pt_holdout.yaml'], None)
        print(f"✓ Configuration loaded successfully")
        print(f"  - Dataset: {cfg.dataset}")
        print(f"  - Holdout mode: {cfg.holdout_mode}")
        print(f"  - Holdout classes: {list(cfg.holdout_classes)}")
        print(f"  - Number of holdout classes: {len(cfg.holdout_classes)}")
        
        if not cfg.holdout_mode:
            print("✗ ERROR: holdout_mode is False!")
            return False
        
        if len(cfg.holdout_classes) == 0:
            print("✗ ERROR: holdout_classes is empty!")
            return False
            
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False
    
    # Test 2: Create dataset with holdout filtering
    print("\n[Test 2] Creating dataset with holdout filtering...")
    try:
        dataset, test_dataset = create_dataset(cfg)
        print(f"✓ Dataset created successfully")
        print(f"  - Training videos: {len(dataset.video_list)}")
        print(f"  - Test videos: {len(test_dataset.video_list)}")
        print(f"  - Number of classes: {dataset.nclasses}")
        print(f"  - Holdout classes in dataset: {dataset.holdout_classes}")
        print(f"  - Seen classes count: {len(dataset.seen_classes)}")
        
        if len(dataset.video_list) == 0:
            print("✗ ERROR: No training videos remaining after filtering!")
            return False
        
        if len(dataset.holdout_classes) != len(cfg.holdout_classes):
            print(f"✗ ERROR: Holdout classes mismatch!")
            return False
            
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Verify filtering worked correctly
    print("\n[Test 3] Verifying holdout filtering...")
    try:
        from src.utils.dataset import video_contains_holdout_classes
        
        # Check a few training videos don't contain holdout classes
        groundTruth_path = os.path.join(cfg.dataset, 'groundTruth')
        if not os.path.isabs(groundTruth_path):
            BASE = get_project_base()
            groundTruth_path = BASE + 'data/HAViD/ActionSegmentation/data/view0_lh_pt/groundTruth'
        
        checked_videos = 0
        contaminated = []
        for vname in dataset.video_list[:min(5, len(dataset.video_list))]:
            if video_contains_holdout_classes(vname, groundTruth_path, 
                                             dataset.label2index, cfg.holdout_classes):
                contaminated.append(vname)
            checked_videos += 1
        
        if contaminated:
            print(f"✗ ERROR: Found {len(contaminated)} training videos with holdout classes:")
            for v in contaminated:
                print(f"    - {v}")
            return False
        else:
            print(f"✓ Filtering verified: checked {checked_videos} training videos")
            print(f"  - No holdout classes found in training set")
            
    except Exception as e:
        print(f"✗ Failed verification: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Check evaluation setup
    print("\n[Test 4] Checking evaluation setup...")
    try:
        from src.utils.evaluate import Checkpoint
        
        ckpt = Checkpoint(
            0,
            bg_class=test_dataset.bg_class,
            holdout_classes=test_dataset.holdout_classes,
            seen_classes=test_dataset.seen_classes
        )
        
        print(f"✓ Checkpoint initialized successfully")
        print(f"  - Holdout classes: {len(ckpt.holdout_classes)}")
        print(f"  - Seen classes: {len(ckpt.seen_classes)}")
        print(f"  - Background classes: {ckpt.bg_class}")
        
        if len(ckpt.holdout_classes) == 0:
            print("✗ WARNING: Checkpoint has no holdout classes!")
            
    except Exception as e:
        print(f"✗ Failed to initialize checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print("✓ All tests passed!")
    print(f"✓ Configuration: {len(cfg.holdout_classes)} classes held out")
    print(f"✓ Dataset: {len(dataset.video_list)} training videos (filtered)")
    print(f"✓ Dataset: {len(test_dataset.video_list)} test videos (unchanged)")
    print(f"✓ Evaluation: Checkpoint properly configured")
    print("\nThe holdout setup is ready for training!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = test_holdout_setup()
    sys.exit(0 if success else 1)

