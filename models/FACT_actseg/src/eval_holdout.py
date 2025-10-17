#!/usr/bin/env python3
"""
Standalone evaluation script for analyzing holdout training results.

Usage:
    python -m src.eval_holdout --checkpoint_path log/exp/run/saves/10000.gz
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from utils.evaluate import Checkpoint
from home import get_project_base

def analyze_checkpoint(ckpt_path, output_dir=None):
    """
    Analyze a saved checkpoint and generate detailed reports.
    
    Args:
        ckpt_path: Path to checkpoint file (.gz)
        output_dir: Directory to save analysis results
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = Checkpoint.load(ckpt_path)
    
    print(f"\nCheckpoint Iteration: {ckpt.iteration}")
    print(f"Number of videos: {len(ckpt.videos)}")
    
    # Print metrics summary
    print("\n" + "="*80)
    print("METRICS SUMMARY")
    print("="*80)
    
    if hasattr(ckpt, 'metrics'):
        # Separate metrics by type
        all_metrics = {}
        seen_metrics = {}
        unseen_metrics = {}
        
        for k, v in ckpt.metrics.items():
            if '-seen' in k:
                seen_metrics[k.replace('-seen', '')] = v
            elif '-unseen' in k:
                unseen_metrics[k.replace('-unseen', '')] = v
            else:
                all_metrics[k] = v
        
        # Print all classes metrics
        print("\nAll Classes:")
        print("-" * 80)
        for k, v in all_metrics.items():
            print(f"  {k:20s}: {v:6.2f}")
        
        # Print seen classes metrics
        if seen_metrics:
            print("\nSeen Classes:")
            print("-" * 80)
            for k, v in seen_metrics.items():
                print(f"  {k:20s}: {v:6.2f}")
        
        # Print unseen classes metrics
        if unseen_metrics:
            print("\nUnseen Classes:")
            print("-" * 80)
            for k, v in unseen_metrics.items():
                print(f"  {k:20s}: {v:6.2f}")
        
        # Compute zero-shot gap
        if seen_metrics and unseen_metrics:
            print("\nZero-Shot Gap (Seen - Unseen):")
            print("-" * 80)
            for k in seen_metrics:
                if k in unseen_metrics:
                    gap = seen_metrics[k] - unseen_metrics[k]
                    print(f"  {k:20s}: {gap:6.2f}")
    
    # Print per-class metrics
    if hasattr(ckpt, 'per_class_metrics') and len(ckpt.per_class_metrics) > 0:
        print("\n" + "="*80)
        print("PER-CLASS METRICS")
        print("="*80)
        
        # Sort by class ID
        sorted_classes = sorted(ckpt.per_class_metrics.items())
        
        print(f"{'Class ID':<10} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Type':<10}")
        print("-" * 80)
        
        for cls_id, metrics in sorted_classes:
            cls_type = "Unseen" if cls_id in ckpt.holdout_classes else "Seen"
            print(f"{cls_id:<10} {metrics['correct']:<10} {metrics['total']:<10} {metrics['accuracy']:<10.2f} {cls_type:<10}")
    
    # Save detailed analysis if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comprehensive JSON report
        report_path = os.path.join(output_dir, f"analysis_iter_{ckpt.iteration}.json")
        report = {
            'iteration': ckpt.iteration,
            'num_videos': len(ckpt.videos),
            'holdout_classes': ckpt.holdout_classes if hasattr(ckpt, 'holdout_classes') else [],
            'seen_classes': ckpt.seen_classes if hasattr(ckpt, 'seen_classes') else [],
            'metrics': dict(ckpt.metrics) if hasattr(ckpt, 'metrics') else {},
            'per_class_metrics': ckpt.per_class_metrics if hasattr(ckpt, 'per_class_metrics') else {}
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")
        
        # Generate simple CSV for per-class metrics
        csv_path = os.path.join(output_dir, f"per_class_metrics_iter_{ckpt.iteration}.csv")
        with open(csv_path, 'w') as f:
            f.write("class_id,correct,total,accuracy,type\n")
            for cls_id in sorted(ckpt.per_class_metrics.keys()):
                metrics = ckpt.per_class_metrics[cls_id]
                cls_type = "unseen" if cls_id in ckpt.holdout_classes else "seen"
                f.write(f"{cls_id},{metrics['correct']},{metrics['total']},{metrics['accuracy']:.2f},{cls_type}\n")
        print(f"Per-class CSV saved to: {csv_path}")
    
    print("\n" + "="*80)
    return ckpt

def main():
    parser = argparse.ArgumentParser(
        description="Analyze holdout training checkpoint results"
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Path to checkpoint file (.gz)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save analysis results (default: same as checkpoint directory)'
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.checkpoint_path), 'analysis')
    
    # Analyze checkpoint
    analyze_checkpoint(args.checkpoint_path, args.output_dir)

if __name__ == "__main__":
    main()

