#!/usr/bin/env python3
"""
Analyze HAViD dataset to select optimal classes for holdout (zero-shot) training.

This script:
1. Counts class occurrences across train/test splits
2. Identifies which videos contain each class
3. Recommends holdout classes (mix of frequent and medium-frequency)
4. Verifies sufficient test coverage and no train/test overlap
"""

import os
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import argparse

# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from home import get_project_base

def load_mapping(map_path):
    """Load class index to label mapping."""
    label2index = {}
    index2label = {}
    with open(map_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            idx, label = line.strip().split(' ', 1)
            idx = int(idx)
            label2index[label] = idx
            index2label[idx] = label
    return label2index, index2label

def read_video_labels(video_name, gt_path, label2index):
    """Read labels for a single video."""
    label_file = os.path.join(gt_path, video_name + '.txt')
    with open(label_file, 'rb') as f:
        raw = f.read().replace(b'\r\n', b'\n')
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        text = raw.decode('latin-1')
    
    labels = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line in label2index:
            labels.append(label2index[line])
    return labels

def analyze_dataset(dataset_path, split_name='split1'):
    """Analyze dataset to compute class statistics."""
    base_path = Path(dataset_path)
    map_path = base_path / 'mapping.txt'
    gt_path = base_path / 'groundTruth'
    train_split = base_path / 'splits' / f'train.{split_name}.bundle'
    test_split = base_path / 'splits' / f'test.{split_name}.bundle'
    
    print(f"Analyzing dataset: {dataset_path}")
    print(f"Split: {split_name}")
    print("-" * 80)
    
    # Load mapping
    label2index, index2label = load_mapping(map_path)
    nclasses = len(label2index)
    print(f"Total classes: {nclasses}")
    
    # Load splits
    with open(train_split) as f:
        train_videos = [ln.strip() for ln in f if ln.strip()]
    train_videos = [v[:-4] if v.endswith('.txt') else v for v in train_videos]
    
    with open(test_split) as f:
        test_videos = [ln.strip() for ln in f if ln.strip()]
    test_videos = [v[:-4] if v.endswith('.txt') else v for v in test_videos]
    
    print(f"Train videos: {len(train_videos)}")
    print(f"Test videos: {len(test_videos)}")
    print("-" * 80)
    
    # Analyze train split
    train_frame_counts = Counter()
    train_video_counts = Counter()
    train_class_to_videos = defaultdict(set)
    
    print("Analyzing training split...")
    for vname in train_videos:
        labels = read_video_labels(vname, gt_path, label2index)
        unique_classes = set(labels)
        train_frame_counts.update(labels)
        for cls in unique_classes:
            train_video_counts[cls] += 1
            train_class_to_videos[cls].add(vname)
    
    # Analyze test split
    test_frame_counts = Counter()
    test_video_counts = Counter()
    test_class_to_videos = defaultdict(set)
    
    print("Analyzing test split...")
    for vname in test_videos:
        labels = read_video_labels(vname, gt_path, label2index)
        unique_classes = set(labels)
        test_frame_counts.update(labels)
        for cls in unique_classes:
            test_video_counts[cls] += 1
            test_class_to_videos[cls].add(vname)
    
    return {
        'label2index': label2index,
        'index2label': index2label,
        'nclasses': nclasses,
        'train_videos': train_videos,
        'test_videos': test_videos,
        'train_frame_counts': train_frame_counts,
        'train_video_counts': train_video_counts,
        'train_class_to_videos': train_class_to_videos,
        'test_frame_counts': test_frame_counts,
        'test_video_counts': test_video_counts,
        'test_class_to_videos': test_class_to_videos,
    }

def select_holdout_classes(stats, n_frequent=6, n_medium=3, min_test_videos=3, bg_class=27, skip_top_n=5):
    """
    Select classes for holdout training.
    
    Args:
        stats: Dictionary with dataset statistics
        n_frequent: Number of frequent classes to hold out
        n_medium: Number of medium-frequency classes to hold out
        min_test_videos: Minimum number of test videos required per class
        bg_class: Background class index (excluded from holdout)
        skip_top_n: Skip the top N most frequent classes before selecting (default: 0)
    
    Returns:
        List of recommended holdout class indices
    """
    train_frame_counts = stats['train_frame_counts']
    test_video_counts = stats['test_video_counts']
    
    # Get classes sorted by training frame count (excluding background)
    sorted_classes = [
        (idx, count) for idx, count in train_frame_counts.most_common()
        if idx != bg_class
    ]
    
    # Filter classes that have sufficient test videos
    eligible_classes = [
        idx for idx, count in sorted_classes
        if test_video_counts[idx] >= min_test_videos
    ]
    
    print(f"\nClasses with >={min_test_videos} test videos: {len(eligible_classes)}/{len(sorted_classes)}")
    
    if skip_top_n > 0:
        print(f"Skipping top {skip_top_n} most frequent classes")
        print(f"  Skipped classes: {eligible_classes[:skip_top_n]}")
        skipped_names = [stats['index2label'][c] for c in eligible_classes[:skip_top_n]]
        print(f"  Skipped class names: {skipped_names}")
    
    if len(eligible_classes) < skip_top_n + n_frequent + n_medium:
        print(f"Warning: Not enough eligible classes after skipping. Adjusting selection criteria...")
        available = len(eligible_classes) - skip_top_n
        n_frequent = min(n_frequent, available // 2)
        n_medium = min(n_medium, available - n_frequent)
    
    # Select frequent classes (after skipping top N)
    frequent_start = skip_top_n
    frequent_end = skip_top_n + n_frequent
    frequent_holdout = eligible_classes[frequent_start:frequent_end]
    
    # Select medium-frequency classes (from middle third)
    middle_start = len(eligible_classes) // 3
    middle_end = 2 * len(eligible_classes) // 3
    medium_candidates = eligible_classes[middle_start:middle_end]
    
    # Randomly sample from medium candidates
    np.random.seed(42)  # For reproducibility
    n_medium = min(n_medium, len(medium_candidates))
    medium_holdout = list(np.random.choice(medium_candidates, n_medium, replace=False))
    
    holdout_classes = sorted(frequent_holdout + medium_holdout)
    
    return holdout_classes, eligible_classes

def print_class_statistics(stats, holdout_classes):
    """Print detailed statistics for selected holdout classes."""
    index2label = stats['index2label']
    train_frame_counts = stats['train_frame_counts']
    train_video_counts = stats['train_video_counts']
    test_frame_counts = stats['test_frame_counts']
    test_video_counts = stats['test_video_counts']
    
    print("\n" + "=" * 80)
    print("SELECTED HOLDOUT CLASSES")
    print("=" * 80)
    print(f"{'ID':<4} {'Label':<15} {'Train Frames':<12} {'Train Videos':<12} {'Test Frames':<12} {'Test Videos':<12}")
    print("-" * 80)
    
    total_train_frames = sum(train_frame_counts.values())
    total_test_frames = sum(test_frame_counts.values())
    
    for cls_idx in holdout_classes:
        label = index2label[cls_idx]
        train_f = train_frame_counts[cls_idx]
        train_v = train_video_counts[cls_idx]
        test_f = test_frame_counts[cls_idx]
        test_v = test_video_counts[cls_idx]
        print(f"{cls_idx:<4} {label:<15} {train_f:<12} {train_v:<12} {test_f:<12} {test_v:<12}")
    
    holdout_train_frames = sum(train_frame_counts[c] for c in holdout_classes)
    holdout_test_frames = sum(test_frame_counts[c] for c in holdout_classes)
    
    print("-" * 80)
    print(f"Total holdout classes: {len(holdout_classes)}")
    print(f"Holdout train frames: {holdout_train_frames} ({100*holdout_train_frames/total_train_frames:.1f}%)")
    print(f"Holdout test frames: {holdout_test_frames} ({100*holdout_test_frames/total_test_frames:.1f}%)")

def check_data_leakage(stats, holdout_classes):
    """Check for videos that appear in both train and test containing holdout classes."""
    train_videos = set(stats['train_videos'])
    test_videos = set(stats['test_videos'])
    train_class_to_videos = stats['train_class_to_videos']
    test_class_to_videos = stats['test_class_to_videos']
    
    # Videos containing holdout classes in train
    train_with_holdout = set()
    for cls in holdout_classes:
        train_with_holdout.update(train_class_to_videos[cls])
    
    # Videos containing holdout classes in test
    test_with_holdout = set()
    for cls in holdout_classes:
        test_with_holdout.update(test_class_to_videos[cls])
    
    # Check overlap
    overlap = train_with_holdout & test_with_holdout
    
    print("\n" + "=" * 80)
    print("DATA LEAKAGE CHECK")
    print("=" * 80)
    print(f"Train videos with holdout classes: {len(train_with_holdout)}")
    print(f"Test videos with holdout classes: {len(test_with_holdout)}")
    print(f"Videos in both train and test: {len(overlap)}")
    
    if overlap:
        print("\nWARNING: Found videos in both splits:")
        for v in list(overlap)[:5]:
            print(f"  - {v}")
        if len(overlap) > 5:
            print(f"  ... and {len(overlap)-5} more")
    else:
        print("\nNo data leakage detected!")
    
    return len(overlap) == 0

def compute_filtered_statistics(stats, holdout_classes):
    """Compute statistics after filtering training videos with holdout classes."""
    train_videos = stats['train_videos']
    train_class_to_videos = stats['train_class_to_videos']
    
    # Videos to be removed (containing ANY holdout class)
    videos_to_remove = set()
    for cls in holdout_classes:
        videos_to_remove.update(train_class_to_videos[cls])
    
    # Remaining videos
    filtered_train_videos = [v for v in train_videos if v not in videos_to_remove]
    
    print("\n" + "=" * 80)
    print("FILTERED TRAINING SET STATISTICS")
    print("=" * 80)
    print(f"Original train videos: {len(train_videos)}")
    print(f"Videos to remove: {len(videos_to_remove)} ({100*len(videos_to_remove)/len(train_videos):.1f}%)")
    print(f"Remaining train videos: {len(filtered_train_videos)} ({100*len(filtered_train_videos)/len(train_videos):.1f}%)")
    
    return filtered_train_videos

def main():
    parser = argparse.ArgumentParser(
        description="Analyze HAViD dataset and select classes for holdout training"
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='Path to dataset'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='split1',
        help='Dataset split to analyze'
    )
    parser.add_argument(
        '--n_frequent',
        type=int,
        default=6,
        help='Number of frequent classes to hold out'
    )
    parser.add_argument(
        '--n_medium',
        type=int,
        default=3,
        help='Number of medium-frequency classes to hold out'
    )
    parser.add_argument(
        '--min_test_videos',
        type=int,
        default=5,
        help='Minimum number of test videos required per holdout class'
    )
    parser.add_argument(
        '--skip_top_n',
        type=int,
        default=0,
        help='Skip the top N most frequent classes before selecting holdout classes (default: 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file to save holdout class list'
    )
    
    args = parser.parse_args()
    
    # Default dataset path
    if args.dataset_path is None:
        BASE = get_project_base()
        args.dataset_path = os.path.join(BASE, 'data/HAViD/ActionSegmentation/data/view0_lh_pt')
    
    # Analyze dataset
    stats = analyze_dataset(args.dataset_path, args.split)
    
    # Select holdout classes
    holdout_classes, eligible_classes = select_holdout_classes(
        stats,
        n_frequent=args.n_frequent,
        n_medium=args.n_medium,
        min_test_videos=args.min_test_videos,
        skip_top_n=args.skip_top_n
    )
    
    # Print statistics
    print_class_statistics(stats, holdout_classes)
    
    # Check for data leakage
    no_leakage = check_data_leakage(stats, holdout_classes)
    
    # Compute filtered training set statistics
    filtered_train = compute_filtered_statistics(stats, holdout_classes)
    
    # Output results
    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 80)
    print(f"holdout_classes: {holdout_classes}")
    print(f"\nClass names:")
    for cls in holdout_classes:
        print(f"  {cls}: {stats['index2label'][cls]}")
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"# Holdout classes for zero-shot training\n")
            f.write(f"holdout_classes: {holdout_classes}\n\n")
            f.write(f"# Class details:\n")
            for cls in holdout_classes:
                f.write(f"# {cls}: {stats['index2label'][cls]}\n")
        print(f"\nSaved to: {args.output}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Selected {len(holdout_classes)} classes for holdout")
    print(f"All classes have >={args.min_test_videos} test videos")
    print(f"No data leakage: {no_leakage}")
    print(f"Training set will be reduced from {len(stats['train_videos'])} to {len(filtered_train)} videos")
    print("=" * 80)

if __name__ == "__main__":
    main()


