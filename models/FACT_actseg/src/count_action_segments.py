#!/usr/bin/env python3
"""
Script to count action segments in HAViD ground truth files.
An action segment is defined as a contiguous sequence of frames with the same label.
Every label change creates a new segment.
"""

import os
import json
from collections import defaultdict
from src.home import get_project_base


def count_segments_in_file(filepath):
    """
    Count the number of action segments in a single ground truth file.
    
    Args:
        filepath: Path to the ground truth file
        
    Returns:
        Number of action segments (label transitions)
    """
    segments = 0
    previous_label = None
    
    with open(filepath, 'r') as f:
        for line in f:
            current_label = line.strip()
            
            # Count a new segment when the label changes
            if current_label != previous_label:
                segments += 1
                previous_label = current_label
    
    return segments


def main():
    # Get the project base directory
    project_base = get_project_base()
    
    # Construct path to ground truth directory
    gt_dir = os.path.join(
        project_base,
        'data/HAViD/ActionSegmentation/data/view0_lh_pt/groundTruth'
    )
    
    # Dictionary to store segment count for each file
    file_segment_counts = {}
    
    # Process each file in the directory
    for filename in sorted(os.listdir(gt_dir)):
        # Skip hidden files (starting with '.')
        if filename.startswith('.'):
            continue
        
        # Only process .txt files
        if not filename.endswith('.txt'):
            continue
        
        filepath = os.path.join(gt_dir, filename)
        
        # Count segments in this file
        segment_count = count_segments_in_file(filepath)
        file_segment_counts[filename] = segment_count
    
    # Calculate statistics
    if not file_segment_counts:
        print("No files found to process!")
        return
    
    # Create distribution: how many files have each segment count
    distribution = defaultdict(int)
    for count in file_segment_counts.values():
        distribution[count] += 1
    
    # Calculate average, min, max
    all_counts = list(file_segment_counts.values())
    average = sum(all_counts) / len(all_counts)
    min_count = min(all_counts)
    max_count = max(all_counts)
    
    # Print distribution (sorted by segment count)
    print("\n=== Action Segment Distribution ===")
    for segment_count in sorted(distribution.keys()):
        num_files = distribution[segment_count]
        print(f"{segment_count} action segments: {num_files} files")
    
    # Print statistics
    print(f"\nAverage of action segments per file: {average:.2f}")
    print(f"Highest number of action segments in a file: {max_count}")
    print(f"Lowest number of action segments in a file: {min_count}")
    print(f"\nTotal files processed: {len(file_segment_counts)}")
    
    # Prepare data for JSON export
    output_data = {
        "distribution": {str(k): v for k, v in sorted(distribution.items())},
        "per_file_counts": file_segment_counts,
        "statistics": {
            "average": round(average, 2),
            "min": min_count,
            "max": max_count,
            "total_files": len(file_segment_counts)
        }
    }
    
    # Export to JSON
    json_output_path = os.path.join(gt_dir, 'action_segment_stats.json')
    with open(json_output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults exported to: {json_output_path}")


if __name__ == "__main__":
    main()

