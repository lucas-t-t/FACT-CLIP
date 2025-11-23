#!/usr/bin/env python3
"""
Print training summary from a saved training session.

Usage:
    python -m src.print_training_summary --logdir log/havid_view0_lh_pt/split1/havid_view0_lh_pt_view0_lh_pt_clean_eval_1000_ntoken_72/0
    python -m src.print_training_summary --checkpoint log/havid_view0_lh_pt/split1/havid_view0_lh_pt_view0_lh_pt_clean_eval_1000_ntoken_72/0/best_ckpt.gz
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from utils.evaluate import Checkpoint


def find_best_checkpoint(logdir):
    """Find the best checkpoint file in the log directory."""
    best_ckpt_path = os.path.join(logdir, 'best_ckpt.gz')
    if os.path.exists(best_ckpt_path):
        return best_ckpt_path
    
    # If best_ckpt.gz doesn't exist, try to find the latest checkpoint
    saves_dir = os.path.join(logdir, 'saves')
    if os.path.exists(saves_dir):
        checkpoint_files = [f for f in os.listdir(saves_dir) if f.endswith('.gz')]
        if checkpoint_files:
            # Sort by iteration number
            checkpoint_files.sort(key=lambda x: int(x.replace('.gz', '')))
            return os.path.join(saves_dir, checkpoint_files[-1])
    
    return None


def load_metrics_from_checkpoints(logdir):
    """Load metrics history from checkpoint files."""
    saves_dir = os.path.join(logdir, 'saves')
    if not os.path.exists(saves_dir):
        return None, None
    
    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(saves_dir) if f.endswith('.gz')]
    if not checkpoint_files:
        return None, None
    
    # Sort by iteration number
    checkpoint_files.sort(key=lambda x: int(x.replace('.gz', '')))
    
    # Load metrics from each checkpoint
    test_metrics_history = {}
    train_metrics_history = {}
    # Don't create summary here - we'll use the best checkpoint for that
    
    for ckpt_file in checkpoint_files:
        try:
            ckpt_path = os.path.join(saves_dir, ckpt_file)
            ckpt = Checkpoint.load(ckpt_path)
            
            if not hasattr(ckpt, 'metrics') or len(ckpt.metrics) == 0:
                ckpt.compute_metrics()
            
            iteration = ckpt.iteration
            
            # Store test metrics (from evaluation checkpoints) for history only
            for key, value in ckpt.metrics.items():
                if '-seen' not in key and '-unseen' not in key:
                    metric_key = f'test-metric-all/{key}'
                    if metric_key not in test_metrics_history:
                        test_metrics_history[metric_key] = []
                    test_metrics_history[metric_key].append((iteration, value))
        
        except Exception:
            continue
    
    # Combine histories
    all_history = {}
    all_history.update(test_metrics_history)
    all_history.update(train_metrics_history)
    
    return all_history, None  # Return None for summary - we'll use best checkpoint


def read_wandb_run_file(run_file_path):
    """Read wandb run file to extract history and final values."""
    try:
        import struct
        try:
            from wandb.proto import wandb_internal_pb2
        except ImportError:
            # Try alternative import
            try:
                import wandb.sdk.wandb_internal_pb2 as wandb_internal_pb2
            except ImportError:
                return None
        
        history_data = {}
        summary_data = {}
        max_step = 0
        final_train_loss = None
        final_train_metrics = {}
        final_iter = None
        records_read = 0
        records_parsed = 0
        
        with open(run_file_path, 'rb') as f:
            while True:
                length_bytes = f.read(4)
                if len(length_bytes) < 4:
                    break
                
                length = struct.unpack('<I', length_bytes)[0]
                if length == 0 or length > 10000000:  # 10MB max
                    break
                
                records_read += 1
                try:
                    data = f.read(length)
                    if len(data) < length:
                        break
                    
                    pb = wandb_internal_pb2.Record()
                    pb.ParseFromString(data)
                    records_parsed += 1
                    
                    record_type = pb.WhichOneof('record_type')
                    
                    if record_type == 'history':
                        step = pb.history.step.num if pb.history.step else 0
                        max_step = max(max_step, step)
                        final_iter = max_step
                        
                        for item in pb.history.item:
                            key = item.key
                            try:
                                value = json.loads(item.value_json)
                                if key not in history_data:
                                    history_data[key] = []
                                history_data[key].append((step, value))
                                
                                # Track final training metrics (keep updating with latest)
                                if key.startswith('train-metric/'):
                                    metric_name = key.replace('train-metric/', '')
                                    final_train_metrics[metric_name] = value
                                elif key == 'train-loss/loss':
                                    final_train_loss = value
                                    
                            except Exception as e:
                                # Skip bad JSON
                                pass
                    
                    elif record_type == 'summary':
                        for item in pb.summary.update.item:
                            key = item.key
                            try:
                                value = json.loads(item.value_json)
                                summary_data[key] = value
                            except:
                                pass
                                
                except Exception as e:
                    # Skip malformed records
                    continue
        
        # Only return if we parsed at least some records
        if records_parsed > 0:
            return {
                'history': history_data,
                'summary': summary_data,
                'final_iter': final_iter,
                'final_train_loss': final_train_loss,
                'final_train_metrics': final_train_metrics
            }
        return None
    except Exception as e:
        # Return None on any error
        return None


def load_wandb_data(logdir):
    """Try to load wandb run history and summary from wandb run file or checkpoints."""
    # First try to load from checkpoint files (more reliable for test metrics)
    history, _ = load_metrics_from_checkpoints(logdir)
    
    # Try to read from wandb file if available
    wandb_dir = os.path.join(logdir, 'wandb')
    summary = None
    wandb_run_data = None
    wandb_run_dir = None
    
    if os.path.exists(wandb_dir):
        # Find wandb run file and directory
        run_file = None
        for item in os.listdir(wandb_dir):
            item_path = os.path.join(wandb_dir, item)
            if os.path.isdir(item_path) and (item.startswith('offline-run-') or item == 'latest-run'):
                for file_item in os.listdir(item_path):
                    if file_item.startswith('run-') and file_item.endswith('.wandb'):
                        run_file = os.path.join(item_path, file_item)
                        wandb_run_dir = item_path
                        break
                if run_file:
                    break
        
        # Try to read wandb run file
        if run_file and os.path.exists(run_file):
            wandb_run_data = read_wandb_run_file(run_file)
            if wandb_run_data:
                # Use summary from run file if available
                if wandb_run_data['summary']:
                    summary = wandb_run_data['summary']
                # Merge with history from run file
                if wandb_run_data['history']:
                    # Merge wandb history with checkpoint history
                    if history is None:
                        history = {}
                    for k, v in wandb_run_data['history'].items():
                        if k not in history:
                            history[k] = []
                        history[k].extend(v)
                    # Sort each metric's history by step
                    for k in history:
                        history[k].sort(key=lambda x: x[0])
        
        # Fallback: Try to get summary from JSON file
        if summary is None:
            summary_paths = [
                os.path.join(wandb_dir, 'latest-run', 'files', 'wandb-summary.json'),
            ]
            
            for item in os.listdir(wandb_dir):
                if item.startswith('offline-run-'):
                    summary_paths.append(
                        os.path.join(wandb_dir, item, 'files', 'wandb-summary.json')
                    )
            
            for summary_path in summary_paths:
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, 'r') as f:
                            wandb_summary = json.load(f)
                            summary = {}
                            for k, v in wandb_summary.items():
                                if not k.startswith('_wandb'):
                                    summary[k] = v
                            break
                    except:
                        pass
    
    return history, summary, wandb_run_data, wandb_run_dir


def generate_ascii_chart(values, width=20):
    """Generate ASCII chart from values."""
    if not values or len(values) == 0:
        return ""
    
    # Extract numeric values
    nums = []
    for v in values:
        if isinstance(v, (int, float)):
            nums.append(v)
        elif isinstance(v, tuple) and len(v) >= 2:
            nums.append(v[1])  # Assume (step, value) tuple
        else:
            nums.append(float(v) if isinstance(v, str) else 0)
    
    if not nums:
        return ""
    
    min_val = min(nums)
    max_val = max(nums)
    
    if max_val == min_val:
        return "█" * width
    
    # Normalize to 0-1 range
    normalized = [(v - min_val) / (max_val - min_val) for v in nums]
    
    # Map to characters
    chars = "▁▂▃▄▅▆▇█"
    chart = ""
    for n in normalized:
        idx = int(n * (len(chars) - 1))
        chart += chars[idx]
    
    return chart


def find_latest_checkpoint(logdir):
    """Find the latest checkpoint file in saves directory."""
    saves_dir = os.path.join(logdir, 'saves')
    if os.path.exists(saves_dir):
        checkpoint_files = [f for f in os.listdir(saves_dir) if f.endswith('.gz')]
        if checkpoint_files:
            # Sort by iteration number
            checkpoint_files.sort(key=lambda x: int(x.replace('.gz', '')))
            return os.path.join(saves_dir, checkpoint_files[-1])
    return None


def print_summary(ckpt_path, logdir=None, show_wandb=True):
    """Print training summary in the same format as training output."""
    
    # Load the checkpoint (best test checkpoint)
    ckpt = Checkpoint.load(ckpt_path)
    
    # Ensure metrics are computed
    if not hasattr(ckpt, 'metrics') or len(ckpt.metrics) == 0:
        ckpt.compute_metrics()
    
    # Try to get final training iteration, loss, and training metrics from wandb
    final_iter = ckpt.iteration
    loss_val = None
    train_metrics = None
    
    if logdir and show_wandb:
        _, wandb_summary, wandb_run_data, wandb_run_dir = load_wandb_data(logdir)
        
        # Get data from wandb run file if available
        if wandb_run_data:
            if wandb_run_data['final_iter']:
                final_iter = wandb_run_data['final_iter']
            if wandb_run_data['final_train_loss'] is not None:
                loss_val = wandb_run_data['final_train_loss']
            if wandb_run_data['final_train_metrics']:
                train_metrics = wandb_run_data['final_train_metrics']
        
        # Fallback to summary file if run file data not available
        if train_metrics is None and wandb_summary:
            # Get training metrics from summary
            train_metrics = {}
            for k, v in wandb_summary.items():
                if k.startswith('train-metric/'):
                    metric_name = k.replace('train-metric/', '')
                    if isinstance(v, dict):
                        train_metrics[metric_name] = v.get('value', v.get('_value', v))
                    else:
                        train_metrics[metric_name] = v
            
            # Get training loss from summary
            if loss_val is None:
                train_loss_key = None
                for k in wandb_summary.keys():
                    if k.startswith('train-loss/loss'):
                        train_loss_key = k
                        break
                if train_loss_key and train_loss_key in wandb_summary:
                    loss_val = wandb_summary[train_loss_key]
                    if isinstance(loss_val, dict):
                        loss_val = loss_val.get('value', loss_val.get('_value', None))
        
        # Try to infer final iteration from latest checkpoint if not found
        if final_iter == ckpt.iteration and logdir:
            latest_ckpt_path = find_latest_checkpoint(logdir)
            if latest_ckpt_path:
                try:
                    latest_ckpt = Checkpoint.load(latest_ckpt_path)
                    # Final iteration is usually a bit after the last test checkpoint
                    # But we'll use the latest checkpoint iteration as approximation
                    final_iter = latest_ckpt.iteration
                except:
                    pass
    
    # Format output similar to training
    # Use training metrics if available, otherwise use test metrics
    if train_metrics:
        # Use training metrics for the first metrics line
        metrics_to_show = train_metrics
    else:
        # Fallback to test metrics if training metrics not available
        metrics_to_show = ckpt.metrics
    
    if loss_val is not None:
        print(f"Iter{final_iter}, loss:{loss_val:.1f},")
    else:
        print(f"Iter{final_iter},")
    
    # Format metrics - order: AccB, Acc, F1@0.10, F1@0.25, F1@0.50
    metrics_str = "           "
    metric_order = ['AccB', 'Acc', 'F1@0.10', 'F1@0.25', 'F1@0.50']
    
    for metric_name in metric_order:
        if metric_name in metrics_to_show:
            val = metrics_to_show[metric_name]
            metrics_str += f"{metric_name}:{val:.3f}, "
    
    # Add Edit metric if available (usually only in test metrics)
    if 'Edit' in metrics_to_show:
        metrics_str += f"Edit:{metrics_to_show['Edit']:.3f}, "
    elif 'Edit' in ckpt.metrics:
        metrics_str += f"Edit:{ckpt.metrics['Edit']:.3f}, "
    
    # Add any other metrics not in the standard order (but skip seen/unseen variants)
    for k, v in sorted(metrics_to_show.items()):
        if k not in metric_order and k != 'Edit' and '-seen' not in k and '-unseen' not in k:
            metrics_str += f"{k}:{v:.3f}, "
    
    metrics_str = metrics_str.rstrip(', ') + ",\n"
    print(metrics_str)
    
    # Print best checkpoint info
    best_iter = ckpt.iteration
    if logdir:
        best_ckpt_path = os.path.join(logdir, 'best_ckpt.gz')
        if os.path.exists(best_ckpt_path):
            try:
                best_ckpt = Checkpoint.load(best_ckpt_path)
                best_iter = best_ckpt.iteration
            except:
                pass
    
    print(f"Best Checkpoint: {best_iter}")
    
    # Try to load and print wandb summary if requested
    if show_wandb and logdir:
        wandb_history, wandb_summary_from_file, _, wandb_run_dir = load_wandb_data(logdir)
        
        # Start with wandb summary from file (contains both test and train metrics)
        wandb_summary = {}
        if wandb_summary_from_file:
            wandb_summary.update(wandb_summary_from_file)
        
        # Use best checkpoint metrics for test metrics in summary (primary)
        best_ckpt_for_summary = ckpt  # Default to current ckpt
        if logdir:
            best_ckpt_path = os.path.join(logdir, 'best_ckpt.gz')
            if os.path.exists(best_ckpt_path):
                try:
                    best_ckpt_for_summary = Checkpoint.load(best_ckpt_path)
                    if not hasattr(best_ckpt_for_summary, 'metrics') or len(best_ckpt_for_summary.metrics) == 0:
                        best_ckpt_for_summary.compute_metrics()
                except:
                    pass
        
        # Add best checkpoint test metrics to summary (primary - what we compare)
        for key, value in best_ckpt_for_summary.metrics.items():
            if '-seen' not in key and '-unseen' not in key:
                metric_key = f'test-metric-all/{key}'
                wandb_summary[metric_key] = value
        
        # Also get last checkpoint metrics for reference
        last_ckpt_metrics = None
        if logdir:
            latest_ckpt_path = find_latest_checkpoint(logdir)
            if latest_ckpt_path:
                try:
                    last_ckpt = Checkpoint.load(latest_ckpt_path)
                    if not hasattr(last_ckpt, 'metrics') or len(last_ckpt.metrics) == 0:
                        last_ckpt.compute_metrics()
                    last_ckpt_metrics = {}
                    for key, value in last_ckpt.metrics.items():
                        if '-seen' not in key and '-unseen' not in key:
                            metric_key = f'test-metric-all/{key}'
                            last_ckpt_metrics[metric_key] = value
                except:
                    pass
        
        if wandb_history or wandb_summary:
            print("\nwandb:")
            print("wandb:")
            
            # Print run history with ASCII charts
            if wandb_history:
                print("wandb: Run history:")
                
                # Order metrics: test metrics first, then train metrics
                test_metrics = [k for k in wandb_history.keys() if k.startswith('test-metric')]
                train_metrics = [k for k in wandb_history.keys() if k.startswith('train-')]
                
                # Sort within each group
                test_metrics.sort()
                train_metrics.sort()
                
                all_metrics = test_metrics + train_metrics
                
                for metric_key in all_metrics:
                    values = wandb_history[metric_key]
                    chart = generate_ascii_chart(values, width=20)
                    if chart:
                        # Format metric name (pad to align)
                        metric_name = metric_key
                        print(f"wandb:     {metric_name:<30} {chart}")
            
            # Print run summary
            if wandb_summary:
                print("wandb:")
                print("wandb: Run summary:")
                
                # Filter and format wandb metrics similar to training output
                metric_keys = []
                for k in sorted(wandb_summary.keys()):
                    if not k.startswith('_wandb'):
                        metric_keys.append(k)
                
                # Group by prefix - order: test metrics first, then train metrics
                test_metrics = [k for k in metric_keys if k.startswith('test-metric')]
                train_metrics = [k for k in metric_keys if k.startswith('train-')]
                other_metrics = [k for k in metric_keys if k not in test_metrics + train_metrics]
                
                # Print test metrics first
                for k in test_metrics:
                    v = wandb_summary[k]
                    if isinstance(v, (int, float)):
                        print(f"wandb:     {k} {v:.5f}")
                    elif isinstance(v, dict):
                        val = v.get('value', v.get('_value', v))
                        if isinstance(val, (int, float)):
                            print(f"wandb:     {k} {val:.5f}")
                        else:
                            print(f"wandb:     {k} {val}")
                    else:
                        print(f"wandb:     {k} {v}")
                
                # Print train metrics
                for k in train_metrics:
                    v = wandb_summary[k]
                    if isinstance(v, (int, float)):
                        print(f"wandb:     {k} {v:.5f}")
                    elif isinstance(v, dict):
                        val = v.get('value', v.get('_value', v))
                        if isinstance(val, (int, float)):
                            print(f"wandb:     {k} {val:.5f}")
                        else:
                            print(f"wandb:     {k} {val}")
                    else:
                        print(f"wandb:     {k} {v}")
                
                # Print other metrics
                for k in other_metrics:
                    v = wandb_summary[k]
                    if isinstance(v, (int, float)):
                        print(f"wandb:     {k} {v:.5f}")
                    elif isinstance(v, dict):
                        val = v.get('value', v.get('_value', v))
                        if isinstance(val, (int, float)):
                            print(f"wandb:     {k} {val:.5f}")
                        else:
                            print(f"wandb:     {k} {val}")
                    else:
                        print(f"wandb:     {k} {v}")
            
            # Show last checkpoint metrics for reference (if different from best)
            if last_ckpt_metrics and best_ckpt_for_summary.iteration != last_ckpt.iteration:
                print("wandb:")
                print("wandb: Last checkpoint test metrics (for reference):")
                for k in sorted(last_ckpt_metrics.keys()):
                    v = last_ckpt_metrics[k]
                    if isinstance(v, (int, float)):
                        print(f"wandb:     {k} {v:.5f}")
            
            print("wandb:")
            
            # Add wandb logs path
            if wandb_run_dir:
                logs_path = os.path.join(wandb_run_dir, 'logs')
                if os.path.exists(logs_path):
                    print(f"wandb: Find logs at: {logs_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Print training summary from a saved training session"
    )
    parser.add_argument(
        '--logdir',
        type=str,
        default=None,
        help='Path to log directory (e.g., log/dataset/split/exp/run)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to specific checkpoint file (.gz). If not provided, will use best_ckpt.gz from logdir'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Do not try to load wandb summary'
    )
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint:
        ckpt_path = args.checkpoint
        logdir = os.path.dirname(args.checkpoint) if not args.logdir else args.logdir
    elif args.logdir:
        ckpt_path = find_best_checkpoint(args.logdir)
        if ckpt_path is None:
            print(f"Error: Could not find checkpoint in {args.logdir}")
            print("Please specify --checkpoint or ensure logdir contains best_ckpt.gz or saves/ directory")
            sys.exit(1)
        logdir = args.logdir
    else:
        parser.print_help()
        sys.exit(1)
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file not found: {ckpt_path}")
        sys.exit(1)
    
    print_summary(ckpt_path, logdir=logdir, show_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()

