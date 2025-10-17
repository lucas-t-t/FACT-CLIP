# Holdout Training for Zero-Shot Action Segmentation

## Overview

This document describes the holdout training setup for the FACT model, which enables zero-shot action segmentation evaluation. This serves as a baseline for future open-vocabulary action segmentation methods.

## Concept

**Holdout Training** involves:
1. Selecting a subset of action classes as "unseen" (holdout classes)
2. Removing all training videos that contain any unseen class
3. Training the model only on the remaining "seen" classes
4. Evaluating on all test videos (including those with unseen classes)
5. Measuring how well the model generalizes to unseen classes

This simulates a zero-shot scenario where the model must recognize actions it has never been trained on.

## Selected Holdout Classes

For HAViD view0_lh_pt, we selected 9 classes (6 frequent + 3 medium-frequency):

| ID | Class Name | Type | Train Frames | Test Frames | Test Videos |
|----|------------|------|--------------|-------------|-------------|
| 9  | iglft      | Medium | 2,305 | 781 | 10 |
| 47 | sftg1      | Medium | 2,203 | 1,231 | 10 |
| 49 | sftg2      | Frequent | 4,679 | 1,430 | 9 |
| 51 | sntft      | Frequent | 4,930 | 1,528 | 11 |
| 52 | sntftwn    | Frequent | 6,581 | 2,483 | 7 |
| 55 | sntsb      | Frequent | 6,937 | 2,022 | 12 |
| 65 | sshc3dh    | Medium | 4,004 | 390 | 5 |
| 71 | sspg3dp    | Frequent | 7,717 | 2,404 | 11 |
| 73 | sspn4dp    | Frequent | 8,233 | 2,140 | 10 |

**Selection Criteria:**
- Mix of frequent (top 6 from most common) and medium-frequency (3 from middle third) classes
- All classes appear in ≥5 test videos
- No data leakage: videos containing these classes are completely removed from training

**Training Impact:**
- Holdout classes represent 24.1% of training frames and 29.1% of test frames
- Training set reduced from 161 to 12 videos (7.5% remaining)
- This significant reduction reflects that these frequent actions appear in most HAViD videos

## Configuration

### File: `havid_view0_lh_pt_holdout.yaml`

Key configuration parameters:
```yaml
# Enable holdout mode
holdout_mode: true

# List of class indices to hold out
holdout_classes: [9, 47, 49, 51, 52, 55, 65, 71, 73]

# Other parameters same as full training
dataset: havid_view0_lh_pt
batch_size: 2
epoch: 150
lr: 0.0001
ntoken: 43  # Model still predicts all classes
```

## Usage

### 1. Analyze Dataset and Select Holdout Classes

```bash
cd /path/to/FACT_actseg
conda activate fact

# Analyze dataset and recommend holdout classes
python -m src.utils.analyze_holdout_classes \
    --n_frequent 6 \
    --n_medium 3 \
    --min_test_videos 5 \
    --output holdout_classes.txt
```

### 2. Train with Holdout Classes

```bash
# Train model with holdout classes
python -m src.train \
    --cfg src/configs/havid_view0_lh_pt_holdout.yaml \
    --set aux.gpu 0 aux.mark "holdout_exp1"
```

The training script will:
- Print holdout configuration at startup
- Filter training videos containing any holdout class
- Train only on remaining videos (12 videos for HAViD view0_lh_pt)
- Evaluate on all test videos at specified intervals
- Log separate metrics for seen/unseen/all classes

### 3. Monitor Training

During training, metrics are logged separately:
- `test-metric-all/*`: Performance on all test frames
- `test-metric-seen/*`: Performance on frames with seen classes only
- `test-metric-unseen/*`: Performance on frames with unseen classes only

Key metrics:
- **Acc**: Frame-wise accuracy (excluding background)
- **AccB**: Frame-wise accuracy (including background)
- **F1@0.10, F1@0.25, F1@0.50**: F1-scores at different IoU thresholds

### 4. Analyze Results

```bash
# Analyze a saved checkpoint
python -m src.eval_holdout \
    --checkpoint_path log/exp/run/saves/10000.gz \
    --output_dir log/exp/run/analysis
```

This generates:
- Comprehensive metrics summary (seen vs. unseen)
- Per-class accuracy breakdown
- Zero-shot gap analysis (performance difference)
- JSON report with all metrics
- CSV file with per-class results

### 5. View Detailed Results

Detailed results are automatically saved during training as JSON files:
```
log/exp/run/saves/
├── 2000.gz              # Checkpoint
├── 2000_detailed.json   # Detailed metrics
├── 4000.gz
├── 4000_detailed.json
...
```

## Expected Results

### Baseline Performance

Based on the holdout setup:
- **Seen Classes**: Should achieve reasonable performance (depends on having only 12 training videos)
- **Unseen Classes**: Baseline zero-shot performance (random guessing would be ~1.3% for 75 classes)
- **Zero-Shot Gap**: Significant gap expected between seen and unseen performance

### Important Notes

1. **Limited Training Data**: With only 12 training videos, the model may not reach optimal performance even on seen classes. This is expected and reflects the challenge of the zero-shot scenario.

2. **Model Capacity**: The model still outputs probabilities for all 75 classes (including unseen ones). It has no explicit information about which classes are held out.

3. **Evaluation Fairness**: Test set is unchanged - we evaluate on complete videos to measure true zero-shot capability.

## Implementation Details

### Dataset Filtering (`src/utils/dataset.py`)

```python
def video_contains_holdout_classes(vname, groundTruth_path, label2index, holdout_classes):
    """Check if a video contains any holdout classes"""
    # Load video labels
    # Return True if any frame has a holdout class
```

Key points:
- Videos are filtered BEFORE creating Dataset objects
- Filtering is based on per-frame labels
- If ANY frame contains a holdout class, entire video is removed

### Evaluation Metrics (`src/utils/evaluate.py`)

```python
class Checkpoint():
    def __init__(self, ..., holdout_classes=[], seen_classes=None):
        # Track holdout and seen classes
        
    def _joint_metrics(self, gt_list, pred_list):
        # Compute metrics for:
        # 1. All classes (standard evaluation)
        # 2. Seen classes only
        # 3. Unseen classes only
```

Key features:
- Per-class accuracy tracking
- Separate F1-scores for seen/unseen
- Zero-shot gap computation
- Detailed JSON export

## Comparison with Full Training

| Metric | Full Training | Holdout Training |
|--------|---------------|------------------|
| Training Videos | 161 | 12 (7.5%) |
| Training Classes | 75 | 66 (88%) |
| Evaluation Classes | 75 | 75 (100%) |
| Test Videos | 41 | 41 (100%) |

## Future Improvements

This baseline can be extended with:
1. **Text Embeddings**: Use class names/descriptions for zero-shot recognition
2. **Visual-Linguistic Models**: Leverage CLIP or similar models
3. **Few-Shot Learning**: Provide a few examples of unseen classes
4. **Compositional Learning**: Learn action primitives that combine into new actions
5. **Cross-Dataset Transfer**: Train on one dataset, evaluate on another

## Troubleshooting

### Issue: Training set too small

**Symptom**: Only a few training videos remain after filtering

**Solutions**:
- Reduce number of holdout classes
- Select less frequent classes
- Use different selection criteria (e.g., by activity type)

### Issue: Metrics not showing seen/unseen breakdown

**Symptom**: Only seeing "all classes" metrics

**Check**:
- `holdout_mode` is set to `true` in config
- `holdout_classes` list is not empty
- Dataset filtering is working (check console output at training start)

### Issue: Poor performance on seen classes

**Symptom**: Low accuracy even on seen classes

**Possible causes**:
- Insufficient training data (only 12 videos)
- May need more training epochs
- Consider adjusting learning rate or batch size

## References

- FACT Paper: [Link to paper]
- HAViD Dataset: https://iai-hrc.github.io/ha-vid
- Zero-Shot Learning: Survey papers and related work

## Contact

For questions or issues, please open an issue in the repository or contact the maintainers.

