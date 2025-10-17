# Holdout Training Implementation Summary

## Overview

Successfully implemented zero-shot action segmentation evaluation for the FACT model by introducing holdout training mode. This allows training on a subset of classes and evaluating generalization to unseen classes.

## Implementation Completed

### ✅ 1. Dataset Analysis Tool

**File**: `src/utils/analyze_holdout_classes.py`

- Analyzes dataset to count class occurrences across train/test splits
- Recommends holdout classes based on frequency and test coverage
- Verifies no data leakage between train/test splits
- Computes filtered training set statistics

**Usage**:
```bash
python -m src.utils.analyze_holdout_classes \
    --n_frequent 6 --n_medium 3 --min_test_videos 5
```

### ✅ 2. Configuration Schema Extension

**File**: `src/configs/default.py`

Added new configuration fields:
- `holdout_mode`: Boolean to enable holdout training
- `holdout_classes`: List of class indices to hold out

### ✅ 3. Holdout Configuration File

**File**: `src/configs/havid_view0_lh_pt_holdout.yaml`

Complete configuration for HAViD view0_lh_pt with:
- 9 holdout classes (6 frequent + 3 medium-frequency)
- Classes: [9, 47, 49, 51, 52, 55, 65, 71, 73]
- Reduces training set from 161 to 12 videos (7.5%)
- Comprehensive documentation of holdout classes

### ✅ 4. Dataset Filtering

**File**: `src/utils/dataset.py`

Implemented:
- `video_contains_holdout_classes()`: Checks if video contains any holdout class
- Automatic filtering of training videos with holdout classes
- Preservation of test set (unchanged for fair evaluation)
- Detailed logging of filtering statistics
- Holdout/seen class information passed to Dataset objects

**Key Features**:
- Filters training videos BEFORE creating Dataset objects
- Removes entire videos if ANY frame contains a holdout class
- No modification to test set
- Comprehensive console logging

### ✅ 5. Enhanced Evaluation Metrics

**File**: `src/utils/evaluate.py`

Enhanced `Checkpoint` class with:
- Separate metrics for seen/unseen/all classes
- Per-class accuracy tracking
- Zero-shot gap computation
- F1-scores computed separately for seen and unseen classes
- `save_detailed_results()` method for comprehensive JSON export

**Metrics Computed**:
- `Acc-seen`, `Acc-unseen`: Frame-wise accuracy
- `AccFG-seen`, `AccFG-unseen`: Accuracy excluding background
- `F1@IoU-seen`, `F1@IoU-unseen`: F1-scores at different IoU thresholds
- Per-class: correct, total, accuracy for each class

### ✅ 6. Training Script Updates

**File**: `src/train.py`

Modifications:
- Pass holdout information to Checkpoint initialization
- Separate wandb logging for seen/unseen/all metrics
- Console summary of holdout evaluation results
- Automatic saving of detailed results JSON for holdout experiments
- Holdout information logged at training start

**Logging Structure**:
- `test-metric-all/*`: Performance on all classes
- `test-metric-seen/*`: Performance on seen classes only
- `test-metric-unseen/*`: Performance on unseen classes only

### ✅ 7. Standalone Evaluation Script

**File**: `src/eval_holdout.py`

Features:
- Load and analyze saved checkpoints
- Generate comprehensive metrics summary
- Compute zero-shot gap (seen - unseen performance)
- Export results to JSON and CSV
- Per-class metrics breakdown

**Usage**:
```bash
python -m src.eval_holdout \
    --checkpoint_path log/exp/run/saves/10000.gz \
    --output_dir log/exp/run/analysis
```

### ✅ 8. Documentation

**File**: `src/configs/HOLDOUT_TRAINING_README.md`

Comprehensive documentation including:
- Concept and rationale
- Selected holdout classes with statistics
- Configuration details
- Step-by-step usage guide
- Expected results and baseline performance
- Implementation details
- Troubleshooting guide
- Future improvements

### ✅ 9. Validation Testing

**File**: `test_holdout_setup.py`

Validation script that verifies:
- Configuration loads correctly
- Dataset filtering works as expected
- No holdout classes in training set
- Evaluation checkpoint properly configured

**Test Results**: ✅ All tests passed!
- Configuration: 9 classes held out
- Dataset: 12 training videos (filtered from 161)
- Dataset: 41 test videos (unchanged)
- No data leakage detected

## Selected Holdout Classes

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

**Total**: 24.1% of training frames, 29.1% of test frames

## Training Data Impact

- **Original training videos**: 161
- **Videos with holdout classes**: 149 (92.5%)
- **Remaining training videos**: 12 (7.5%)
- **Test videos**: 41 (unchanged)
- **Original classes**: 75
- **Seen classes**: 66 (88%)
- **Unseen classes**: 9 (12%)

**Note**: The significant reduction in training data reflects that these frequent actions appear in most HAViD videos. This is expected for zero-shot evaluation.

## How to Use

### 1. Quick Validation

```bash
cd /path/to/FACT_actseg
conda activate fact
python test_holdout_setup.py
```

### 2. Start Training

```bash
python -m src.train \
    --cfg src/configs/havid_view0_lh_pt_holdout.yaml \
    --set aux.gpu 0
```

### 3. Monitor Metrics

During training, watch for:
- `test-metric-seen/Acc`: Accuracy on seen classes
- `test-metric-unseen/Acc`: Accuracy on unseen classes (zero-shot)
- `test-metric-all/F1@0.50`: Overall F1-score

### 4. Analyze Results

```bash
python -m src.eval_holdout \
    --checkpoint_path log/<exp>/<run>/saves/10000.gz
```

## Files Modified/Created

### Created Files:
1. `src/utils/analyze_holdout_classes.py` (new)
2. `src/configs/havid_view0_lh_pt_holdout.yaml` (new)
3. `src/eval_holdout.py` (new)
4. `src/configs/HOLDOUT_TRAINING_README.md` (new)
5. `test_holdout_setup.py` (new)
6. `HOLDOUT_IMPLEMENTATION_SUMMARY.md` (new, this file)

### Modified Files:
1. `src/configs/default.py` - Added holdout_mode and holdout_classes fields
2. `src/utils/dataset.py` - Added filtering logic and video_contains_holdout_classes()
3. `src/utils/evaluate.py` - Enhanced Checkpoint with seen/unseen metrics
4. `src/train.py` - Updated to pass holdout info and log separate metrics

## Key Features

1. **Clean Separation**: Videos with unseen classes completely removed from training
2. **Fair Evaluation**: Test set unchanged, evaluates on complete videos
3. **Comprehensive Metrics**: Separate tracking for seen/unseen/all classes
4. **Per-Class Analysis**: Detailed accuracy for each class
5. **Zero-Shot Gap**: Quantifies performance difference between seen and unseen
6. **Detailed Logging**: JSON export with all metrics and per-video results
7. **Validation Tools**: Automated testing to verify correct implementation

## Expected Behavior

### Training Output:
```
================================================================================
HOLDOUT MODE ENABLED
================================================================================
Holdout classes: [9, 47, 49, 51, 52, 55, 65, 71, 73]
Holdout class names: ['iglft', 'sftg1', ...]
Original training videos: 161
Videos removed (contain holdout classes): 149
Remaining training videos: 12 (7.5%)
================================================================================
```

### Evaluation Output:
```
================================================================================
HOLDOUT EVALUATION SUMMARY
================================================================================
Seen classes: 66
Unseen (holdout) classes: 9
Accuracy (seen): XX.X%
Accuracy (unseen): XX.X%
F1@0.50 (seen): XX.X%
F1@0.50 (unseen): XX.X%
================================================================================
```

## Important Notes

1. **Limited Training Data**: With only 12 training videos, model performance may be limited even on seen classes
2. **Model Capacity**: Model still outputs probabilities for all 75 classes (including unseen)
3. **No Explicit Holdout Signal**: Model has no information about which classes are held out
4. **Baseline Expectation**: Unseen class performance provides zero-shot baseline for future improvements

## Next Steps

1. **Run Training**: Start training with the holdout configuration
2. **Monitor Convergence**: Check if model trains successfully with limited data
3. **Analyze Results**: Use eval_holdout.py to analyze zero-shot performance
4. **Compare with Full Training**: Compare with baseline trained on all classes
5. **Iterate**: Adjust holdout classes or training parameters if needed

## Future Enhancements

- Text embeddings for class descriptions
- Visual-linguistic models (CLIP integration)
- Few-shot learning extensions
- Cross-dataset transfer
- Compositional action understanding

## Validation Status

✅ **All validation tests passed**
- Configuration loads correctly
- Dataset filtering works as expected  
- No holdout classes in training set
- Evaluation properly configured
- Ready for training!

## Contact

For questions or issues with the holdout training implementation, please refer to:
- `src/configs/HOLDOUT_TRAINING_README.md` for detailed usage
- Run `python test_holdout_setup.py` to verify your setup
- Check console logs during training for filtering statistics

