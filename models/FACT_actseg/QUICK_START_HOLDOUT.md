# Quick Start: Holdout Training

## TL;DR - Run This

```bash
# 1. Activate environment
cd /cvhci/temp/lthomaz/models/FACT_actseg
conda activate fact

# 2. Validate setup (optional but recommended)
python test_holdout_setup.py

# 3. Start training
python -m src.train \
    --cfg src/configs/havid_view0_lh_pt_holdout.yaml \
    --set aux.gpu 0 aux.mark "holdout_baseline"

# 4. Monitor training (in another terminal)
tail -f log/havid_view0_lh_pt/holdout_baseline/wandb/latest-run/run-*.log

# 5. Analyze results after training
python -m src.eval_holdout \
    --checkpoint_path log/havid_view0_lh_pt/holdout_baseline/saves/10000.gz
```

## What This Does

- **Trains** FACT model on HAViD view0_lh_pt dataset
- **Holds out** 9 classes (never seen during training)
- **Evaluates** on all classes including unseen ones
- **Measures** zero-shot generalization capability

## Key Metrics to Watch

During training, look for:
- `test-metric-seen/Acc`: Performance on seen classes
- `test-metric-unseen/Acc`: **Zero-shot performance** on unseen classes
- Gap between seen and unseen indicates generalization challenge

## Holdout Classes

9 classes held out (6 frequent + 3 medium):
- `9:iglft`, `47:sftg1`, `49:sftg2`, `51:sntft`, `52:sntftwn`
- `55:sntsb`, `65:sshc3dh`, `71:sspg3dp`, `73:sspn4dp`

## Training Data

- **Before filtering**: 161 training videos
- **After filtering**: 12 training videos (7.5%)
- **Test set**: 41 videos (unchanged)

## Files Created

- **Config**: `src/configs/havid_view0_lh_pt_holdout.yaml`
- **Analysis**: `src/utils/analyze_holdout_classes.py`
- **Evaluation**: `src/eval_holdout.py`
- **Test**: `test_holdout_setup.py`
- **Docs**: `src/configs/HOLDOUT_TRAINING_README.md`
- **Summary**: `HOLDOUT_IMPLEMENTATION_SUMMARY.md`

## Need Help?

1. **Setup issues**: Run `python test_holdout_setup.py`
2. **Training details**: See `src/configs/HOLDOUT_TRAINING_README.md`
3. **Implementation details**: See `HOLDOUT_IMPLEMENTATION_SUMMARY.md`

## Expected Runtime

- **Training**: ~6-12 hours (depends on GPU, 150 epochs, 12 videos)
- **Evaluation**: ~1-2 minutes per checkpoint

## Troubleshooting

**Issue**: "No training videos remaining"
- **Fix**: Reduce number of holdout classes or select less frequent ones

**Issue**: "CUDA out of memory"
- **Fix**: Reduce batch_size in config (currently set to 2)

**Issue**: Metrics not showing seen/unseen breakdown
- **Fix**: Ensure `holdout_mode: true` in config file

## What's Next?

After training completes:
1. Analyze zero-shot gap (seen vs. unseen performance)
2. Compare with full training baseline (all 75 classes)
3. Use as baseline for open-vocabulary implementations
4. Experiment with different holdout class selections

