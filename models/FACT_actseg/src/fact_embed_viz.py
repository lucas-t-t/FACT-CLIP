#!/usr/bin/env python3
"""
Visualise FACT embeddings (visual vs. action branch) on Breakfast.

Improvements over the original script
-------------------------------------
* **Cap the number of points per class** via `--max-per-class` (default = 200)
  to avoid oversampling frequent actions and to keep UMAP reasonably fast.
* **Remove the deterministic seed** from PCA/UMAP so the reducer can exploit
  multi‑threading / GPU back‑ends (e.g. cuML) without serialising the run.
* Cleaned up a few minor style issues; functional logic unchanged.

Example
~~~~~~
python fact_embed_viz.py \
    --logdir FACT_actseg/log/breakfast/split1/breakfast/0 \
    --split val

Add `--max-per-class 100` (or another value) to tighten the cap even further.
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import umap  # cuML-compatible if installed
import numpy as np
from sklearn.decomposition import PCA

# ─────────────────────────── repo setup ──
repo_root = Path(__file__).resolve().parent          # …/FACT_actseg/src
project_root = repo_root.parent                     # …/FACT_actseg
sys.path.insert(0, str(project_root))               # make 'src.' importable

# ───────────────────────────────── helpers ──

def load_cfg(logdir: Path):
    """Load FACT's YACS config dumped at train‑time (see train.py)."""
    from yacs.config import CfgNode as CN
    import json

    args_path = logdir / "args.json"
    cfg_dict = json.load(open(args_path))
    cfg = CN(cfg_dict)
    cfg.freeze()
    return cfg


def load_model(logdir: Path, device="cuda"):
    from src.utils.dataset import create_dataset
    from src.models.blocks import FACT

    cfg = load_cfg(logdir)

    # Infer input dim & #classes from one dataset instance
    _, test_ds = create_dataset(cfg)
    in_dim, n_cls = test_ds.input_dimension, test_ds.nclasses

    model = FACT(cfg, in_dim, n_cls).to(device).eval()

    ckpt_dir = logdir / "ckpts"
    net_files = sorted(ckpt_dir.glob("network.iter-*.net"))
    if not net_files:
        raise FileNotFoundError(f"No *.net files found in {ckpt_dir}")
    state_dict = torch.load(net_files[-1], map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model, cfg


def register_hooks(model):
    """Attach hooks to grab the last‑block outputs of both branches."""
    cache = {}

    def hook(name):
        def _h(_, __, out):
            cache[name] = out.detach().cpu()
        return _h

    last_blk = model.block_list[-1]
    last_blk.frame_branch.register_forward_hook(hook("visual"))
    last_blk.action_branch.register_forward_hook(hook("action"))
    return cache


def get_loader(cfg, split):
    """1‑liner wrapper around FACT's custom DataLoader."""
    from src.utils.dataset import DataLoader, create_dataset

    train_ds, test_ds = create_dataset(cfg)
    ds = {"train": train_ds, "val": test_ds, "test": test_ds}[split]
    return DataLoader(ds, batch_size=1, shuffle=False)


def collect_embeddings(model, loader, cache, device, cap):
    """Run inference and cache up to *cap* items per class."""
    vis, act, lab_v, lab_a, per_cls = [], [], [], [], {}
    with torch.no_grad():
        for sample in loader:
            seqs = [s.to(device) for s in sample[1]]      # sequence list
            label_list = [l.to(device) for l in sample[2]]  # train labels
            labels = sample[3][0]  # eval labels for this sample

            model(seqs, label_list)  # forward pass

            F = cache.pop("visual").squeeze(1)   # (T, d)
            A = cache.pop("action").squeeze(1)   # (M, d)

            # Frame‑wise (visual) – cap per class
            for i, cls in enumerate(labels):
                if per_cls.get(cls, 0) >= cap:
                    continue
                per_cls[cls] = per_cls.get(cls, 0) + 1
                vis.append(F[i])
                lab_v.append(int(cls))

            # Token‑wise (action) – no cap (much fewer points)
            for t in range(A.shape[0]):
                act.append(A[t])
                lab_a.append(int(labels[min(t, len(labels) - 1)]))
    return (
        torch.stack(vis),
        torch.tensor(lab_v),
        torch.stack(act),
        torch.tensor(lab_a),
    )


# ───────────────────────── dimensionality reduction ──

def umap2d(x: np.ndarray, nn: int = 50, md: float = 0.1):
    """PCA‑50 ➜ UMAP‑2 using cosine distance (no random_state)."""
    x50 = PCA(50).fit_transform(x)
    return umap.UMAP(
        n_components=2,
        n_neighbors=nn,
        min_dist=md,
        metric="cosine",
        init="spectral",
    ).fit_transform(x50)


# ─────────────────────────────── plotting ──

def scatter(points, labels, title, out):
    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab20", s=4, alpha=0.8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()


# ────────────────────────────────────── main ──
if __name__ == "__main__":
    DEFAULT_CAP = 200  # frames per class (visual branch)

    ap = argparse.ArgumentParser(description="FACT embedding visualisation (Breakfast)")
    ap.add_argument("--logdir", type=Path, required=True,
                    help="FACT run dir containing args.json & ckpts/")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="Dataset split to visualise (default: val)")
    ap.add_argument("--outdir", type=Path, default="figures",
                    help="Output directory for PNGs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device for inference (cuda or cpu)")
    ap.add_argument("--max-per-class", type=int, default=DEFAULT_CAP,
                    help=f"Maximum frames per class to sample (default: {DEFAULT_CAP})")
    args = ap.parse_args()

    print("Loading model …")
    model, cfg = load_model(args.logdir, args.device)
    cache = register_hooks(model)

    print("Collecting embeddings …")
    loader = get_loader(cfg, args.split)
    V, yV, A, yA = collect_embeddings(
        model, loader, cache, args.device, cap=args.max_per_class
    )
    print("Shapes – visual:", V.shape, "action:", A.shape)

    print("Running UMAP …")
    V2d, A2d = umap2d(V.numpy()), umap2d(A.numpy())

    print("Plotting …")
    scatter(V2d, yV, "FACT – visual branch", args.outdir / "breakfast_visual_umap.png")
    scatter(A2d, yA, "FACT – action branch", args.outdir / "breakfast_action_umap.png")
    print("✓ Done – figures in", args.outdir)

