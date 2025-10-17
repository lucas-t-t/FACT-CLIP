#!/usr/bin/env python3
# fact_embed_viz.py
"""
Visualise FACT embeddings (visual vs. action branch) on Breakfast.

Usage:
    python fact_embed_viz.py --logdir <LOG_DIR> --split val
"""

import argparse, gzip, json, os, sys, numba
from pathlib import Path
import torch
import numpy as np
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────── helpers ──
def load_cfg_and_model(logdir: Path, device='cuda'):
    """Rebuild FACT with the exact args & checkpoint saved in <logdir>."""
    cfg = json.load(open(logdir / "breakfast/split1/breakfast/0/args.json"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # repo root on PYTHONPATH
    from src.models.fact import FACT  # noqa: E402

    model = FACT(**cfg["model"]).to(device).eval()

    with gzip.open(logdir / "best_ckpt.gz", "rb") as f:
        ckpt = torch.load(f, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    return model, cfg


def register_hooks(model):
    """Return dict that fills with tensors from the last visual & action layers."""
    cache = {}

    def hook(name):
        def _hook(_, __, out):
            cache[name] = out.detach().cpu()
        return _hook

    # *These attribute names match the original repo.
    #   If you've changed them, print(model) and adjust here.*
    model.frame_branch[-1].register_forward_hook(hook("visual"))
    model.action_branch[-1].register_forward_hook(hook("action"))
    return cache


def get_loader(cfg, split):
    from src.dataset.breakfast import Breakfast  # noqa: E402
    ds = Breakfast(split=cfg["data"]["split"], mode=split)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)


def collect_embeddings(model, loader, cache, device, max_per_class=None):
    vis, act = [], []
    lab_vis, lab_act = [], []

    per_cls_counter = {}

    with torch.no_grad():
        for sample in loader:
            feat = sample["feat"].to(device)          # (1, T, C)
            labels = sample["label"][0]               # (T,)
            model(feat)                               # fills cache

            F = cache.pop("visual").squeeze(0)        # (T, d)
            A = cache.pop("action").squeeze(0)        # (M, d)

            # ── Down-sample frames if requested (keeps distribution balanced)
            for idx, cls in enumerate(labels.tolist()):
                if max_per_class:
                    if per_cls_counter.get(cls, 0) >= max_per_class:
                        continue
                    per_cls_counter[cls] = per_cls_counter.get(cls, 0) + 1
                vis.append(F[idx])
                lab_vis.append(cls)

            # Map each action token to its dominant class by the loader’s mapping,
            # or fallback to majority frame label if unavailable.
            token2cls = sample.get("token2frame")      # (M,) or None
            if token2cls is not None:
                for t, frame_idx in enumerate(token2cls[0].tolist()):
                    act.append(A[t])
                    lab_act.append(labels[frame_idx].item())
            else:
                # simple majority vote
                maj = torch.mode(labels).values.item()
                act.append(A.mean(0))
                lab_act.append(maj)

    return torch.stack(vis), torch.tensor(lab_vis), \
           torch.stack(act), torch.tensor(lab_act)


def umap_2d(x, seed=42, n_neighbors=50, min_dist=0.1):
    numba.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"

    x50 = PCA(n_components=50, random_state=seed).fit_transform(x)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        init="spectral",
        random_state=seed,
    )
    return reducer.fit_transform(x50)


def plot_scatter(points, labels, title, path):
    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(points[:, 0], points[:, 1],
                c=labels, cmap="tab20", s=4, alpha=0.8)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────── main script ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, type=Path,
                    help="FACT log dir containing args.json & best_ckpt.gz")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--outdir", default="figures", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="cap #frames per class (for speed)")
    args = ap.parse_args()

    print("Loading model …")
    model, cfg = load_cfg_and_model(args.logdir, args.device)
    cache = register_hooks(model)

    print("Collecting embeddings …")
    loader = get_loader(cfg, args.split)
    V, yV, A, yA = collect_embeddings(
        model, loader, cache, args.device, args.max_per_class)

    print(f"Visual embeddings shape:  {V.shape}")
    print(f"Action  embeddings shape: {A.shape}")

    print("Running UMAP …")
    V_2d = umap_2d(V.numpy())
    A_2d = umap_2d(A.numpy())

    print("Plotting …")
    plot_scatter(V_2d, yV, "FACT – visual branch", args.outdir / "breakfast_visual_umap.png")
    plot_scatter(A_2d, yA, "FACT – action branch", args.outdir / "breakfast_action_umap.png")

    print("✓ Done. Figures saved in", args.outdir)


if __name__ == "__main__":
    main()
