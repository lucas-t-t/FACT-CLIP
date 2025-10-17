#!/usr/bin/env python3
# fact_embed_viz.py
"""
Visualise FACT embeddings (visual vs. action branch) on Breakfast.

Example:
    python fact_embed_viz.py \
        --logdir FACT_actseg/log/breakfast/split1/breakfast/0 \
        --split val
"""
import argparse, gzip, json, os, sys, pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numba, torch, umap, yaml
import numpy as np
from sklearn.decomposition import PCA

# ───────────────────────────────────────────────── helpers ──
def load_cfg(logdir: Path):
    """Re-create the YACS CfgNode that FACT expects."""
    from yacs.config import CfgNode as CN
    from configs.default import get_cfg_defaults          # repo file
    # args.json that was dumped after training
    arg_file = logdir / "args.json"
    args_dict = json.load(open(arg_file))
    # path to the YAML you trained with
    cfg_file = Path(args_dict["cfg_file"][0]).resolve()
    cfg_yaml = CN(yaml.safe_load(open(cfg_file)))
    cfg = get_cfg_defaults()
    cfg.merge_from_other_cfg(cfg_yaml)
    cfg.merge_from_other_cfg(CN(args_dict))               # command-line “--set …”
    return cfg

def load_model(logdir: Path, device="cuda"):
    """
    1. rebuild FACT
    2. load the *network* weight file (ckpts/network.iter-XXXX.net)
    """
    repo_root = Path(__file__).resolve().parent           # FACT_actseg/src
    sys.path.insert(0, str(repo_root))                    # put repo on PYTHONPATH

    cfg = load_cfg(logdir)

    # ── create a dataset once to learn the input dim & #classes
    from utils.dataset import create_dataset              # repo file
    _, test_ds = create_dataset(cfg)                      # we only need meta-info
    in_dim, n_cls = test_ds.input_dimension, test_ds.nclasses

    from models.blocks import FACT                        # FACT lives here
    model = FACT(cfg, in_dim, n_cls).to(device).eval()

    # pick the latest (or best) .net file
    ckpt_dir = logdir / "ckpts"
    net_files = sorted(ckpt_dir.glob("network.iter-*.net"))
    if not net_files:
        raise FileNotFoundError(f"No *.net files found in {ckpt_dir}")
    state_dict = torch.load(net_files[-1], map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model, cfg

def register_hooks(model):
    """Cache the *features* at the end of the last block."""
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
    """Thin wrapper around FACT’s own DataLoader."""
    from utils.dataset import DataLoader, create_dataset
    train_ds, test_ds = create_dataset(cfg)
    ds = {"train": train_ds, "val": test_ds, "test": test_ds}[split]
    return DataLoader(ds, batch_size=1, shuffle=False)

def collect_embeddings(model, loader, cache, device, cap=None):
    vis, act, lab_v, lab_a, per_cls = [], [], [], [], {}
    with torch.no_grad():
        for sample in loader:
            seqs = [s.to(device) for s in sample[1]]       # (1,T,C)
            labels = sample[3][0]                          # (T,)
            model(seqs)                                    # fills cache

            F = cache.pop("visual").squeeze(1)             # (T,d)
            A = cache.pop("action").squeeze(1)             # (M,d)

            for i, cls in enumerate(labels):
                if cap and per_cls.get(cls,0) >= cap: continue
                per_cls[cls] = per_cls.get(cls,0)+1
                vis.append(F[i]); lab_v.append(int(cls))

            for t in range(A.shape[0]):
                act.append(A[t]); lab_a.append(int(labels[min(t,len(labels)-1)]))
    return torch.stack(vis), torch.tensor(lab_v), \
           torch.stack(act), torch.tensor(lab_a)

def umap2d(x, seed=42, nn=50, md=0.1):
    numba.set_num_threads(1); os.environ["OMP_NUM_THREADS"]="1"
    x50 = PCA(50, random_state=seed).fit_transform(x)
    return umap.UMAP(2, nn, md, "cosine", init="spectral",
                     random_state=seed).fit_transform(x50)

def scatter(points, labels, title, out):
    plt.figure(figsize=(6,6), dpi=300)
    plt.scatter(points[:,0], points[:,1], c=labels,
                cmap="tab20", s=4, alpha=.8)
    plt.title(title); plt.axis("off"); plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight"); plt.close()

# ─────────────────────────────────────────────────────── main ──
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=Path, required=True,
                    help="FACT run dir containing args.json & ckpts/")
    ap.add_argument("--split", default="val",
                    choices=["train","val","test"])
    ap.add_argument("--outdir", type=Path, default="figures")
    ap.add_argument("--device", default="cuda" \
        if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-per-class", type=int)
    args = ap.parse_args()

    print("Loading model …")
    model, cfg          = load_model(args.logdir, args.device)
    cache               = register_hooks(model)

    print("Collecting embeddings …")
    loader              = get_loader(cfg, args.split)
    V,yV, A,yA          = collect_embeddings(model, loader, cache,
                                             args.device, args.max_per_class)
    print("Shapes – visual:", V.shape, "action:", A.shape)

    print("Running UMAP …")
    V2d, A2d            = umap2d(V.numpy()), umap2d(A.numpy())

    print("Plotting …")
    scatter(V2d, yV, "FACT – visual branch",
            args.outdir/"breakfast_visual_umap.png")
    scatter(A2d, yA, "FACT – action branch",
            args.outdir/"breakfast_action_umap.png")
    print("✓ Done – figures in", args.outdir)

