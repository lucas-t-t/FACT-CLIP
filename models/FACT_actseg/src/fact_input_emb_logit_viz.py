#!/usr/bin/env python3
# fact_emb_and_logit_viz.py

import argparse, sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch, umap, numpy as np
from sklearn.decomposition import PCA

repo_root    = Path(__file__).resolve().parent
project_root = repo_root.parent
sys.path.insert(0, str(project_root))

def load_cfg(logdir: Path):
    from yacs.config import CfgNode as CN; import json
    cfg = CN(json.load(open(logdir / "args.json")));  cfg.freeze();  return cfg

def load_model(logdir: Path, device="cuda"):
    from src.utils.dataset import create_dataset
    from src.models.blocks import FACT
    cfg             = load_cfg(logdir)
    _, test_ds      = create_dataset(cfg)
    in_dim, n_cls   = test_ds.input_dimension, test_ds.nclasses
    model           = FACT(cfg, in_dim, n_cls).to(device).eval()
    ckpt            = sorted((logdir/"ckpts").glob("network.iter-*.net"))[-1]
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model, cfg, n_cls

def register_hooks(model):
    cache, last_blk = {}, model.block_list[-1]
    last_blk.frame_branch .register_forward_hook(
        lambda _m,_i,o: cache.__setitem__("visual",  o.detach()))
    last_blk.action_branch.register_forward_hook(
        lambda _m,_i,o: cache.__setitem__("action",  o.detach()))
    cache["last_blk"] = last_blk
    return cache

def get_loader(cfg, split):
    from src.utils.dataset import DataLoader, create_dataset
    tr, te = create_dataset(cfg);  ds = {"train":tr,"val":te,"test":te}[split]
    return DataLoader(ds, batch_size=1, shuffle=False)

def collect_embeddings(model, loader, cache, device, cap):
    vis, lab_v, act, lab_a, per_cls = [], [], [], [], {}
    with torch.no_grad():
        for sample in loader:
            seqs        = [s.to(device) for s in sample[1]]
            train_lbls  = sample[2]
            labels      = sample[3][0]
            model(seqs, train_lbls)

            F = cache.pop("visual").squeeze(1)
            A = cache.pop("action").squeeze(1)

            for i, cls in enumerate(labels):
                if per_cls.get(cls,0) >= cap: continue
                per_cls[cls] = per_cls.get(cls,0)+1
                vis.append(F[i]);  lab_v.append(int(cls))
            for t in range(A.size(0)):
                act.append(A[t])
                lab_a.append(int(labels[min(t, len(labels)-1)]))

    return (torch.stack(vis),  torch.tensor(lab_v),
            torch.stack(act),  torch.tensor(lab_a))

def collect_input_embeddings(model, loader, device, cap):
    vis, lab_v, act, lab_a, per_cls = [], [], [], [], {}
    blk = model.block_list[0]

    cache = {}
    blk.frame_branch .register_forward_hook(lambda _m,_i,o: cache.__setitem__('vis_in', o.detach()))
    blk.action_branch.register_forward_hook(lambda _m,_i,o: cache.__setitem__('act_in', o.detach()))

    with torch.no_grad():
        for sample in loader:
            seqs       = [s.to(device) for s in sample[1]]
            train_lbls = sample[2]
            labels     = sample[3][0]
            model(seqs, train_lbls)

            F = cache.pop("vis_in").squeeze(1)
            A = cache.pop("act_in").squeeze(1)

            for i, cls in enumerate(labels):
                if per_cls.get(cls, 0) >= cap: continue
                per_cls[cls] = per_cls.get(cls, 0) + 1
                vis.append(F[i]); lab_v.append(int(cls))
            for t in range(A.size(0)):
                act.append(A[t])
                lab_a.append(int(labels[min(t, len(labels)-1)]))

    return (torch.stack(vis), torch.tensor(lab_v),
            torch.stack(act), torch.tensor(lab_a))

def collect_logits(model, loader, cache, n_cls, device, cap):
    rows_f, rows_a, labs, per_cls = [], [], [], {}
    last_blk = cache["last_blk"]

    with torch.no_grad():
        for sample in loader:
            seqs        = [s.to(device) for s in sample[1]]
            train_lbls  = sample[2]
            labels      = sample[3][0]
            model(seqs, train_lbls)

            frame_logit  = last_blk.frame_clogit.squeeze(1)[:, :n_cls]
            action_logit = last_blk.action_clogit.squeeze(1)[:, :n_cls]
            Lambda_f     = last_blk.a2f_attn.squeeze(0)

            T = min(frame_logit.size(0), Lambda_f.size(0), len(labels))
            frame_logit = frame_logit[:T]
            Lambda_f    = Lambda_f[:T]
            labels      = labels[:T]

            token_idx   = Lambda_f.argmax(dim=-1)
            token2frame = action_logit[token_idx]

            for i, cls in enumerate(labels):
                if per_cls.get(cls,0) >= cap: continue
                per_cls[cls] = per_cls.get(cls,0)+1
                rows_f.append(frame_logit[i].unsqueeze(0))
                rows_a.append(token2frame[i].unsqueeze(0))
                labs.append(int(cls))

    Pf = torch.cat(rows_f, dim=0)
    Pa = torch.cat(rows_a, dim=0)
    return Pf, Pa, torch.tensor(labs)

def umap2d(t: torch.Tensor, n_neighbors: int = 50, min_dist: float = 0.1):
    max_comp = min(50, t.shape[1] - 1, t.shape[0] - 1)

    if t.is_cuda:
        try:
            import cupy as cp
            from cuml.decomposition import PCA as gpuPCA
            from cuml.manifold      import UMAP as gpuUMAP

            X = cp.asarray(t.detach()).astype(cp.float32, copy=False)
            X = gpuPCA(n_components=max_comp).fit_transform(X)
            X = gpuUMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric="cosine",
                ).fit_transform(X)
            return cp.asnumpy(X)
        except ModuleNotFoundError:
            pass

    X = t.detach().cpu().numpy().astype(np.float32, copy=False)
    X = PCA(n_components=max_comp).fit_transform(X)
    return umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric="cosine",
                init="spectral",
            ).fit_transform(X)

def scatter(pts, lbls, title, out):
    plt.figure(figsize=(6,6), dpi=300)
    plt.scatter(pts[:,0], pts[:,1], c=lbls, cmap="tab20", s=4, alpha=.8)
    plt.title(title); plt.axis("off"); plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=Path, required=True)
    ap.add_argument("--split", choices=["train","val","test"], default="val")
    ap.add_argument("--outdir", type=Path, default="figures")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-per-class", type=int, default=200)
    args = ap.parse_args()

    print("Loading model …")
    model, cfg, n_cls = load_model(args.logdir, args.device)
    cache             = register_hooks(model)
    loader            = get_loader(cfg, args.split)

    print("Collecting raw embeddings …")
    V, yV, A, yA = collect_embeddings(model, loader, cache, args.device, args.max_per_class)

    print("Collecting last-block logits …")
    Pf, Pa, yL = collect_logits(model, loader, cache, n_cls, args.device, args.max_per_class)

    print("Collecting input-block embeddings …")
    Vi, yVi, Ai, yAi = collect_input_embeddings(model, loader, args.device, args.max_per_class)

    print("Running UMAP …")
    V2d  = umap2d(V)
    A2d  = umap2d(A)
    Pf2d = umap2d(Pf)
    Pa2d = umap2d(Pa)
    Vi2d = umap2d(Vi)
    Ai2d = umap2d(Ai)

    print("Plotting …")
    od = args.outdir
    scatter(V2d,  yV,  "FACT – visual branch",     od/"breakfast_visual_umap.png")
    scatter(A2d,  yA,  "FACT – action branch",     od/"breakfast_action_umap.png")
    scatter(Pf2d, yL,  "FACT – frame-logit rail",  od/"breakfast_frame_logits_umap.png")
    scatter(Pa2d, yL,  "FACT – token-logit rail",  od/"breakfast_token_logits_umap.png")
    scatter(Vi2d, yVi, "FACT – input visual",       od/"breakfast_input_visual_umap.png")
    scatter(Ai2d, yAi, "FACT – input action",       od/"breakfast_input_action_umap.png")

    print("\u2713 Done – figures in", od)

