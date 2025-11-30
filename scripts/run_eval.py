#!/usr/bin/python3

import numpy as np
import argparse
import os
import json
import torch
from tqdm import tqdm

from fact_clip.utils.dataset import DataLoader, create_dataset
from fact_clip.utils.evaluate import Checkpoint
from fact_clip.home import get_project_base
from fact_clip.configs.utils import cfg2flatdict, setup_cfg
from fact_clip.utils.train_tools import save_results

def evaluate(net, testloader, savedir):
    print("TESTING" + "~"*10)
    
    # Get holdout information from dataset
    holdout_classes = getattr(testloader.dataset, 'holdout_classes', [])
    seen_classes = getattr(testloader.dataset, 'seen_classes', [])
    
    ckpt = Checkpoint(
        -1, 
        bg_class=([] if net.cfg.eval_bg else testloader.dataset.bg_class),
        holdout_classes=holdout_classes,
        seen_classes=seen_classes
    )
    net.eval()
    with torch.no_grad():
        for batch_idx, (vnames, seq_list, train_label_list, eval_label_list) in enumerate(tqdm(testloader)):
            seq_list = [ s.cuda() for s in seq_list ]
            train_label_list = [ s.cuda() for s in train_label_list ]
            video_saves = net(seq_list, train_label_list)
            save_results(ckpt, vnames, eval_label_list, video_saves)

    ckpt.compute_metrics()

    string = ""
    for k, v in ckpt.metrics.items():
        string += "%s:%.1f, " % (k, v)
    
    print(string + '\n')
    
    if len(holdout_classes) > 0:
        print("="*80)
        print("HOLDOUT EVALUATION SUMMARY")
        print("="*80)
        print(f"Seen classes: {len(seen_classes)}")
        print(f"Unseen (holdout) classes: {len(holdout_classes)}")
        if 'Acc-seen' in ckpt.metrics:
            print(f"Accuracy (seen): {ckpt.metrics['Acc-seen']:.1f}%")
        if 'Acc-unseen' in ckpt.metrics:
            print(f"Accuracy (unseen): {ckpt.metrics['Acc-unseen']:.1f}%")
        if 'F1@0.10-seen' in ckpt.metrics:
            print(f"F1@0.10 (seen): {ckpt.metrics['F1@0.10-seen']:.1f}%")
        if 'F1@0.10-unseen' in ckpt.metrics:
            print(f"F1@0.10 (unseen): {ckpt.metrics['F1@0.10-unseen']:.1f}%")
        print("="*80 + "\n")
    
    os.makedirs(savedir, exist_ok=True)
    fname = "eval_result.gz"
    ckpt.save(os.path.join(savedir, fname))
    
    if len(holdout_classes) > 0:
        detail_fname = os.path.join(savedir, "eval_detailed.json")
        ckpt.save_detailed_results(detail_fname)
        print(f"Detailed results saved to: {detail_fname}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_file", nargs="*", help="optional config file", default=[])
    parser.add_argument("--set", dest="set_cfgs", help="set config keys", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--ckpt", dest="ckpt_file", help="checkpoint file to evaluate", required=True)

    args = parser.parse_args()
    
    # Setup config
    cfg = setup_cfg(args.cfg_file, args.set_cfgs)
    
    # GPU setup
    try:
        torch.cuda.set_device('cuda:%d'%cfg.aux.gpu)
    except Exception as e:
        print(e)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.aux.gpu)

    print('============')
    print(cfg)
    print('============')

    # Load Dataset
    dataset, test_dataset = create_dataset(cfg)
    testloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    print('Test dataset ', test_dataset)

    # Create Network
    use_clip = getattr(cfg, 'use_clip', False) or getattr(cfg.CLIP, 'enabled', False)
    
    if use_clip:
        print("Creating FACT_CLIP Model...")
        from fact_clip.utils.dataset import load_action_mapping
        BASE = get_project_base()
        
        if cfg.map_fname:
            map_fname = cfg.map_fname
        elif cfg.dataset.startswith('havid'):
            variant = cfg.dataset.replace("havid_", "")
            map_fname = os.path.join(BASE, 'data', 'HAViD', 'ActionSegmentation', 'data', variant, 'mapping.txt')
        else:
            map_fname = None
        
        text_embeddings = None
        if map_fname and os.path.exists(map_fname):
            label2index, index2label = load_action_mapping(map_fname)
            from fact_clip.utils.text_embeddings import get_or_compute_text_embeddings
            try:
                text_embeddings = get_or_compute_text_embeddings(
                    cfg, label2index, index2label, device=f'cuda:{cfg.aux.gpu}'
                )
            except Exception as e:
                print(f"Warning: Failed to load/compute text embeddings: {e}")
        
        from fact_clip.models.blocks import FACT_CLIP
        net = FACT_CLIP(cfg, dataset.input_dimension, dataset.nclasses, text_embeddings=text_embeddings)
    else:
        from fact_clip.models.blocks import FACT
        net = FACT(cfg, dataset.input_dimension, dataset.nclasses)

    net.cuda()
    
    # Load Checkpoint
    print(f"Loading checkpoint: {args.ckpt_file}")
    ckpt = torch.load(args.ckpt_file, map_location="cpu")
    if 'frame_pe.pe' in ckpt: del ckpt['frame_pe.pe']
    if 'action_pe.pe' in ckpt: del ckpt['action_pe.pe']
    net.load_state_dict(ckpt, strict=False)
    print("Checkpoint loaded.")

    # Evaluate
    savedir = os.path.join(os.path.dirname(args.ckpt_file), '../eval_results')
    evaluate(net, testloader, savedir)

