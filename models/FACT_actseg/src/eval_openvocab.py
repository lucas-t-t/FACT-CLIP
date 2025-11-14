#!/usr/bin/python3

import numpy as np
import argparse
import os
import json
import torch
from tqdm import tqdm

from .utils.dataset import DataLoader
from .utils.dataset_OpenVocab import create_dataset_openvocab, format_action_description
from .utils.evaluate import Checkpoint
from .home import get_project_base
from .configs.utils import cfg2flatdict, setup_cfg
from .utils.train_tools import save_results
from .models.loss_OpenVocab import MatchCriterion_OV
from .models.blocks_OpenVocab import FACT_OpenVocab


def evaluate_standard(cfg, net, testloader):
    """
    Standard evaluation on seen classes
    """
    print("\n" + "="*80)
    print("STANDARD EVALUATION (Seen Classes)")
    print("="*80)
    
    ckpt = Checkpoint(
        -1, 
        bg_class=([] if cfg.eval_bg else testloader.dataset.bg_class)
    )
    
    net.eval()
    with torch.no_grad():
        for vnames, seq_list, train_label_list, eval_label_list in tqdm(testloader):
            seq_list = [s.cuda() for s in seq_list]
            train_label_list = [s.cuda() for s in train_label_list]
            video_saves = net(seq_list, train_label_list)
            save_results(ckpt, vnames, eval_label_list, video_saves)
    
    ckpt.compute_metrics()
    
    print("\nResults:")
    for k, v in ckpt.metrics.items():
        print(f"  {k:20s}: {v:.1f}")
    print("="*80 + "\n")
    
    return ckpt


def evaluate_zero_shot(cfg, net, testloader, unseen_action_descriptions):
    """
    Zero-shot evaluation on unseen action classes
    
    Args:
        cfg: config
        net: FACT_OpenVocab model
        testloader: dataloader
        unseen_action_descriptions: List[str] - text descriptions for unseen classes
    """
    print("\n" + "="*80)
    print("ZERO-SHOT EVALUATION (Unseen Classes)")
    print("="*80)
    print(f"Number of unseen action descriptions: {len(unseen_action_descriptions)}")
    print("\nUnseen action descriptions:")
    for i, desc in enumerate(unseen_action_descriptions[:10]):
        print(f"  {i}: {desc}")
    if len(unseen_action_descriptions) > 10:
        print(f"  ... and {len(unseen_action_descriptions) - 10} more")
    print("="*80)
    
    # Disable cached text embeddings for dynamic encoding
    net.text_embeddings_clip = None
    net.text_embeddings_fact = None
    
    # Update action descriptions in the model
    net.action_descriptions = unseen_action_descriptions
    
    ckpt = Checkpoint(
        -1, 
        bg_class=([] if cfg.eval_bg else testloader.dataset.bg_class)
    )
    
    net.eval()
    with torch.no_grad():
        for vnames, seq_list, train_label_list, eval_label_list in tqdm(testloader):
            seq_list = [s.cuda() for s in seq_list]
            train_label_list = [s.cuda() for s in train_label_list]
            
            # Forward with dynamic text encoding
            video_saves = net(seq_list, train_label_list)
            save_results(ckpt, vnames, eval_label_list, video_saves)
    
    ckpt.compute_metrics()
    
    print("\nZero-shot Results:")
    for k, v in ckpt.metrics.items():
        print(f"  {k:20s}: {v:.1f}")
    print("="*80 + "\n")
    
    return ckpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_file", nargs="*",
                        help="optional config file", default=[])
    parser.add_argument("--ckpt", dest="ckpt_path", type=str, required=True,
                        help="path to model checkpoint")
    parser.add_argument("--set", dest="set_cfgs",
                        help="set config keys", default=None, nargs=argparse.REMAINDER,)
    parser.add_argument("--zero_shot", action="store_true",
                        help="perform zero-shot evaluation with custom action descriptions")
    parser.add_argument("--unseen_actions", type=str, default=None,
                        help="comma-separated list of unseen action names for zero-shot eval")
    parser.add_argument("--unseen_actions_file", type=str, default=None,
                        help="path to file with unseen action names (one per line)")

    args = parser.parse_args()
    BASE = get_project_base()

    ### Initialize experiment
    cfg = setup_cfg(args.cfg_file, args.set_cfgs)
    
    try:
        torch.cuda.set_device('cuda:%d'%cfg.aux.gpu)
    except Exception as e:
        print(e)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.aux.gpu)

    print('='*80)
    print('Configuration:')
    print('='*80)
    print(cfg)
    print('='*80 + '\n')

    ### Load dataset
    print("="*80)
    print("LOADING DATASET")
    print("="*80)
    
    dataset, test_dataset, text_embeddings, action_descriptions = create_dataset_openvocab(cfg)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f'\nVisual feature dim: {dataset.input_dimension}')
    print(f'Number of action classes: {len(action_descriptions)}')
    print('Test dataset:', test_dataset)
    print("="*80 + "\n")

    ### Load model
    print("="*80)
    print("LOADING MODEL")
    print("="*80)
    print(f"Checkpoint: {args.ckpt_path}")
    
    net = FACT_OpenVocab(
        cfg,
        visual_input_dim=dataset.input_dimension,
        action_descriptions=action_descriptions
    )
    
    # Load checkpoint
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    if 'frame_pe.pe' in ckpt:
        del ckpt['frame_pe.pe']
    if 'action_pe.pe' in ckpt:
        del ckpt['action_pe.pe']
    
    net.load_state_dict(ckpt, strict=False)
    net.cuda()
    net.eval()
    
    print(f"Model loaded successfully")
    print("="*80 + "\n")

    ### Evaluation
    if not args.zero_shot:
        # Standard evaluation on seen classes
        # Register pre-computed text embeddings
        if text_embeddings is not None:
            net.register_text_embeddings(text_embeddings.cuda())
        
        evaluate_standard(cfg, net, testloader)
    
    else:
        # Zero-shot evaluation on unseen classes
        # Parse unseen action descriptions
        if args.unseen_actions_file:
            with open(args.unseen_actions_file, 'r') as f:
                unseen_action_names = [line.strip() for line in f if line.strip()]
        elif args.unseen_actions:
            unseen_action_names = [a.strip() for a in args.unseen_actions.split(',')]
        else:
            raise ValueError("Must provide --unseen_actions or --unseen_actions_file for zero-shot evaluation")
        
        # Format as natural language descriptions
        unseen_action_descriptions = [
            format_action_description(action_name, cfg.CLIP.use_prompt)
            for action_name in unseen_action_names
        ]
        
        evaluate_zero_shot(cfg, net, testloader, unseen_action_descriptions)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)



