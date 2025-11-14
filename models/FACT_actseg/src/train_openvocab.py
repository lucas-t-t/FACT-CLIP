#!/usr/bin/python3

import numpy as np
import argparse
import os
import json
from torch import optim
import torch
import wandb

from .utils.dataset import DataLoader
from .utils.dataset_OpenVocab import create_dataset_openvocab
from .utils.evaluate import Checkpoint
from .home import get_project_base
from .configs.utils import cfg2flatdict, setup_cfg
from .utils.train_tools import resume_ckpt, compute_null_weight, save_results
from .models.loss_OpenVocab import MatchCriterion_OV
from .models.blocks_OpenVocab import FACT_OpenVocab


def evaluate(global_step, net, testloader, run, savedir):
    print("TESTING" + "~"*10)

    # Get holdout information from dataset
    holdout_classes = getattr(testloader.dataset, 'holdout_classes', [])
    seen_classes = getattr(testloader.dataset, 'seen_classes', [])
    
    ckpt = Checkpoint(
        global_step+1, 
        bg_class=([] if net.cfg.eval_bg else testloader.dataset.bg_class),
        holdout_classes=holdout_classes,
        seen_classes=seen_classes
    )
    net.eval()
    with torch.no_grad():
        for batch_idx, (vnames, seq_list, train_label_list, eval_label_list) in enumerate(testloader):

            seq_list = [ s.cuda() for s in seq_list ]
            train_label_list = [ s.cuda() for s in train_label_list ]
            video_saves = net(seq_list, train_label_list)
            save_results(ckpt, vnames, eval_label_list, video_saves)

    net.train()
    ckpt.compute_metrics()

    log_dict = {}
    string = ""
    
    # Log standard metrics
    for k, v in ckpt.metrics.items():
        string += "%s:%.1f, " % (k, v)
        # Organize metrics by type for wandb
        if '-seen' in k:
            log_dict[f'test-metric-seen/{k.replace("-seen", "")}'] = v
        elif '-unseen' in k:
            log_dict[f'test-metric-unseen/{k.replace("-unseen", "")}'] = v
        else:
            log_dict[f'test-metric-all/{k}'] = v
    
    print(string + '\n')
    
    # Print holdout-specific summary if applicable
    if len(holdout_classes) > 0:
        print("="*80)
        print("OPEN-VOCABULARY EVALUATION SUMMARY")
        print("="*80)
        print(f"Seen classes: {len(seen_classes)}")
        print(f"Unseen (holdout) classes: {len(holdout_classes)}")
        if 'Acc-seen' in ckpt.metrics:
            print(f"Accuracy (seen): {ckpt.metrics['Acc-seen']:.1f}%")
        if 'Acc-unseen' in ckpt.metrics:
            print(f"Accuracy (unseen): {ckpt.metrics['Acc-unseen']:.1f}%")
        if 'F1@0.50-seen' in ckpt.metrics:
            print(f"F1@0.50 (seen): {ckpt.metrics['F1@0.50-seen']:.1f}%")
        if 'F1@0.50-unseen' in ckpt.metrics:
            print(f"F1@0.50 (unseen): {ckpt.metrics['F1@0.50-unseen']:.1f}%")
        print("="*80 + "\n")
    
    run.log(log_dict, step=global_step+1)

    fname = "%d.gz" % (global_step+1) 
    ckpt.save(os.path.join(savedir, fname))
    
    # Save detailed results for holdout experiments
    if len(holdout_classes) > 0:
        detail_fname = os.path.join(savedir, f"{global_step+1}_detailed.json")
        ckpt.save_detailed_results(detail_fname)

    return ckpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_file", nargs="*",
                            help="optional config file", default=[])
    parser.add_argument("--set", dest="set_cfgs",
            help="set config keys", default=None, nargs=argparse.REMAINDER,)

    args = parser.parse_args()
    BASE = get_project_base()

    ### initialize experiment #########################################################
    cfg = setup_cfg(args.cfg_file, args.set_cfgs)
    try:
        torch.cuda.set_device('cuda:%d'%cfg.aux.gpu)
    except Exception as e:
        print(e)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.aux.gpu)

    print('============')
    print(cfg)
    print('============')

    if cfg.aux.debug:
        seed = 1 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    logdir = os.path.join(BASE, cfg.aux.logdir)
    ckptdir = os.path.join(logdir, 'ckpts')
    savedir = os.path.join(logdir, 'saves')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    print('Saving log at', logdir)

    run = wandb.init(
                project=cfg.aux.wandb_project, entity=cfg.aux.wandb_user,
                dir=cfg.aux.logdir,
                group=cfg.aux.exp, resume="allow",
                config=cfg2flatdict(cfg),
                reinit=True, save_code=False,
                mode="offline" if cfg.aux.debug else "online",
                )

    argSaveFile = os.path.join(logdir, 'args.json')
    with open(argSaveFile, 'w') as f:
        json.dump(cfg, f, indent=True)

    ### load dataset #########################################################
    print("\n" + "="*80)
    print("LOADING OPEN-VOCABULARY DATASET")
    print("="*80)
    
    dataset, test_dataset, text_embeddings, action_descriptions = create_dataset_openvocab(cfg)
    
    if not cfg.aux.debug:
        trainloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    else:
        trainloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    print(f'\nVisual feature dim: {dataset.input_dimension}')
    print(f'Number of action classes: {len(action_descriptions)}')
    print('Train dataset', dataset)
    print('Test dataset ', test_dataset)
    print("="*80 + "\n")

    ### create network #########################################################
    print("="*80)
    print("CREATING OPEN-VOCABULARY FACT MODEL")
    print("="*80)
    
    net = FACT_OpenVocab(
        cfg, 
        visual_input_dim=dataset.input_dimension,
        action_descriptions=action_descriptions
    )

    if cfg.Loss.nullw == -1:
        compute_null_weight(cfg, dataset)
    net.mcriterion = MatchCriterion_OV(cfg, dataset.nclasses, dataset.bg_class)

    global_step, ckpt_file = resume_ckpt(cfg, logdir)
    if ckpt_file is not None:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        if 'frame_pe.pe' in ckpt: del ckpt['frame_pe.pe']
        if 'action_pe.pe' in ckpt: del ckpt['action_pe.pe']
        net.load_state_dict(ckpt, strict=False)
        print(f"Loaded checkpoint from {ckpt_file}")
    
    net.cuda()
    
    # Register pre-computed text embeddings (after moving model to CUDA)
    if text_embeddings is not None:
        print(f"Registering pre-computed text embeddings: {text_embeddings.shape}")
        net.register_text_embeddings(text_embeddings.cuda())
    else:
        print("Text embeddings will be computed during training")
    
    print(net)
    print("="*80 + "\n")

    ### optimizer with different learning rates #########################################################
    print("="*80)
    print("SETTING UP OPTIMIZER")
    print("="*80)
    
    # Different learning rates for different components
    clip_text_params = list(net.clip_text.parameters())
    visual_projection_params = list(net.visual_projection.parameters())
    fact_params = list(net.block_list.parameters()) + \
                  list(net.visual_to_fact.parameters()) + \
                  list(net.text_to_fact.parameters())
    
    # Add temperature parameter
    other_params = [net.temperature]
    if hasattr(net, 'action_query'):
        other_params.append(net.action_query)
    
    param_groups = [
        {'params': clip_text_params, 'lr': cfg.lr * 0.1, 'name': 'clip_text'},  # Lower LR for CLIP
        {'params': visual_projection_params, 'lr': cfg.lr, 'name': 'visual_projection'},  # Full LR
        {'params': fact_params, 'lr': cfg.lr, 'name': 'fact_blocks'},  # Full LR
        {'params': other_params, 'lr': cfg.lr, 'name': 'other'},  # Full LR
    ]
    
    print(f"CLIP text encoder LR: {cfg.lr * 0.1}")
    print(f"Visual projection LR: {cfg.lr}")
    print(f"FACT blocks LR: {cfg.lr}")
    print(f"Other params LR: {cfg.lr}")
    
    if cfg.optimizer == 'SGD':
        optimizer = optim.SGD(param_groups, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        optimizer = optim.Adam(param_groups, weight_decay=cfg.weight_decay)
    
    print("="*80 + "\n")

    ### start training #########################################################
    start_epoch = global_step // len(trainloader)
    
    # Get holdout information from dataset
    holdout_classes = getattr(testloader.dataset, 'holdout_classes', [])
    seen_classes = getattr(testloader.dataset, 'seen_classes', [])
    
    # Log holdout information
    if len(holdout_classes) > 0:
        print("="*80)
        print("ZERO-SHOT / HOLDOUT TRAINING MODE")
        print("="*80)
        print(f"Training with {len(seen_classes)} seen classes")
        print(f"Will evaluate on {len(holdout_classes)} unseen classes")
        print(f"Holdout classes: {holdout_classes}")
        print("="*80 + "\n")
    
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    print(f"Starting from epoch {start_epoch}, step {global_step}")
    print(f"Total epochs: {cfg.epoch}")
    print(f"Steps per epoch: {len(trainloader)}")
    print("="*80 + "\n")
    
    for epoch_idx in range(start_epoch, cfg.epoch):
        for batch_idx, (vnames, seq_list, label_list, _) in enumerate(trainloader):

            seq_list = [ s.cuda() for s in seq_list ]
            label_list = [ s.cuda() for s in label_list ]

            loss, video_saves = net(seq_list, label_list, compute_loss=True)
            
            optimizer.zero_grad()
            loss.backward()
            
            if cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_grad_norm)
            
            optimizer.step()

            if (global_step+1) % cfg.aux.print_every == 0:
                loss_list = [l.item() for l in net.loss_list]
                string = "Ep:%d Stp:%d  Loss:%.3f (%.3f)" % (epoch_idx, global_step+1, loss.item(), np.mean(loss_list))
                for i, l in enumerate(loss_list):
                    string += " %.3f" % l
                print(string)
                
                run.log({'train/loss': loss.item(), 'train/loss_mean': np.mean(loss_list)}, step=global_step+1)

            if (global_step+1) % cfg.aux.eval_every == 0:
                ckpt_fname = os.path.join(ckptdir, 'network.iter-%d.net'%(global_step+1))
                net.save_model(ckpt_fname)
                evaluate(global_step, net, testloader, run, savedir)

            global_step += 1
        
        # Learning rate decay
        if cfg.lr_decay > 0 and (epoch_idx + 1) % cfg.lr_decay == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"Learning rate decayed at epoch {epoch_idx+1}")
    
    # Save final model
    ckpt_fname = os.path.join(ckptdir, 'network.iter-%d.net'%(global_step))
    net.save_model(ckpt_fname)
    
    # Final evaluation
    evaluate(global_step-1, net, testloader, run, savedir)
    
    # Mark completion
    with open(os.path.join(logdir, 'FINISH_PROOF'), 'w') as f:
        f.write('done')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)



