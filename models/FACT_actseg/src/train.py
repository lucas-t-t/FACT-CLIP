#!/usr/bin/python3

import numpy as np
import argparse
import os
import json
from torch import optim
import torch
import wandb

from .utils.dataset import DataLoader, create_dataset
from .utils.evaluate import Checkpoint
from .home import get_project_base
from .configs.utils import cfg2flatdict, setup_cfg
from .utils.train_tools import resume_ckpt, compute_null_weight, save_results
from .models.loss import MatchCriterion

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
        print("HOLDOUT EVALUATION SUMMARY")
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
                mode="offline" if (cfg.aux.debug or cfg.aux.wandb_offline) else "online",
                )

    argSaveFile = os.path.join(logdir, 'args.json')
    with open(argSaveFile, 'w') as f:
        json.dump(cfg, f, indent=True)

    ### load dataset #########################################################
    dataset, test_dataset = create_dataset(cfg)
    if not cfg.aux.debug:
        trainloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    else:
        trainloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    print('Train dataset', dataset)
    print('Test dataset ', test_dataset)

    ### create network #########################################################
    # Check if using CLIP version (default to False for backward compatibility)
    use_clip = getattr(cfg, 'use_clip', False) or getattr(cfg.CLIP, 'enabled', False)
    
    if use_clip:
        print("="*80)
        print("CREATING FACT_CLIP MODEL (Open-Vocabulary)")
        print("="*80)
        
        # Load action mapping for text description generation
        from .utils.dataset import load_action_mapping
        BASE = get_project_base()
        
        # Determine mapping file path
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
            
            # Get or compute text embeddings
            from .utils.text_embeddings import get_or_compute_text_embeddings
            try:
                text_embeddings = get_or_compute_text_embeddings(
                    cfg, label2index, index2label, device=f'cuda:{cfg.aux.gpu}'
                )
            except Exception as e:
                print(f"Warning: Failed to load/compute text embeddings: {e}")
                print("Continuing without text embeddings (contrastive loss will be disabled)")
        else:
            print(f"Warning: Mapping file not found at {map_fname if map_fname else 'default path'}")
            print("Continuing without text embeddings (contrastive loss will be disabled)")
        
        # Import and create FACT_CLIP model
        if cfg.dataset == 'epic':
            raise ValueError("FACT_CLIP not yet supported for epic dataset")
        else:
            from .models.blocks import FACT_CLIP
            net = FACT_CLIP(cfg, dataset.input_dimension, dataset.nclasses, text_embeddings=text_embeddings)
        
        print("="*80 + "\n")
    else:
        # Use vanilla FACT model
        if cfg.dataset == 'epic':
            from .models.blocks_SepVerbNoun import FACT
            net = FACT(cfg, dataset.input_dimension, 98, 301)
        else:
            from .models.blocks import FACT
            net = FACT(cfg, dataset.input_dimension, dataset.nclasses)

    if cfg.Loss.nullw == -1:
        compute_null_weight(cfg, dataset)
    net.mcriterion = MatchCriterion(cfg, dataset.nclasses, dataset.bg_class)

    global_step, ckpt_file = resume_ckpt(cfg, logdir)
    if ckpt_file is not None:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        if 'frame_pe.pe' in ckpt: del ckpt['frame_pe.pe']
        if 'action_pe.pe' in ckpt: del ckpt['action_pe.pe']
        net.load_state_dict(ckpt, strict=False)
    net.cuda()

    print(net)

    if cfg.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(),
                            lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)

    ### start training #########################################################
    start_epoch = global_step // len(trainloader)
    
    # Get holdout information from dataset
    holdout_classes = getattr(testloader.dataset, 'holdout_classes', [])
    seen_classes = getattr(testloader.dataset, 'seen_classes', [])
    
    # Log holdout information
    if cfg.holdout_mode and len(holdout_classes) > 0:
        print("\n" + "="*80)
        print("HOLDOUT TRAINING MODE")
        print("="*80)
        print(f"Holdout (unseen) classes: {len(holdout_classes)}")
        print(f"Seen classes: {len(seen_classes)}")
        print(f"Holdout class IDs: {holdout_classes}")
        if hasattr(dataset, 'index2label'):
            print(f"Holdout class names: {[dataset.index2label[c] for c in holdout_classes if c in dataset.index2label]}")
        print("="*80 + "\n")
    
    ckpt = Checkpoint(
        -1, 
        bg_class=([] if net.cfg.eval_bg else testloader.dataset.bg_class), 
        eval_edit=False,
        holdout_classes=holdout_classes,
        seen_classes=seen_classes
    )
    best_ckpt, best_metric = None, 0

    print(f'Start Training from Epoch {start_epoch}...')
    for eidx in range(start_epoch, cfg.epoch):

        for batch_idx, (vnames, seq_list, train_label_list, eval_label_list) in enumerate(trainloader):

            seq_list = [ s.cuda() for s in seq_list ]
            train_label_list = [ s.cuda() for s in train_label_list ]

            optimizer.zero_grad()
            loss, video_saves = net(seq_list, train_label_list, compute_loss=True)
            loss.backward()

            if cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_grad_norm)
            optimizer.step()

            save_results(ckpt, vnames, eval_label_list, video_saves)

            # print some progress information
            if (global_step+1) % cfg.aux.print_every == 0:

                ckpt.compute_metrics()
                ckpt.average_losses()

                log_dict = {}
                string = "Iter%d, " % (global_step+1)
                _L = len(string)
                for k, v in ckpt.loss.items():
                    log_dict[f"train-loss/{k}"] = v
                    string += f"{k}:{v:.1f}, "
                
                # Log individual FACT and contrastive losses if available (for FACT_CLIP)
                if use_clip and hasattr(net, 'fact_loss') and hasattr(net, 'contrastive_loss'):
                    # These are from the last forward pass
                    log_dict["train-loss/fact_loss"] = net.fact_loss.item() if hasattr(net, 'fact_loss') else 0
                    log_dict["train-loss/contrastive_loss"] = net.contrastive_loss.item() if hasattr(net, 'contrastive_loss') else 0
                
                print(string)

                string = " " * _L 
                for k, v in ckpt.metrics.items():
                    string += "%s:%.3f, " % (k, v)
                    log_dict['train-metric/'+k] = v
                print(string)

                run.log(log_dict, step=global_step+1)

                ckpt = Checkpoint(
                    -1, 
                    bg_class=(dataset.bg_class if cfg.eval_bg else []), 
                    eval_edit=False,
                    holdout_classes=holdout_classes,
                    seen_classes=seen_classes
                )

            # test and save model every x iterations
            if global_step != 0 and (global_step+1) % cfg.aux.eval_every == 0:
                test_ckpt = evaluate(global_step, net, testloader, run, savedir)
                if test_ckpt.metrics['F1@0.50'] >= best_metric:
                    best_ckpt = test_ckpt
                    best_metric = test_ckpt.metrics['F1@0.50']

                network_file = ckptdir + '/network.iter-' + str(global_step+1) + '.net'
                net.save_model(network_file)

            global_step += 1

        if cfg.lr_decay > 0 and ( eidx + 1 ) % cfg.lr_decay == 0:
            for g in optimizer.param_groups:
                g['lr'] = cfg.lr * 0.1
            print('------------------------------------Update Learning rate--------------------------------')

    # Save best checkpoint if evaluation occurred
    if best_ckpt is not None:
        print(f'Best Checkpoint: {best_ckpt.iteration}')
        best_ckpt.eval_edit = True
        best_ckpt.compute_metrics()
        best_ckpt.save(os.path.join(logdir, 'best_ckpt.gz'))
    else:
        print('No evaluation performed during training (best checkpoint not available)')
    
    run.finish()

    # create a file to mark this experiment has completed
    finish_proof_fname = os.path.join(logdir, "FINISH_PROOF")
    open(finish_proof_fname, "w").close()
