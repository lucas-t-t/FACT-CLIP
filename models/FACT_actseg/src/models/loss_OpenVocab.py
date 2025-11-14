import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from . import basic as basic
from ..utils import utils
import numpy as np
from .loss import MatchCriterion


def smooth_loss(logit, is_logit=True):
    """
    Smoothness loss (reuse from original loss.py)
    logit: B, T, C
    """
    if is_logit:
        logsoft = F.log_softmax(logit, dim=2)
    else:
        logsoft = logit
    loss = torch.clamp((logsoft[:, 1:] - logsoft[:, :-1])**2, min=0, max=16)
    loss = loss.mean()
    return loss


def contrastive_frame_loss(frame_sim, labels):
    """
    Frame-level contrastive loss
    
    Args:
        frame_sim: (T, C) - similarity scores (already scaled by temperature)
        labels: (T,) - ground truth class indices
    """
    # Treat similarity as logits for cross-entropy
    loss = F.cross_entropy(frame_sim, labels)
    return loss


def contrastive_action_token_loss(action_sim, match, transcript):
    """
    Action token contrastive loss
    
    Args:
        action_sim: (M, C) - similarity scores
        match: (action_indices, segment_indices)
        transcript: (S,) - segment labels
    """
    aind, sind = match
    M, C = action_sim.shape
    
    # Target: null class for unmatched, transcript label for matched
    target = torch.zeros(M, dtype=torch.long, device=action_sim.device) + (C - 1)
    target[aind] = transcript[sind]
    
    loss = F.cross_entropy(action_sim, target)
    return loss


class MatchCriterion_OV(MatchCriterion):
    """
    Matching criterion for open-vocabulary FACT
    Inherits matching logic from original MatchCriterion,
    replaces classification losses with contrastive versions
    """
    
    def __init__(self, cfg, nclasses, bg_ids=[], class_weight=None):
        super().__init__(cfg, nclasses, bg_ids, class_weight)
    
    def frame_loss(self, frame_sim):
        """Contrastive frame loss"""
        return contrastive_frame_loss(frame_sim, self.class_label)
    
    def action_token_loss(self, match, action_sim):
        """Contrastive action token loss"""
        return contrastive_action_token_loss(action_sim, match, self.transcript)
    
    def frame_loss_tdu(self, seg_sim, tdu):
        """
        Frame loss for temporal downsampling (TDU)
        
        Args:
            seg_sim: (S, C) - segment similarity scores
            tdu: TemporalDownsampleUpsample object
        """
        # Create zoomed labels for segments
        ohl = self.onehot_class_label
        zoomed_label = torch.zeros([tdu.num_seg, ohl.shape[1]], dtype=ohl.dtype).to(ohl.device) 
        zoomed_label.index_add_(0, tdu.seg_label, ohl)
        zoomed_label = zoomed_label / tdu.seg_lens[:, None]
        
        # Convert similarity to log probabilities
        logp = F.log_softmax(seg_sim.squeeze(1), dim=-1)
        
        seg_loss = (- logp * zoomed_label)
        _cweight = self.cweight[:logp.shape[-1]]  # remove the weight for null class
        seg_loss = (seg_loss * _cweight)
        
        seg_loss = seg_loss.sum(-1).sum() / zoomed_label.sum()
        
        return seg_loss
    
    # Keep cross-attention losses for temporal alignment
    # They don't depend on classification, so reuse from parent class
    # cross_attn_loss() and cross_attn_loss_tdu() inherited from MatchCriterion



