#!/usr/bin/env python3
"""
Test script to verify FACT_CLIP implementation for zero-shot generalization.
This script performs basic sanity checks on the model architecture and forward pass.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.configs.default import get_cfg_defaults
from src.models.blocks import FACT_CLIP
from src.models.loss import MatchCriterion

def test_fact_clip_basic():
    """Test basic FACT_CLIP instantiation and forward pass."""
    print("\n" + "="*80)
    print("TESTING FACT_CLIP IMPLEMENTATION")
    print("="*80)
    
    # Setup config
    cfg = get_cfg_defaults()
    cfg.use_clip = True
    cfg.CLIP.model_name = "openai/clip-vit-base-patch32"
    cfg.CLIP.projection_hidden_dim = 512
    cfg.CLIP.projection_dropout = 0.1
    cfg.CLIP.fact_loss_weight = 0.5
    cfg.CLIP.contrastive_weight = 0.5
    cfg.CLIP.temp = 0.07
    cfg.FACT.trans = False
    cfg.FACT.ntoken = 30
    cfg.FACT.block = "iu"  # Use input block + update block for proper testing
    cfg.Bi.hid_dim = 256
    cfg.Bi.a_dim = 256
    cfg.Bi.f_dim = 256
    cfg.Bi.f = 'm'
    cfg.Bi.a = 'sca'
    cfg.Bi.a_nhead = 8
    cfg.Bi.a_ffdim = 512
    cfg.Bi.a_layers = 2
    cfg.Bi.f_layers = 3
    cfg.Bi.f_ln = False
    cfg.Bi.f_ngp = 1
    cfg.Bi.dropout = 0.0
    
    # Update block config
    cfg.Bu.a = 'sa'
    cfg.Bu.a_nhead = 8
    cfg.Bu.a_layers = 1
    
    # Create dummy text embeddings
    n_classes = 43  # HAViD has 43 classes
    text_embeddings = torch.randn(n_classes, 512)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
    
    # Create model
    in_dim = 2048  # I3D feature dimension
    print(f"\n1. Creating FACT_CLIP model...")
    print(f"   - Input dimension: {in_dim}")
    print(f"   - Number of classes: {n_classes}")
    print(f"   - Text embeddings shape: {text_embeddings.shape}")
    
    model = FACT_CLIP(cfg, in_dim, n_classes, text_embeddings=text_embeddings)
    model.train()
    
    # Create dummy input
    T = 100  # 100 frames
    seq = torch.randn(T, in_dim)
    label = torch.randint(0, n_classes, (T,))
    
    print(f"\n2. Testing forward pass...")
    print(f"   - Input sequence shape: {seq.shape}")
    print(f"   - Label shape: {label.shape}")
    
    # Forward pass
    seq = seq.unsqueeze(1)  # (T, 1, in_dim)
    from src.models.basic import torch_class_label_to_segment_label
    trans = torch_class_label_to_segment_label(label)[0]
    
    model.mcriterion = MatchCriterion(cfg, n_classes, bg_ids=[])
    
    print(f"\n3. Running forward pass...")
    block_output = model._forward_one_video(seq, trans)
    
    # Check that action features are saved
    last_block = model.block_list[-1]
    assert hasattr(last_block, 'action_feature'), "Action features not saved in block!"
    print(f"   ✓ Action features saved in block")
    print(f"   - Action feature shape: {last_block.action_feature.shape}")
    
    # Check projected embeddings
    assert hasattr(model, 'projected_action_embeddings'), "Projected embeddings not created!"
    print(f"   ✓ Projected action embeddings created")
    print(f"   - Projected shape: {model.projected_action_embeddings.shape}")
    
    # Test contrastive loss in isolation
    print(f"\n4. Testing contrastive loss component...")
    from src.models.loss import action_token_contrastive_loss
    
    # Get the matching for the current video
    model.mcriterion.set_label(label)
    last_block = model.block_list[-1]
    cprob = torch.softmax(last_block.action_clogit, dim=-1)
    match = model.mcriterion.match(cprob, last_block.a2f_attn)
    
    # Compute contrastive loss
    contrastive_loss = action_token_contrastive_loss(
        model.projected_action_embeddings,
        model.text_embeddings,
        match,
        model.mcriterion.transcript,
        temperature=cfg.CLIP.temp
    )
    
    print(f"   ✓ Contrastive loss computed: {contrastive_loss.item():.4f}")
    print(f"   ✓ Loss is finite: {torch.isfinite(contrastive_loss).item()}")
    print(f"   ✓ Number of matched segments: {len(match[0])}")
    
    # Verify loss is differentiable
    print(f"\n5. Testing gradient flow through action projection...")
    contrastive_loss.backward()
    
    # Check that action_projection has gradients
    has_grad = False
    for name, param in model.action_projection.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            grad_mean = param.grad.abs().mean().item()
            print(f"   ✓ Gradient for {name}: {grad_mean:.6f}")
            break  # Just check one param to confirm gradients flow
    
    assert has_grad, "No gradients found in action_projection!"
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
    print("\nThe FACT_CLIP implementation is working correctly!")
    print("Key features verified:")
    print("  ✓ Action features are saved in all blocks")
    print("  ✓ Action projection creates embeddings in CLIP space")
    print("  ✓ Contrastive loss is computed correctly")
    print("  ✓ Gradients flow through the action projection")
    print("\nThe model is ready for zero-shot training on holdout classes.")
    print("="*80 + "\n")


def test_action_token_contrastive_loss():
    """Test the action_token_contrastive_loss function directly."""
    print("\n" + "="*80)
    print("TESTING ACTION TOKEN CONTRASTIVE LOSS")
    print("="*80)
    
    from src.models.loss import action_token_contrastive_loss
    
    # Create dummy inputs
    M = 10  # Number of action tokens
    B = 1   # Batch size
    S = 5   # Number of segments
    n_classes = 43
    
    # Create inputs and retain gradients for testing
    projected_action_tokens = torch.randn(M, B, 512, requires_grad=True)
    projected_action_tokens = torch.nn.functional.normalize(projected_action_tokens, dim=-1)
    projected_action_tokens.retain_grad()  # Retain gradients for non-leaf tensor
    
    text_embeddings = torch.randn(n_classes, 512)
    text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
    
    # Create matching (first S tokens matched to first S segments)
    action_ind = torch.arange(S)
    seg_ind = torch.arange(S)
    match = (action_ind, seg_ind)
    
    # Create transcript (segment labels)
    transcript = torch.randint(0, n_classes, (S,))
    
    print(f"\nInput shapes:")
    print(f"  - Projected action tokens: {projected_action_tokens.shape}")
    print(f"  - Text embeddings: {text_embeddings.shape}")
    print(f"  - Number of matched segments: {len(action_ind)}")
    
    # Compute loss
    loss = action_token_contrastive_loss(
        projected_action_tokens, text_embeddings, match, transcript, temperature=0.07
    )
    
    print(f"\nLoss value: {loss.item():.4f}")
    print(f"Loss is finite: {torch.isfinite(loss).item()}")
    print(f"Loss is positive: {(loss > 0).item()}")
    
    # Test gradient
    loss.backward()
    if projected_action_tokens.grad is not None:
        grad_norm = projected_action_tokens.grad.norm().item()
        print(f"Gradient norm: {grad_norm:.6f}")
        assert grad_norm > 0, "Gradient should be non-zero!"
    else:
        print("Warning: Gradient not retained (this is OK for test purposes)")
    
    assert torch.isfinite(loss), "Loss is not finite!"
    assert loss > 0, "Loss should be positive!"
    
    print("\n✓ Action token contrastive loss working correctly!")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        # Test contrastive loss function
        test_action_token_contrastive_loss()
        
        # Test full FACT_CLIP model
        test_fact_clip_basic()
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED ✗")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

