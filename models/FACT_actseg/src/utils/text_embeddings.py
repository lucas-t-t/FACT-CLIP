#!/usr/bin/python3
"""
Utility functions for pre-computing and loading CLIP text embeddings.
"""

import os
import torch
from typing import List, Optional, Dict
from yacs.config import CfgNode

try:
    from transformers import CLIPModel, CLIPTokenizer
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: transformers library not available. CLIP functionality will be disabled.")


def generate_text_descriptions(cfg: CfgNode, label2index: Dict[str, int], index2label: Dict[int, str]) -> List[str]:
    """
    Generate text descriptions for all action classes.
    
    Args:
        cfg: Configuration node
        label2index: Mapping from label string to class index
        index2label: Mapping from class index to label string
    
    Returns:
        List of text descriptions ordered by class index
    """
    from .havid_text_prompts import generate_action_prompt, is_havid_label
    
    n_classes = len(index2label)
    descriptions = []
    
    for i in range(n_classes):
        label = index2label.get(i, f"action_{i}")
        
        # Check if HAViD dataset and use appropriate prompt generation
        if cfg.dataset.startswith('havid') and is_havid_label(label):
            if cfg.CLIP.use_prompt:
                desc = generate_action_prompt(label)
            else:
                desc = label  # Use raw label
        else:
            # For non-HAViD datasets, use simple format
            desc = label.replace('_', ' ')
            if cfg.CLIP.use_prompt:
                desc = f"a person {desc}"
        
        descriptions.append(desc)
    
    return descriptions


def precompute_text_embeddings(
    text_descriptions: List[str],
    clip_model_name: str = "openai/clip-vit-base-patch32",
    device: str = "cuda",
    save_path: Optional[str] = None
) -> torch.Tensor:
    """
    Pre-compute CLIP text embeddings for action descriptions.
    
    Args:
        text_descriptions: List of text descriptions
        clip_model_name: CLIP model name
        device: Device to run computation on
        save_path: Optional path to save embeddings
    
    Returns:
        text_embeddings: (n_classes, 512) tensor of CLIP embeddings
    """
    if not CLIP_AVAILABLE:
        raise ImportError("transformers library required for CLIP functionality")
    
    print(f"Pre-computing text embeddings for {len(text_descriptions)} action classes...")
    print(f"Using CLIP model: {clip_model_name}")
    
    # Map common model names to correct transformers identifiers
    model_name_map = {
        "openai/clip-vit-b-32": "openai/clip-vit-base-patch32",
        "ViT-B/32": "openai/clip-vit-base-patch32",
        "clip-vit-b-32": "openai/clip-vit-base-patch32",
    }
    
    actual_model_name = model_name_map.get(clip_model_name, clip_model_name)
    
    # Load CLIP model
    try:
        clip_model = CLIPModel.from_pretrained(actual_model_name).eval().to(device)
        tokenizer = CLIPTokenizer.from_pretrained(actual_model_name)
    except Exception as e:
        # Try alternative identifier
        if actual_model_name != "openai/clip-vit-base-patch32":
            print(f"Warning: Failed to load {actual_model_name}, trying openai/clip-vit-base-patch32")
            actual_model_name = "openai/clip-vit-base-patch32"
            clip_model = CLIPModel.from_pretrained(actual_model_name).eval().to(device)
            tokenizer = CLIPTokenizer.from_pretrained(actual_model_name)
        else:
            raise e
    
    # Tokenize text descriptions
    with torch.no_grad():
        inputs = tokenizer(
            text_descriptions,
            padding=True,
            truncation=True,
            max_length=77,  # CLIP max length
            return_tensors="pt"
        ).to(device)
        
        # Get text features
        text_embeddings = clip_model.get_text_features(**inputs)
        
        # Normalize embeddings
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
    
    print(f"Text embeddings computed: shape {text_embeddings.shape}")
    
    # Save if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save(text_embeddings.cpu(), save_path)
        print(f"Saved text embeddings to {save_path}")
    
    return text_embeddings.cpu()


def load_text_embeddings(emb_path: str, device: str = "cuda") -> torch.Tensor:
    """
    Load pre-computed text embeddings from file.
    
    Args:
        emb_path: Path to saved embeddings file
        device: Device to load embeddings to
    
    Returns:
        text_embeddings: (n_classes, 512) tensor of CLIP embeddings
    """
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Text embeddings file not found: {emb_path}")
    
    embeddings = torch.load(emb_path, map_location=device)
    print(f"Loaded text embeddings from {emb_path}: shape {embeddings.shape}")
    return embeddings


def get_or_compute_text_embeddings(
    cfg: CfgNode,
    label2index: Dict[str, int],
    index2label: Dict[int, str],
    device: str = "cuda"
) -> torch.Tensor:
    """
    Get text embeddings, loading from file if exists, otherwise computing and saving.
    
    Args:
        cfg: Configuration node
        label2index: Mapping from label string to class index
        index2label: Mapping from class index to label string
        device: Device to run computation on
    
    Returns:
        text_embeddings: (n_classes, 512) tensor of CLIP embeddings
    """
    # Determine save path
    if cfg.CLIP.text_emb_path is not None:
        emb_path = cfg.CLIP.text_emb_path
    else:
        # Default path based on dataset
        from ..home import get_project_base
        BASE = get_project_base()
        # Use dataset-specific path for HAViD
        if cfg.dataset.startswith('havid'):
            variant = cfg.dataset.replace("havid_", "")
            emb_path = os.path.join(BASE, "data", "HAViD", "ActionSegmentation", "data", variant, f"{cfg.dataset}_text_embeddings.pt")
        else:
            emb_path = os.path.join(BASE, "data", f"{cfg.dataset}_text_embeddings.pt")
    
    # Try to load existing embeddings
    if os.path.exists(emb_path) and cfg.CLIP.precompute_text:
        try:
            return load_text_embeddings(emb_path, device)
        except Exception as e:
            print(f"Warning: Failed to load embeddings from {emb_path}: {e}")
            print("Re-computing embeddings...")
    
    # Generate text descriptions
    text_descriptions = generate_text_descriptions(cfg, label2index, index2label)
    
    # Print sample descriptions
    print(f"\nGenerated {len(text_descriptions)} text descriptions:")
    for i in range(min(5, len(text_descriptions))):
        print(f"  Class {i}: {text_descriptions[i]}")
    if len(text_descriptions) > 5:
        print(f"  ... and {len(text_descriptions) - 5} more")
    
    # Compute embeddings
    embeddings = precompute_text_embeddings(
        text_descriptions,
        clip_model_name=cfg.CLIP.model_name,
        device=device,
        save_path=emb_path if cfg.CLIP.precompute_text else None
    )
    
    return embeddings

