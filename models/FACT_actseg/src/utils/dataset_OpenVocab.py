#!/usr/bin/python3

import numpy as np
import os
import torch
from ..home import get_project_base
from yacs.config import CfgNode
from .dataset import create_dataset, load_action_mapping

BASE = get_project_base()

# Import HAViD-specific text prompt generation
try:
    from .havid_text_prompts import (
        is_havid_label, 
        generate_action_prompt as generate_havid_prompt,
        generate_simple_prompt
    )
    HAVID_PROMPTS_AVAILABLE = True
except ImportError:
    HAVID_PROMPTS_AVAILABLE = False
    print("Warning: HAViD text prompts not available. Using simple formatting.")


def format_action_description(action_name, use_prompt=True, dataset_name=None):
    """
    Convert action class name to natural language description
    
    Examples:
        HAViD: "sshc1dh" → "a person screws a hex screw into cylinder plate hole 1 with a hex screwdriver"
        Breakfast: "crack_egg" → "a video of a person cracking an egg"
        
    Args:
        action_name: str - original action class name
        use_prompt: bool - whether to use prompt engineering template
        dataset_name: str - dataset name (to determine format type)
    
    Returns:
        str - formatted natural language description
    """
    # Check if it's a HAViD label and HAViD prompts are available
    if HAVID_PROMPTS_AVAILABLE and (
        (dataset_name and dataset_name.startswith('havid')) or 
        is_havid_label(action_name)
    ):
        # Use HAViD-specific prompt generation
        return generate_havid_prompt(action_name)
    
    # Fallback to simple prompt generation for other datasets
    if HAVID_PROMPTS_AVAILABLE:
        if use_prompt:
            return generate_simple_prompt(action_name, template="a video of a person {action}")
        else:
            return action_name.replace('_', ' ')
    
    # Legacy fallback if HAViD prompts not available
    action_clean = action_name.replace('_', ' ')
    
    # Handle numeric codes
    if action_name.startswith('S') and len(action_name) > 1 and action_name[1:].replace('_', '').replace('-', '').replace('.', '').isdigit():
        action_clean = f"performing action {action_name}"
    
    if use_prompt:
        return f"a video of a person {action_clean}"
    return action_clean


def precompute_text_embeddings(action_descriptions, clip_model_name, device="cuda"):
    """
    Pre-compute CLIP text embeddings for all action classes
    Run once before training for efficiency
    
    Args:
        action_descriptions: List[str] - list of action text descriptions
        clip_model_name: str - CLIP model name (e.g., "openai/clip-vit-b-32")
        device: str - device to run computation on
    
    Returns:
        torch.Tensor - text embeddings (C, 512) where C is number of classes
    """
    from transformers import CLIPModel, CLIPTokenizer
    
    print(f"Pre-computing text embeddings for {len(action_descriptions)} action classes...")
    
    clip = CLIPModel.from_pretrained(clip_model_name).eval().to(device)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    
    with torch.no_grad():
        inputs = tokenizer(
            action_descriptions,
            padding=True,
            truncation=True,
            max_length=77,  # CLIP max length
            return_tensors="pt"
        ).to(device)
        
        text_embeddings = clip.get_text_features(**inputs)
    
    print(f"Text embeddings computed: shape {text_embeddings.shape}")
    return text_embeddings.cpu()


def get_mapping_file_path(cfg):
    """
    Get the path to the mapping.txt file based on dataset configuration
    
    Args:
        cfg: YACS config node
    
    Returns:
        str - path to mapping.txt file
    """
    if cfg.dataset == "breakfast":
        return BASE + 'data/breakfast/mapping.txt'
    elif cfg.dataset == "gtea":
        return BASE + 'data/gtea/mapping.txt'
    elif cfg.dataset == "ego":
        return BASE + 'data/egoprocel/mapping.txt'
    elif cfg.dataset == "epic":
        return BASE + 'data/epic-kitchens/processed/mapping.txt'
    elif cfg.dataset.startswith("havid"):
        variant = cfg.dataset.replace("havid_", "")
        havid_base = BASE + 'data/HAViD/ActionSegmentation/data'
        return f'{havid_base}/{variant}/mapping.txt'
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")


def create_dataset_openvocab(cfg: CfgNode):
    """
    Create dataset with text descriptions for open-vocabulary training
    Uses existing I3D features from HAViD/Breakfast/etc.
    
    Args:
        cfg: YACS config node
    
    Returns:
        dataset: training dataset (with I3D features)
        test_dataset: test dataset (with I3D features)
        text_embeddings: pre-computed CLIP text embeddings (C, 512) or None
        action_descriptions: list of text descriptions (length C)
    """
    # Create standard dataset (loads I3D features)
    dataset, test_dataset = create_dataset(cfg)
    
    # Load action mapping
    map_fname = get_mapping_file_path(cfg)
    label2index, index2label = load_action_mapping(map_fname)
    
    # Generate text descriptions for all action classes
    # Pass dataset name to format_action_description for proper formatting
    action_descriptions = [
        format_action_description(
            index2label[i], 
            cfg.CLIP.use_prompt,
            dataset_name=cfg.dataset
        )
        for i in range(len(index2label))
    ]
    
    print(f"\nDataset: {cfg.dataset}")
    print(f"Generated {len(action_descriptions)} action descriptions:")
    # Print first few examples
    for i in range(min(10, len(action_descriptions))):
        print(f"  {index2label[i]:20s} -> {action_descriptions[i]}")
    if len(action_descriptions) > 10:
        print(f"  ... and {len(action_descriptions) - 10} more")
    
    # Pre-compute text embeddings if requested
    text_embeddings = None
    if cfg.CLIP.precompute_text:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            text_embeddings = precompute_text_embeddings(
                action_descriptions, 
                cfg.CLIP.model_name,
                device=device
            )
        except Exception as e:
            print(f"Warning: Failed to pre-compute text embeddings: {e}")
            print("Will compute them during training instead.")
            text_embeddings = None
    
    return dataset, test_dataset, text_embeddings, action_descriptions



