import torch
import sys
import os

# Add parent directory to path to handle imports if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from havid_text_prompts import (
        parse_havid_label, 
        generate_action_prompt,
        VERB_MAP, 
        MANIPULATED_OBJECT_MAP, 
        TARGET_OBJECT_MAP, 
        TOOL_MAP
    )
except ImportError:
    # Fallback for running from different context
    from src.utils.havid_text_prompts import (
        parse_havid_label, 
        generate_action_prompt,
        VERB_MAP, 
        MANIPULATED_OBJECT_MAP, 
        TARGET_OBJECT_MAP, 
        TOOL_MAP
    )

try:
    from transformers import CLIPModel, CLIPTokenizer
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

def print_breakdown(label):
    parsed = parse_havid_label(label)
    prompt = generate_action_prompt(label)
    print(f"\nLabel: {label}")
    print(f"Generated Prompt: {prompt}")
    print("Breakdown:")
    
    lab = label.lower().strip()
    if lab in ['null', 'w']:
         print(f"  {lab} -> {parsed['verb']}")
         return

    # Verb
    if len(lab) >= 1:
        v_char = lab[0]
        v_name = VERB_MAP.get(v_char, v_char)
        # Ensure we show the mapping even if parsed returned something else (though it shouldn't)
        print(f"  {v_char} -> {v_name}")
    
    # Manipulated Object
    if len(lab) >= 3:
        m_code = lab[1:3]
        m_name = MANIPULATED_OBJECT_MAP.get(m_code, m_code)
        print(f"  {m_code} -> {m_name}")
    
    # Target Object
    if len(lab) >= 5:
        t_code = lab[3:5]
        t_name = TARGET_OBJECT_MAP.get(t_code, t_code)
        print(f"  {t_code} -> {t_name}")
        
    # Tool
    if len(lab) >= 7:
        tool_code = lab[5:7]
        tool_name = TOOL_MAP.get(tool_code, tool_code)
        print(f"  {tool_code} -> {tool_name}")

def test_clip_embedding(prompt):
    if not CLIP_AVAILABLE:
        print("\n[WARNING] transformers library not found. Skipping CLIP embedding test.")
        return

    print("\n" + "="*60)
    print("Testing CLIP Embedding")
    print("="*60)
    
    model_name = "openai/clip-vit-base-patch32"
    print(f"Loading model: {model_name} ...")
    
    try:
        model = CLIPModel.from_pretrained(model_name)
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    print("Embedding prompt...")
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    emb = text_features[0].tolist()
    
    print(f"\nOriginal Prompt: \"{prompt}\"")
    print(f"Embedding Vector (first 5 dims): [{', '.join(f'{x:.4f}' for x in emb[:5])}, ...]")
    print(f"Vector Length: {len(emb)}")
    
    # Reconstruction / Retrieval Test
    print("\nPerforming 'Convert Back' Verification (Retrieval Test)...")
    print("Since CLIP embeddings are dense vectors, we verify by checking if the embedding")
    print("retrieves the original text from a set of candidate distractors.")
    
    distractors = [
        "a person walking down the street",
        "a chef cooking a meal in the kitchen",
        "a robot assembling a car part",
        "someone writing python code on a computer",
        prompt  # The correct prompt
    ]
    
    print("\nCandidates:")
    for d in distractors:
        print(f"  - {d}")
        
    all_inputs = tokenizer(distractors, padding=True, return_tensors="pt")
    with torch.no_grad():
        all_features = model.get_text_features(**all_inputs)
        all_features = all_features / all_features.norm(dim=-1, keepdim=True)
        
    # Cosine similarity
    sims = (text_features @ all_features.T).squeeze()
    
    print("\nSimilarity Scores:")
    for i, (txt, score) in enumerate(zip(distractors, sims)):
        mark = " (* Match)" if txt == prompt else ""
        print(f"  {score:.4f}: {txt}{mark}")
        
    best_idx = sims.argmax().item()
    retrieved = distractors[best_idx]
    
    print(f"\nResult: Best match is \"{retrieved}\"")
    
    if retrieved == prompt:
        print("SUCCESS: Embedding correctly maps back to original text.")
    else:
        print("FAILURE: Embedding mapped to wrong text.")

if __name__ == "__main__":
    print("="*60)
    print("Testing HA-ViD Label Parsing & Prompt Generation")
    print("="*60)
    
    # Test labels
    test_labels = ["sshc1dh", "pglg1", "ishc3"]
    
    for l in test_labels:
        print_breakdown(l)
        print("-" * 40)
    
    # Use the first prompt for CLIP test
    first_prompt = generate_action_prompt(test_labels[0])
    test_clip_embedding(first_prompt)

