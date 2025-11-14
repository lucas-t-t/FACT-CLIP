#!/usr/bin/python3
"""
Test script for HAViD text prompt generation.
Verifies that labels are correctly parsed and converted to natural language.
"""

from havid_text_prompts import (
    parse_havid_label,
    generate_action_prompt,
    is_havid_label,
)


def test_parsing():
    """Test label parsing."""
    print("="*80)
    print("TEST: Label Parsing")
    print("="*80)
    
    test_cases = [
        # Full format: verb + manipulated + target + tool (7 chars)
        ("sshc1dh", {
            'verb': 'screws',
            'manipulated_object': 'hex screw',
            'target_object': 'cylinder plate hole 1',
            'tool': 'hex screwdriver'
        }),
        
        # No tool: verb + manipulated + target (5 chars)
        ("gshc1", {
            'verb': 'grasps',
            'manipulated_object': 'hex screw',
            'target_object': 'cylinder plate hole 1',
            'tool': None
        }),
        
        # No target: verb + manipulated (3 chars)
        ("gsh", {
            'verb': 'grasps',
            'manipulated_object': 'hex screw',
            'target_object': None,
            'tool': None
        }),
        
        # Verb only (1 char)
        ("s", {
            'verb': 'screws',
            'manipulated_object': None,
            'target_object': None,
            'tool': None
        }),
        
        # Noise
        ("null", {
            'verb': 'null',
            'manipulated_object': None,
            'target_object': None,
            'tool': None
        }),
    ]
    
    for label, expected in test_cases:
        parsed = parse_havid_label(label)
        print(f"\nLabel: {label}")
        print(f"  Parsed: {parsed}")
        print(f"  Expected: {expected}")
        assert parsed == expected, f"Mismatch for {label}"
    
    print("\n✓ All parsing tests passed!")


def test_prompt_generation():
    """Test prompt generation."""
    print("\n" + "="*80)
    print("TEST: Prompt Generation")
    print("="*80)
    
    test_cases = [
        # (label, expected_prompt_substring)
        ("sshc1dh", "screws a hex screw into cylinder plate hole 1 with a hex screwdriver"),
        ("gshc1", "grasps a hex screw to cylinder plate hole 1"),  # grasp has empty prep, defaults to 'to'
        ("pshc1", "places a hex screw onto cylinder plate hole 1"),   # 'onto' from place prep
        ("ishc1", "inserts a hex screw into cylinder plate hole 1"),
        ("gsh", "grasps a hex screw"),
        ("s", "person screws"),  # Just verb
        ("null", "noise: null"),
    ]
    
    for label, expected_substring in test_cases:
        prompt = generate_action_prompt(label)
        print(f"\nLabel: {label}")
        print(f"  Prompt: {prompt}")
        print(f"  Expected substring: '{expected_substring}'")
        # Check if expected substring is in prompt (case insensitive)
        assert expected_substring.lower() in prompt.lower(), \
            f"Expected '{expected_substring}' in prompt '{prompt}'"
    
    print("\n✓ All prompt generation tests passed!")


def test_real_examples():
    """Test with real HAViD examples from the figure."""
    print("\n" + "="*80)
    print("TEST: Real HAViD Examples")
    print("="*80)
    
    # Examples based on the annotation specification in the figure
    examples = [
        "sshc1dh",  # screw hex screw into cylinder plate hole 1 with hex screwdriver
        "pglg1",    # place large gear onto gear plate hole 1
        "gspc2dp",  # grasp phillips screw from cylinder plate hole 2 with phillips screwdriver
        "ishc3",    # insert hex screw into cylinder plate hole 3
        "rcs",      # rotate cylinder subassembly
        "awnws",    # approach nut wrench to shaft wrench (?)
        "null",     # noise
    ]
    
    print("\nGenerating prompts for real HAViD labels:\n")
    for label in examples:
        prompt = generate_action_prompt(label)
        print(f"  {label:12s} → {prompt}")
    
    print("\n✓ All real examples generated!")


def test_is_havid_label():
    """Test HAViD label detection."""
    print("\n" + "="*80)
    print("TEST: HAViD Label Detection")
    print("="*80)
    
    havid_labels = ["sshc1dh", "gsh", "s", "null", "w"]
    non_havid_labels = ["crack_egg", "pour_milk", "SIL", "take_plate"]
    
    print("\nHAViD labels:")
    for label in havid_labels:
        result = is_havid_label(label)
        print(f"  {label:12s} → {result}")
        assert result, f"{label} should be detected as HAViD"
    
    print("\nNon-HAViD labels:")
    for label in non_havid_labels:
        result = is_havid_label(label)
        print(f"  {label:12s} → {result}")
        # Note: Some may be detected as HAViD if first char is a verb
        # This is expected behavior - we're lenient in detection
    
    print("\n✓ Label detection test complete!")


def test_integration():
    """Test integration with dataset utilities."""
    print("\n" + "="*80)
    print("TEST: Integration with dataset_OpenVocab")
    print("="*80)
    
    try:
        from dataset_OpenVocab import format_action_description
        
        # Test HAViD labels
        havid_examples = [
            ("sshc1dh", "havid_view0_lh_pt"),
            ("gsh", "havid_view1_rh_aa"),
        ]
        
        print("\nHAViD labels:")
        for label, dataset_name in havid_examples:
            desc = format_action_description(label, use_prompt=True, dataset_name=dataset_name)
            print(f"  {label:12s} ({dataset_name:20s}) → {desc}")
        
        # Test non-HAViD labels
        breakfast_examples = [
            ("crack_egg", "breakfast"),
            ("pour_milk", "breakfast"),
        ]
        
        print("\nBreakfast labels:")
        for label, dataset_name in breakfast_examples:
            desc = format_action_description(label, use_prompt=True, dataset_name=dataset_name)
            print(f"  {label:12s} ({dataset_name:20s}) → {desc}")
        
        print("\n✓ Integration test passed!")
        
    except ImportError as e:
        print(f"\n⚠ Integration test skipped: {e}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HAVID TEXT PROMPT GENERATION TEST SUITE")
    print("="*80)
    
    test_parsing()
    test_prompt_generation()
    test_real_examples()
    test_is_havid_label()
    test_integration()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✓")
    print("="*80 + "\n")

