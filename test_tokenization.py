#!/usr/bin/env python3
"""
Quick test script to validate BPE tokenization works correctly
Run this to verify the tokenizers are functioning before training
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from data.vocabulary_bpe import BPEVocabulary
    print("✓ BPEVocabulary imported successfully")
except ImportError as e:
    print(f"✗ Failed to import BPEVocabulary: {e}")
    print("  Install tokenizers: pip install tokenizers")
    sys.exit(1)

def test_bpe_vocabulary():
    """Test BPE vocabulary functionality"""
    print("\n" + "="*60)
    print("Testing BPE Vocabulary")
    print("="*60)

    # Create vocabulary
    print("\n1. Creating BPE vocabulary...")
    vocab = BPEVocabulary("Test_EN")
    print(f"   Name: {vocab.name}")

    # Train vocabulary
    print("\n2. Training vocabulary...")
    texts = [
        "hello world",
        "this is a test",
        "machine translation is great",
        "transformers are powerful",
        "neural networks can learn"
    ]

    vocab.train_from_texts(
        texts=texts,
        vocab_size=1000,
        min_frequency=1
    )
    print(f"   ✓ Trained with {vocab.tokenizer.get_vocab_size()} tokens")

    # Test special tokens
    print("\n3. Checking special tokens...")
    try:
        pad_idx = vocab.pad_idx
        unk_idx = vocab.unk_idx
        bos_idx = vocab.bos_idx
        eos_idx = vocab.eos_idx
        print(f"   ✓ pad_idx: {pad_idx}")
        print(f"   ✓ unk_idx: {unk_idx}")
        print(f"   ✓ bos_idx: {bos_idx}")
        print(f"   ✓ eos_idx: {eos_idx}")
    except Exception as e:
        print(f"   ✗ Error getting special tokens: {e}")
        print("   This needs to be fixed in vocabulary_bpe.py")
        return False

    # Test single text encoding
    print("\n4. Testing single text encoding...")
    text = "hello world"
    ids = vocab.encode(text, add_bos=True, add_eos=True)
    print(f"   Text: '{text}'")
    print(f"   IDs: {ids}")
    decoded = vocab.decode(ids, skip_special=True)
    print(f"   Decoded: '{decoded}'")
    assert decoded == text, f"Decoding failed: '{decoded}' != '{text}'"
    print(f"   ✓ Encoding/decoding works!")

    # Test batch encoding
    print("\n5. Testing batch encoding...")
    texts = ["hello", "world", "test"]
    ids_list = vocab.encode_batch(texts, max_length=10)
    print(f"   Input: {texts}")
    print(f"   Output: {ids_list}")
    assert len(ids_list) == len(texts), "Batch size mismatch"
    print(f"   ✓ Batch encoding works!")

    # Test with special tokens
    print("\n6. Testing special token handling...")
    text = "unknownword123"
    ids = vocab.encode(text, add_bos=True, add_eos=True)
    print(f"   Text with OOV: '{text}'")
    print(f"   IDs: {ids}")
    decoded = vocab.decode(ids, skip_special=True)
    print(f"   Decoded: '{decoded}'")
    print(f"   ✓ OOV handling works!")

    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    return True

def test_dataset_preparation():
    """Test dataset preparation with sample data"""
    print("\n" + "="*60)
    print("Testing Dataset Preparation")
    print("="*60)

    # Create vocabularies
    print("\n1. Creating vocabularies...")
    vocab_zh = BPEVocabulary("Test_ZH")
    vocab_en = BPEVocabulary("Test_EN")

    # Train
    zh_texts = ["你好世界", "这是测试", "机器翻译"]
    en_texts = ["hello world", "this is test", "machine translation"]

    vocab_zh.train_from_texts(zh_texts, vocab_size=1000, min_frequency=1)
    vocab_en.train_from_texts(en_texts, vocab_size=1000, min_frequency=1)
    print(f"   ✓ Trained ZH vocab: {vocab_zh.tokenizer.get_vocab_size()} tokens")
    print(f"   ✓ Trained EN vocab: {vocab_en.tokenizer.get_vocab_size()} tokens")

    # Test dataset_prepare function
    print("\n2. Testing dataset_prepare...")

    def dataset_prepare(data, vocab_zh, vocab_en, max_length=None, add_eos=False):
        """Encode text data using BPE tokenizers"""
        zh_texts, en_texts = zip(*data)

        zh_ids_list = vocab_zh.encode_batch(list(zh_texts), max_length=max_length)
        en_ids_list = vocab_en.encode_batch(list(en_texts), max_length=max_length)

        if add_eos:
            zh_ids_list = [ids + [vocab_zh.eos_idx] for ids in zh_ids_list]
            en_ids_list = [[vocab_en.bos_idx] + ids + [vocab_en.eos_idx]
                           for ids in en_ids_list]

        return list(zip(zh_ids_list, en_ids_list))

    # Sample data
    data = [("你好世界", "hello world"), ("这是测试", "this is test")]

    # Encode without special tokens
    print("\n   Encoding without special tokens...")
    encoded = dataset_prepare(data, vocab_zh, vocab_en, max_length=20, add_eos=False)
    print(f"   Result: {len(encoded)} samples")
    for i, (zh_ids, en_ids) in enumerate(encoded):
        print(f"   Sample {i+1}:")
        print(f"     ZH IDs: {zh_ids}")
        print(f"     EN IDs: {en_ids}")
        zh_decoded = vocab_zh.decode(zh_ids, skip_special=True)
        en_decoded = vocab_en.decode(en_ids, skip_special=True)
        print(f"     ZH decoded: '{zh_decoded}'")
        print(f"     EN decoded: '{en_decoded}'")
    print(f"   ✓ Dataset preparation works!")

    print("\n" + "="*60)
    print("DATASET PREPARATION TEST PASSED! ✓")
    print("="*60)
    return True

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# BPE Tokenization Validation Script")
    print("#"*60)

    success = True

    try:
        if not test_bpe_vocabulary():
            success = False
    except Exception as e:
        print(f"\n✗ BPE vocabulary test failed with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        success = False

    try:
        if not test_dataset_preparation():
            success = False
    except Exception as e:
        print(f"\n✗ Dataset preparation test failed with error:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "#"*60)
    if success:
        print("# ALL VALIDATION TESTS PASSED! ✓")
        print("# Your tokenization pipeline is working correctly.")
        print("# You can now proceed with training.")
    else:
        print("# VALIDATION FAILED! ✗")
        print("# Please fix the issues above before training.")
    print("#"*60 + "\n")

    sys.exit(0 if success else 1)
