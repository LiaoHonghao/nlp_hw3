"""
Example usage of BPE and HanLP tokenizers for machine translation
This script demonstrates how to use both tokenizers for Chinese-English translation
"""

import json
import os
from typing import List, Tuple
from config import Config
from vocabulary_bpe import BPEVocabulary, train_bpe_vocab_from_jsonl
from vocabulary_hanlp import HanLPVocabulary, train_hanlp_vocab_from_jsonl


def example_bpe_usage():
    """Example: Using BPE tokenizer for English text"""
    print("=" * 80)
    print("EXAMPLE 1: BPE Tokenizer for English")
    print("=" * 80)

    corpus_path = Config.TRAIN_SMALL_PATH
    output_dir = "./data/tokenizers/"
    os.makedirs(output_dir, exist_ok=True)

    # Load sample texts
    texts = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 500:  # Use first 500 lines
                break
            data = json.loads(line)
            texts.append(data['en'])

    # Train BPE vocabulary
    print("\n1. Training BPE vocabulary for English...")
    bpe_vocab = BPEVocabulary("English_BPE")
    bpe_vocab.train_from_texts(texts, vocab_size=10000, min_frequency=2)

    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "en_bpe_tokenizer.json")
    bpe_vocab.save(tokenizer_path)

    # Test encoding/decoding
    print("\n2. Testing encoding/decoding...")
    test_sentences = [
        "Hello world!",
        "This is a machine translation project.",
        "We are using BPE tokenization."
    ]

    for sent in test_sentences:
        encoded = bpe_vocab.encode(sent, add_bos=True, add_eos=True)
        decoded = bpe_vocab.decode(encoded)
        print(f"  Original: {sent}")
        print(f"  Encoded:  {encoded[:10]}... (length: {len(encoded)})")
        print(f"  Decoded:  {decoded}")
        print()

    # Test with loaded tokenizer
    print("3. Testing with loaded tokenizer...")
    loaded_vocab = BPEVocabulary("Loaded_BPE")
    loaded_vocab.load(tokenizer_path)

    test_text = "BPE tokenization is subword-based."
    encoded = loaded_vocab.encode(test_text)
    decoded = loaded_vocab.decode(encoded)
    print(f"  Original: {test_text}")
    print(f"  Decoded:  {decoded}")


def example_hanlp_usage():
    """Example: Using HanLP tokenizer for Chinese text"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: HanLP Tokenizer for Chinese")
    print("=" * 80)

    corpus_path = Config.TRAIN_SMALL_PATH
    output_dir = "./data/tokenizers/"
    os.makedirs(output_dir, exist_ok=True)

    # Load sample texts
    texts = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 500:  # Use first 500 lines
                break
            data = json.loads(line)
            texts.append(data['zh'])

    # Train HanLP vocabulary
    print("\n1. Training HanLP vocabulary for Chinese...")
    hanlp_vocab = HanLPVocabulary("Chinese_HanLP", language="zh")
    hanlp_vocab.train_from_texts(texts, min_freq=2, max_size=10000)

    # Save vocabulary
    vocab_path = os.path.join(output_dir, "zh_hanlp_vocab.pkl")
    hanlp_vocab.save(vocab_path)

    # Test encoding/decoding
    print("\n2. Testing encoding/decoding...")
    test_sentences = [
        "你好世界！",
        "这是一个机器翻译项目。",
        "我们使用HanLP进行中文分词。"
    ]

    for sent in test_sentences:
        encoded = hanlp_vocab.encode(sent, add_bos=True, add_eos=True)
        decoded_tokens = hanlp_vocab.decode(encoded)
        decoded_text = hanlp_vocab.decode_to_text(encoded)
        print(f"  Original:     {sent}")
        print(f"  Encoded:      {encoded[:10]}... (length: {len(encoded)})")
        print(f"  Decoded txt:  {decoded_text}")
        print(f"  Decoded tok:  {decoded_tokens}")
        print()

    # Test with loaded vocabulary
    print("3. Testing with loaded vocabulary...")
    loaded_vocab = HanLPVocabulary("Loaded_HanLP", language="zh")
    loaded_vocab.load(vocab_path)

    test_text = "HanLP支持多种中文分词模式。"
    encoded = loaded_vocab.encode(test_text)
    decoded = loaded_vocab.decode_to_text(encoded)
    print(f"  Original: {test_text}")
    print(f"  Decoded:  {decoded}")


def example_comparison():
    """Compare different tokenization approaches"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Comparison of Tokenization Approaches")
    print("=" * 80)

    # Load sample data
    corpus_path = Config.TRAIN_SMALL_PATH
    zh_texts = []
    en_texts = []

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            data = json.loads(line)
            zh_texts.append(data['zh'])
            en_texts.append(data['en'])

    # Train both tokenizers on small dataset
    print("\n1. Training tokenizers...")
    bpe_en = BPEVocabulary("BPE_EN")
    bpe_en.train_from_texts(en_texts[:100], vocab_size=5000, min_frequency=2)

    hanlp_zh = HanLPVocabulary("HanLP_ZH", language="zh")
    hanlp_zh.train_from_texts(zh_texts[:100], min_freq=2, max_size=5000)

    # Compare on same sentences
    print("\n2. Comparing tokenization on sample sentences...")
    for i in range(min(3, len(zh_texts))):
        zh = zh_texts[i]
        en = en_texts[i]

        # Tokenize English with BPE
        bpe_tokens = bpe_en.encode(en)
        bpe_decoded = bpe_en.decode(bpe_tokens)

        # Tokenize Chinese with HanLP
        hanlp_tokens = hanlp_zh.encode(zh)
        hanlp_decoded = hanlp_zh.decode_to_text(hanlp_tokens)

        print(f"\n  Sentence {i+1}:")
        print(f"    Chinese:      {zh}")
        print(f"    HanLP tokens: {hanlp_tokens[:10]}... (count: {len(hanlp_tokens)})")
        print(f"    English:      {en}")
        print(f"    BPE tokens:   {bpe_tokens[:10]}... (count: {len(bpe_tokens)})")


def example_training_pipeline():
    """Example: Full training pipeline with both tokenizers"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Complete Training Pipeline")
    print("=" * 80)

    corpus_path = Config.TRAIN_SMALL_PATH
    output_dir = "./data/tokenizers/"
    os.makedirs(output_dir, exist_ok=True)

    # Use helper functions to train both tokenizers
    print("\n1. Training English BPE tokenizer...")
    en_bpe = train_bpe_vocab_from_jsonl(
        corpus_path=corpus_path,
        output_path=os.path.join(output_dir, "en_bpe.json"),
        field="en",
        vocab_size=8000,
        min_frequency=2
    )

    print("\n2. Training Chinese HanLP tokenizer...")
    zh_hanlp = train_hanlp_vocab_from_jsonl(
        corpus_path=corpus_path,
        output_path=os.path.join(output_dir, "zh_hanlp.pkl"),
        field="zh",
        min_frequency=2,
        max_size=8000
    )

    # Test with actual sentence pairs
    print("\n3. Testing with sentence pairs...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.readline())

    zh_text = data['zh']
    en_text = data['en']

    # Encode
    zh_encoded = zh_hanlp.encode(zh_text, add_bos=True, add_eos=True)
    en_encoded = en_bpe.encode(en_text, add_bos=True, add_eos=True)

    # Decode
    zh_decoded = zh_hanlp.decode_to_text(zh_encoded)
    en_decoded = en_bpe.decode(en_encoded)

    print(f"\n  Chinese:")
    print(f"    Original: {zh_text}")
    print(f"    Encoded:  {zh_encoded}")
    print(f"    Decoded:  {zh_decoded}")
    print(f"\n  English:")
    print(f"    Original: {en_text}")
    print(f"    Encoded:  {en_encoded}")
    print(f"    Decoded:  {en_decoded}")

    # Calculate compression ratio
    zh_compression = len(zh_text) / len(zh_encoded) if zh_encoded else 0
    en_compression = len(en_text.split()) / len(en_encoded) if en_encoded else 0

    print(f"\n4. Compression Ratios:")
    print(f"    Chinese: {zh_compression:.2f} chars/token")
    print(f"    English: {en_compression:.2f} words/token")


def example_custom_usage():
    """Example: Custom usage with specific parameters"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Custom Tokenizer Configuration")
    print("=" * 80)

    corpus_path = Config.TRAIN_SMALL_PATH
    output_dir = "./data/tokenizers/custom/"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    en_texts = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            data = json.loads(line)
            en_texts.append(data['en'])

    # Custom BPE configuration
    print("\n1. Training BPE with custom parameters...")
    print("   - Vocab size: 3000")
    print("   - Min frequency: 3")
    print("   - Subword prefix: ##")

    custom_bpe = BPEVocabulary("Custom_BPE")
    custom_bpe.train_from_texts(
        en_texts,
        vocab_size=3000,
        min_frequency=3
    )

    custom_bpe.save(os.path.join(output_dir, "custom_bpe.json"))

    # Compare with original vocabulary
    print("\n2. Comparison with original Vocabulary class...")
    from vocabulary import Vocabulary

    # Tokenize texts for original vocabulary
    from data.preprocessor import Preprocessor
    preprocessor = Preprocessor()
    data = preprocessor.load_and_preprocess(corpus_path, max_samples=200)
    en_tokens = [en for zh, en in data]

    # Train original vocabulary
    orig_vocab = Vocabulary("Original")
    orig_vocab.build_vocab(en_tokens)

    # Test on same text
    test_text = "This is a comparison test."

    # BPE tokenization
    bpe_ids = custom_bpe.encode(test_text)
    bpe_decoded = custom_bpe.decode(bpe_ids)

    # Original vocabulary tokenization (requires manual tokenization)
    print(f"\n  Original text: {test_text}")
    print(f"  BPE tokens:    {bpe_decoded}")
    print(f"\n  Vocabulary sizes:")
    print(f"    Original: {len(orig_vocab)} tokens")
    print(f"    BPE:      {len(custom_bpe)} tokens")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BPE AND HANLP TOKENIZER EXAMPLES")
    print("=" * 80)
    print("\nThese examples demonstrate the usage of BPE and HanLP tokenizers")
    print("for machine translation tasks.\n")

    # Run examples
    try:
        example_bpe_usage()
    except Exception as e:
        print(f"\nError in BPE example: {e}")
        print("Make sure tokenizers library is installed: pip install tokenizers")

    try:
        example_hanlp_usage()
    except Exception as e:
        print(f"\nError in HanLP example: {e}")
        print("Make sure HanLP library is installed: pip install hanlp")

    try:
        example_comparison()
    except Exception as e:
        print(f"\nError in comparison example: {e}")

    try:
        example_training_pipeline()
    except Exception as e:
        print(f"\nError in training pipeline example: {e}")

    try:
        example_custom_usage()
    except Exception as e:
        print(f"\nError in custom usage example: {e}")

    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETED")
    print("=" * 80)
    print("\nCheck ./data/tokenizers/ for saved tokenizers and vocabularies.")
    print("\nTo use these tokenizers in your training scripts:")
    print("  from data.vocabulary_bpe import BPEVocabulary")
    print("  from data.vocabulary_hanlp import HanLPVocabulary")
