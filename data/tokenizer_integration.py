"""
Integration helper for BPE and HanLP tokenizers
This module provides easy-to-use functions for integrating tokenizers into training scripts
"""

import json
import os
from typing import List, Tuple, Optional, Union
from config import Config


class TokenizerManager:
    """
    Unified interface for managing multiple tokenizers
    """

    def __init__(self):
        self.en_vocab = None
        self.zh_vocab = None
        self.config = Config()

    def load_tokenizers(self,
                        en_tokenizer_path: Optional[str] = None,
                        zh_tokenizer_path: Optional[str] = None,
                        en_tokenizer_type: str = "bpe",
                        zh_tokenizer_type: str = "hanlp"):
        """
        Load pre-trained tokenizers

        Args:
            en_tokenizer_path: Path to English tokenizer
            zh_tokenizer_path: Path to Chinese tokenizer
            en_tokenizer_type: Type of English tokenizer ('bpe' or 'original')
            zh_tokenizer_type: Type of Chinese tokenizer ('hanlp' or 'original')
        """
        # Load English tokenizer
        if en_tokenizer_path and os.path.exists(en_tokenizer_path):
            if en_tokenizer_type == "bpe":
                from vocabulary_bpe import BPEVocabulary
                self.en_vocab = BPEVocabulary("English_BPE")
                self.en_vocab.load(en_tokenizer_path)
            elif en_tokenizer_type == "original":
                from vocabulary import Vocabulary
                self.en_vocab = Vocabulary("English_Original")
                self.en_vocab.load(en_tokenizer_path)

        # Load Chinese tokenizer
        if zh_tokenizer_path and os.path.exists(zh_tokenizer_path):
            if zh_tokenizer_type == "hanlp":
                from vocabulary_hanlp import HanLPVocabulary
                self.zh_vocab = HanLPVocabulary("Chinese_HanLP", language="zh")
                self.zh_vocab.load(zh_tokenizer_path)
            elif zh_tokenizer_type == "original":
                from vocabulary import Vocabulary
                self.zh_vocab = Vocabulary("Chinese_Original")
                self.zh_vocab.load(zh_tokenizer_path)

    def train_tokenizers(self,
                         corpus_path: str,
                         en_vocab_size: int = 10000,
                         zh_vocab_size: int = 10000,
                         min_freq: int = 2,
                         output_dir: str = "./data/tokenizers/"):
        """
        Train both English and Chinese tokenizers

        Args:
            corpus_path: Path to JSONL corpus
            en_vocab_size: Size of English vocabulary
            zh_vocab_size: Size of Chinese vocabulary
            min_freq: Minimum frequency threshold
            output_dir: Directory to save tokenizers
        """
        os.makedirs(output_dir, exist_ok=True)

        # Train English BPE tokenizer
        print("Training English BPE tokenizer...")
        from vocabulary_bpe import train_bpe_vocab_from_jsonl
        self.en_vocab = train_bpe_vocab_from_jsonl(
            corpus_path=corpus_path,
            output_path=os.path.join(output_dir, "en_bpe.json"),
            field="en",
            vocab_size=en_vocab_size,
            min_frequency=min_freq
        )

        # Train Chinese HanLP tokenizer
        print("Training Chinese HanLP tokenizer...")
        from vocabulary_hanlp import train_hanlp_vocab_from_jsonl
        self.zh_vocab = train_hanlp_vocab_from_jsonl(
            corpus_path=corpus_path,
            output_path=os.path.join(output_dir, "zh_hanlp.pkl"),
            field="zh",
            min_freq=min_freq,
            max_size=zh_vocab_size
        )

    def encode_batch(self,
                     en_texts: Optional[List[str]] = None,
                     zh_texts: Optional[List[str]] = None,
                     max_length: Optional[int] = None) -> Tuple[Optional[List[List[int]]], Optional[List[List[int]]]]:
        """
        Encode batch of texts

        Args:
            en_texts: List of English texts
            zh_texts: List of Chinese texts
            max_length: Maximum sequence length

        Returns:
            Tuple of (encoded_en, encoded_zh)
        """
        encoded_en = None
        encoded_zh = None

        if en_texts and self.en_vocab:
            if hasattr(self.en_vocab, 'encode_batch'):
                encoded_en = self.en_vocab.encode_batch(en_texts, max_length=max_length)
            else:
                # Original vocabulary - requires pre-tokenization
                encoded_en = [self.en_vocab.encode(tokens) for tokens in en_texts]

        if zh_texts and self.zh_vocab:
            if hasattr(self.zh_vocab, 'encode_batch'):
                encoded_zh = self.zh_vocab.encode_batch(zh_texts, max_length=max_length)
            else:
                # Original vocabulary
                encoded_zh = [self.zh_vocab.encode(tokens) for tokens in zh_texts]

        return encoded_en, encoded_zh

    def decode_batch(self,
                     en_ids: Optional[List[List[int]]] = None,
                     zh_ids: Optional[List[List[int]]] = None) -> Tuple[Optional[List[str]], Optional[List[str]]]:
        """
        Decode batch of token IDs

        Args:
            en_ids: List of English token IDs
            zh_ids: List of Chinese token IDs

        Returns:
            Tuple of (decoded_en, decoded_zh)
        """
        decoded_en = None
        decoded_zh = None

        if en_ids and self.en_vocab:
            if hasattr(self.en_vocab, 'decode_batch'):
                decoded_en = self.en_vocab.decode_batch(en_ids)
            else:
                # Original vocabulary
                decoded_en = [self.en_vocab.decode(ids) for ids in en_ids]
                decoded_en = [' '.join(tokens) for tokens in decoded_en]

        if zh_ids and self.zh_vocab:
            if hasattr(self.zh_vocab, 'decode_batch'):
                decoded_zh = self.zh_vocab.decode_batch(zh_ids)
                # Convert to text
                if hasattr(self.zh_vocab, 'language') and self.zh_vocab.language == 'zh':
                    decoded_zh = [''.join(tokens) for tokens in decoded_zh]
            else:
                # Original vocabulary
                decoded_zh = [self.zh_vocab.decode(ids) for ids in zh_ids]
                decoded_zh = [''.join(tokens) for tokens in decoded_zh]

        return decoded_en, decoded_zh

    def get_vocab_sizes(self) -> Tuple[int, int]:
        """Get vocabulary sizes"""
        en_size = len(self.en_vocab) if self.en_vocab else 0
        zh_size = len(self.zh_vocab) if self.zh_vocab else 0
        return en_size, zh_size

    def get_special_token_ids(self) -> dict:
        """Get special token IDs for both vocabularies"""
        return {
            'en': {
                'pad_idx': self.en_vocab.pad_idx if self.en_vocab else 0,
                'unk_idx': self.en_vocab.unk_idx if self.en_vocab else 1,
                'bos_idx': self.en_vocab.bos_idx if self.en_vocab else 2,
                'eos_idx': self.en_vocab.eos_idx if self.en_vocab else 3,
            },
            'zh': {
                'pad_idx': self.zh_vocab.pad_idx if self.zh_vocab else 0,
                'unk_idx': self.zh_vocab.unk_idx if self.zh_vocab else 1,
                'bos_idx': self.zh_vocab.bos_idx if self.zh_vocab else 2,
                'eos_idx': self.zh_vocab.eos_idx if self.zh_vocab else 3,
            }
        }


def load_data_for_training(corpus_path: str,
                           tokenizer_manager: TokenizerManager,
                           max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load and encode data for training

    Args:
        corpus_path: Path to JSONL corpus
        tokenizer_manager: TokenizerManager with loaded tokenizers
        max_samples: Maximum number of samples to load

    Returns:
        Tuple of (source_texts, target_texts)
    """
    source_texts = []
    target_texts = []

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line)
            source_texts.append(data['en'])
            target_texts.append(data['zh'])

    return source_texts, target_texts


def prepare_dataloader(source_texts: List[str],
                       target_texts: List[str],
                       tokenizer_manager: TokenizerManager,
                       batch_size: int = 64,
                       max_length: int = 50):
    """
    Prepare PyTorch DataLoader with tokenization

    Args:
        source_texts: List of English source texts
        target_texts: List of Chinese target texts
        tokenizer_manager: TokenizerManager with loaded tokenizers
        batch_size: Batch size
        max_length: Maximum sequence length

    Returns:
        PyTorch DataLoader
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    # Encode texts
    en_ids, zh_ids = tokenizer_manager.encode_batch(
        en_texts=source_texts,
        zh_texts=target_texts,
        max_length=max_length
    )

    # Pad sequences
    def pad_sequences(sequences, pad_token_id):
        """Pad sequences to same length"""
        max_len = max(len(seq) for seq in sequences)
        padded = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)

        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        return padded

    special_ids = tokenizer_manager.get_special_token_ids()
    en_padded = pad_sequences(en_ids, special_ids['en']['pad_idx'])
    zh_padded = pad_sequences(zh_ids, special_ids['zh']['pad_idx'])

    # Create dataset and dataloader
    dataset = TensorDataset(en_padded, zh_padded)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def quick_setup(corpus_path: str,
                output_dir: str = "./data/tokenizers/",
                en_vocab_size: int = 10000,
                zh_vocab_size: int = 10000,
                min_freq: int = 2):
    """
    Quick setup: Train and save tokenizers

    Args:
        corpus_path: Path to JSONL corpus
        output_dir: Directory to save tokenizers
        en_vocab_size: English vocabulary size
        zh_vocab_size: Chinese vocabulary size
        min_freq: Minimum frequency

    Returns:
        TokenizerManager with trained tokenizers
    """
    manager = TokenizerManager()
    manager.train_tokenizers(
        corpus_path=corpus_path,
        en_vocab_size=en_vocab_size,
        zh_vocab_size=zh_vocab_size,
        min_freq=min_freq,
        output_dir=output_dir
    )
    return manager


if __name__ == "__main__":
    # Example usage
    corpus_path = Config.TRAIN_SMALL_PATH
    output_dir = "./data/tokenizers/quick_setup/"

    print("=" * 80)
    print("Quick Setup Example")
    print("=" * 80)

    # Quick setup
    print("\n1. Training tokenizers...")
    manager = quick_setup(
        corpus_path=corpus_path,
        output_dir=output_dir,
        en_vocab_size=5000,
        zh_vocab_size=5000
    )

    # Load data
    print("\n2. Loading and encoding data...")
    source_texts, target_texts = load_data_for_training(
        corpus_path,
        manager,
        max_samples=10
    )

    print(f"Loaded {len(source_texts)} sentence pairs")

    # Prepare dataloader
    print("\n3. Preparing DataLoader...")
    dataloader = prepare_dataloader(
        source_texts,
        target_texts,
        manager,
        batch_size=4
    )

    # Test batch
    print("\n4. Testing batch...")
    for batch_idx, (en_batch, zh_batch) in enumerate(dataloader):
        print(f"  Batch {batch_idx}:")
        print(f"    English shape: {en_batch.shape}")
        print(f"    Chinese shape: {zh_batch.shape}")
        print(f"    Sample English IDs: {en_batch[0][:10]}")
        print(f"    Sample Chinese IDs: {zh_batch[0][:10]}")
        break

    # Decode sample
    print("\n5. Decoding sample...")
    en_ids = en_batch[0].tolist()
    zh_ids = zh_batch[0].tolist()

    decoded_en, decoded_zh = manager.decode_batch([en_ids], [zh_ids])
    print(f"  Decoded English: {decoded_en[0]}")
    print(f"  Decoded Chinese: {decoded_zh[0]}")

    print("\n" + "=" * 80)
    print("Setup complete! Tokenizers saved to:", output_dir)
    print("=" * 80)
