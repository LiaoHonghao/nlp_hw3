"""
Vocabulary implementation using Byte Pair Encoding (BPE) tokenizer
"""

from typing import List, Optional
import pickle
import os
from config import Config


class BPEVocabulary:
    """
    Vocabulary class using Hugging Face's tokenizers library for BPE tokenization
    """

    def __init__(self, name: str = "bpe_vocab"):
        """
        Initialize BPE Vocabulary

        Args:
            name: Name of the vocabulary
        """
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
        except ImportError:
            raise ImportError(
                "tokenizers library is required. Install with: pip install tokenizers"
            )

        self.name = name
        self.config = Config()

        # Special tokens
        self.pad_token = self.config.PAD_TOKEN
        self.unk_token = self.config.UNK_TOKEN
        self.bos_token = self.config.BOS_TOKEN
        self.eos_token = self.config.EOS_TOKEN

        # Tokenizer
        self.tokenizer = None
        self.is_trained = False

    def train(self,
              corpus_files: List[str],
              vocab_size: int = 30000,
              min_frequency: int = 2,
              special_tokens: Optional[List[str]] = None):
        """
        Train BPE tokenizer on corpus files

        Args:
            corpus_files: List of paths to corpus files (one sentence per line)
            vocab_size: Size of the vocabulary
            min_frequency: Minimum frequency for a subword to be included
            special_tokens: List of special tokens (defaults to PAD, UNK, BOS, EOS)
        """
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer

        if special_tokens is None:
            special_tokens = [
                self.pad_token, self.unk_token, self.bos_token, self.eos_token
            ]

        # Initialize BPE tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))

        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
            continuing_subword_prefix="##"
        )

        # Train tokenizer
        print(f"Training BPE tokenizer on {len(corpus_files)} files...")
        self.tokenizer.train(files=corpus_files, trainer=trainer)
        self.is_trained = True
        print(f"BPE vocabulary trained with {self.tokenizer.get_vocab_size()} tokens")

    def train_from_texts(self,
                         texts: List[str],
                         vocab_size: int = 30000,
                         min_frequency: int = 2,
                         special_tokens: Optional[List[str]] = None):
        """
        Train BPE tokenizer from list of texts

        Args:
            texts: List of text strings
            vocab_size: Size of the vocabulary
            min_frequency: Minimum frequency for a subword to be included
            special_tokens: List of special tokens
        """
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer

        if special_tokens is None:
            special_tokens = [
                self.pad_token, self.unk_token, self.bos_token, self.eos_token
            ]

        # Initialize BPE tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))

        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
            continuing_subword_prefix="##"
        )

        # Train tokenizer
        print(f"Training BPE tokenizer on {len(texts)} texts...")
        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        self.is_trained = True
        print(f"BPE vocabulary trained with {self.tokenizer.get_vocab_size()} tokens")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text to token IDs

        Args:
            text: Input text string
            add_bos: Whether to add beginning-of-sentence token
            add_eos: Whether to add end-of-sentence token

        Returns:
            List of token IDs
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before encoding")

        encoding = self.tokenizer.encode(text, add_special_tokens=add_bos or add_eos)

        # Manually handle BOS/EOS if needed
        tokens = []
        if add_bos:
            tokens.append(self.bos_idx)

        tokens.extend(encoding.ids)

        if add_eos:
            tokens.append(self.eos_idx)

        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs to text

        Args:
            token_ids: List of token IDs
            skip_special: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before decoding")

        # Get special token IDs
        special_ids = {self.pad_idx, self.bos_idx, self.eos_idx}

        # Filter out special tokens if requested
        if skip_special:
            token_ids = [tid for tid in token_ids if tid not in special_ids]

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special)

    def encode_batch(self, texts: List[str], max_length: Optional[int] = None) -> List[List[int]]:
        """
        Encode a batch of texts

        Args:
            texts: List of input text strings
            max_length: Maximum sequence length

        Returns:
            List of token ID lists
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before encoding")

        encodings = self.tokenizer.encode_batch(
            texts,
            max_length=max_length,
            padding=True if max_length else "max_length",
            truncation=True if max_length else False,
            add_special_tokens=True
        )

        return [enc.ids for enc in encodings]

    def decode_batch(self, batch_ids: List[List[int]]) -> List[str]:
        """
        Decode a batch of token IDs

        Args:
            batch_ids: List of token ID lists

        Returns:
            List of decoded text strings
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before decoding")

        return self.tokenizer.decode_batch(batch_ids, skip_special_tokens=True)

    def save(self, filepath: str):
        """
        Save trained tokenizer to file

        Args:
            filepath: Path to save the tokenizer
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before saving")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.tokenizer.save(filepath)
        print(f"BPE tokenizer saved to {filepath}")

    def load(self, filepath: str):
        """
        Load trained tokenizer from file

        Args:
            filepath: Path to the saved tokenizer
        """
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(filepath)
        self.is_trained = True
        print(f"BPE tokenizer loaded from {filepath} (size: {self.tokenizer.get_vocab_size()})")

    def get_vocab(self) -> dict:
        """
        Get vocabulary as dictionary

        Returns:
            Dictionary mapping tokens to IDs
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before getting vocab")

        return self.tokenizer.get_vocab()

    def __len__(self) -> int:
        """Get vocabulary size"""
        if not self.is_trained:
            return 0
        return self.tokenizer.get_vocab_size()

    @property
    def pad_idx(self) -> int:
        """Get pad token ID"""
        return self.tokenizer.token_to_id(self.pad_token)

    @property
    def unk_idx(self) -> int:
        """Get unknown token ID"""
        return self.tokenizer.token_to_id(self.unk_token)

    @property
    def bos_idx(self) -> int:
        """Get beginning-of-sentence token ID"""
        return self.tokenizer.token_to_id(self.bos_token)

    @property
    def eos_idx(self) -> int:
        """Get end-of-sentence token ID"""
        return self.tokenizer.token_to_id(self.eos_token)


def train_bpe_vocab_from_jsonl(corpus_path: str,
                               output_path: str,
                               field: str = "en",
                               vocab_size: int = 30000,
                               min_frequency: int = 2):
    """
    Helper function to train BPE vocabulary from JSONL corpus

    Args:
        corpus_path: Path to JSONL corpus file
        output_path: Path to save the trained tokenizer
        field: Field to use ('en' for English, 'zh' for Chinese)
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency threshold
    """
    import json

    texts = []
    print(f"Loading texts from {corpus_path}...")

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data[field])

    # Create and train vocabulary
    vocab = BPEVocabulary(f"BPE_{field.upper()}")
    vocab.train_from_texts(texts, vocab_size=vocab_size, min_frequency=min_frequency)

    # Save tokenizer
    vocab.save(output_path)

    return vocab


if __name__ == "__main__":
    # Example usage
    import json

    # Test with a small dataset
    corpus_path = Config.TRAIN_SMALL_PATH

    # Load sample texts
    texts = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Use first 100 lines for testing
                break
            data = json.loads(line)
            texts.append(data['en'])

    # Train English BPE vocabulary
    print("Training English BPE vocabulary...")
    en_vocab = BPEVocabulary("English_BPE")
    en_vocab.train_from_texts(texts, vocab_size=5000, min_frequency=2)

    # Test encoding/decoding
    test_text = "Hello world! This is a test sentence."
    print(f"\nOriginal text: {test_text}")

    encoded = en_vocab.encode(test_text, add_bos=True, add_eos=True)
    print(f"Encoded: {encoded}")

    decoded = en_vocab.decode(encoded)
    print(f"Decoded: {decoded}")

    # Save tokenizer
    en_vocab.save("./data/en_bpe_tokenizer.json")

    # Load tokenizer
    print("\nLoading tokenizer from file...")
    loaded_vocab = BPEVocabulary("Loaded_BPE")
    loaded_vocab.load("./data/en_bpe_tokenizer.json")

    # Test loaded tokenizer
    test_text2 = "This is another test."
    encoded2 = loaded_vocab.encode(test_text2)
    decoded2 = loaded_vocab.decode(encoded2)
    print(f"Test with loaded tokenizer: {test_text2} -> {decoded2}")
