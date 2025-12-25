# Tokenizers Documentation

This directory contains advanced vocabulary implementations using modern tokenization techniques: **BPE (Byte Pair Encoding)** and **HanLP**.

## Files

- `vocabulary_bpe.py` - BPE-based vocabulary using Hugging Face's `tokenizers` library
- `vocabulary_hanlp.py` - HanLP-based vocabulary for Chinese/English text
- `tokenizer_examples.py` - Comprehensive examples and usage demonstrations
- `vocabulary.py` - Original word-based vocabulary (baseline)

## Installation

Before using the tokenizers, install the required dependencies:

```bash
# For BPE tokenizer
pip install tokenizers

# For HanLP tokenizer
pip install hanlp

# Or install all at once
pip install tokenizers hanlp
```

## Quick Start

### BPE Tokenizer (English)

```python
from data.vocabulary_bpe import BPEVocabulary

# Create and train BPE vocabulary
vocab = BPEVocabulary("English_BPE")
vocab.train_from_texts(
    texts=["Hello world!", "This is a test."],
    vocab_size=10000,
    min_frequency=2
)

# Encode/decode text
text = "Hello world!"
encoded = vocab.encode(text, add_bos=True, add_eos=True)
decoded = vocab.decode(encoded)
```

### HanLP Tokenizer (Chinese)

```python
from data.vocabulary_hanlp import HanLPVocabulary

# Create and train HanLP vocabulary
vocab = HanLPVocabulary("Chinese_HanLP", language="zh")
vocab.train_from_texts(
    texts=["你好世界！", "这是一个测试。"],
    min_freq=2,
    max_size=10000
)

# Encode/decode text
text = "你好世界！"
encoded = vocab.encode(text, add_bos=True, add_eos=True)
decoded = vocab.decode_to_text(encoded)
```

## Detailed Usage

### BPE Vocabulary

BPE (Byte Pair Encoding) is a subword tokenization method that splits text into smaller units based on frequency. It's particularly effective for handling out-of-vocabulary words.

#### Key Features

- **Subword tokenization**: Breaks words into meaningful subword units
- **Language-agnostic**: Works for any language
- **Compact vocabulary**: Handles rare words through composition
- **Fast encoding/decoding**: Optimized Rust implementation

#### Methods

```python
# Train from list of texts
vocab.train_from_texts(texts, vocab_size=30000, min_frequency=2)

# Train from corpus files
vocab.train(corpus_files=['train.txt', 'valid.txt'], vocab_size=30000)

# Encode single text
ids = vocab.encode("Hello world!", add_bos=True, add_eos=True)

# Decode to text
text = vocab.decode(ids)

# Batch encoding
batch_ids = vocab.encode_batch(["Hello", "World"], max_length=50)

# Save/load tokenizer
vocab.save("tokenizer.json")
vocab.load("tokenizer.json")
```

#### Helper Function

```python
from data.vocabulary_bpe import train_bpe_vocab_from_jsonl

# Train directly from JSONL corpus
vocab = train_bpe_vocab_from_jsonl(
    corpus_path="dataset/train_10k.jsonl",
    output_path="data/en_bpe.json",
    field="en",
    vocab_size=10000
)
```

### HanLP Vocabulary

HanLP is a Chinese NLP toolkit that provides sophisticated tokenization for both Chinese and English text.

#### Key Features

- **Chinese segmentation**: Uses state-of-the-art Chinese Word Segmentation models
- **Language detection**: Automatically configures based on language
- **Multiple models**: Supports various pre-trained tokenizers
- **Traditional & Simplified**: Handles both Chinese variants

#### Methods

```python
# Create vocabulary for Chinese
vocab = HanLPVocabulary("Chinese_HanLP", language="zh")

# Train from tokenized data
vocab.train(tokenized_data=[['你好', '世界'], ['这是', '测试']])

# Train from raw texts (automatic tokenization)
vocab.train_from_texts(texts, min_freq=2, max_size=10000)

# Train from JSONL corpus
vocab.train_from_corpus_file(
    corpus_path="dataset/train_10k.jsonl",
    field="zh",
    min_freq=2
)

# Encode text
ids = vocab.encode("你好世界！", add_bos=True, add_eos=True)

# Decode to tokens
tokens = vocab.decode(ids)

# Decode to text (language-aware joining)
text = vocab.decode_to_text(ids)
```

#### Helper Function

```python
from data.vocabulary_hanlp import train_hanlp_vocab_from_jsonl

# Train directly from JSONL corpus
vocab = train_hanlp_vocab_from_jsonl(
    corpus_path="dataset/train_10k.jsonl",
    output_path="data/zh_hanlp.pkl",
    field="zh",
    min_freq=2,
    max_size=10000
)
```

## API Comparison

| Feature | Original Vocabulary | BPE Vocabulary | HanLP Vocabulary |
|---------|-------------------|----------------|------------------|
| Tokenization | Pre-tokenized required | Automatic | Automatic |
| Subwords | ❌ | ✅ | ❌ |
| Chinese support | ✅ (basic) | ✅ | ✅ (excellent) |
| OOV handling | ❌ (UNK) | ✅ (subword composition) | ✅ (UNK) |
| Speed | Fast | Very fast | Fast |
| Model size | Large vocab | Small vocab | Medium vocab |
| Saved format | pickle | JSON | pickle |

## Examples

Run the comprehensive examples:

```bash
python data/tokenizer_examples.py
```

This will demonstrate:
1. BPE tokenizer for English
2. HanLP tokenizer for Chinese
3. Comparison of different approaches
4. Complete training pipeline
5. Custom configuration

## Integration with Training Scripts

### Using in train_rnn.py

```python
# Replace the original vocabulary with BPE/HanLP
from data.vocabulary_bpe import BPEVocabulary
from data.vocabulary_hanlp import HanLPVocabulary

# Load pre-trained tokenizers
en_vocab = BPEVocabulary("English")
en_vocab.load("./data/tokenizers/en_bpe.json")

zh_vocab = HanLPVocabulary("Chinese", language="zh")
zh_vocab.load("./data/tokenizers/zh_hanlp.pkl")

# Encode data
src = en_vocab.encode_batch(source_texts)
tgt = zh_vocab.encode_batch(target_texts)
```

### Using in train_transformer.py

```python
# Similar integration for Transformer models
# Tokenizers work with both RNN and Transformer architectures

# Encode with special tokens
src = en_vocab.encode_batch(source_texts, max_length=50)
tgt = zh_vocab.encode_batch(target_texts, max_length=50)

# Decode predictions
translated_texts = zh_vocab.decode_batch(predicted_ids)
```

## Advantages Over Original Vocabulary

### BPE Advantages

1. **Smaller vocabulary**: Fixed size (e.g., 10k) vs. potentially unlimited
2. **OOV handling**: Rare words split into known subwords
3. **Language-agnostic**: Works across multiple languages
4. **Better compression**: Subword units are more efficient

### HanLP Advantages

1. **Better Chinese tokenization**: Sophisticated CWS models
2. **Language-aware**: Optimized for Chinese linguistic features
3. **Multiple models**: Access to various pre-trained tokenizers
4. **Production-ready**: Battle-tested in real applications

## Performance Considerations

### Memory Usage

- **BPE**: Smaller vocab size reduces embedding layer memory
- **HanLP**: Moderate vocab size, efficient tokenization
- **Original**: Large vocab size for good coverage

### Training Speed

- **BPE**: Very fast (Rust-based implementation)
- **HanLP**: Fast (C++ backend with Python bindings)
- **Original**: Fast (simple frequency-based)

### Inference Speed

- **BPE**: Very fast (vectorized operations)
- **HanLP**: Fast (optimized tokenizers)
- **Original**: Fast (simple lookup)

## When to Use Each

### Use BPE when:
- Working with multiple languages
- Need to handle OOV words
- Want compact vocabulary
- Working with morphologically rich languages

### Use HanLP when:
- Processing Chinese text
- Need high-quality Chinese tokenization
- Working primarily with Chinese data
- Want access to advanced Chinese NLP features

### Use Original Vocabulary when:
- Already have tokenized data
- Simple word-level processing is sufficient
- Maximum interpretability needed
- No additional dependencies desired

## Troubleshooting

### Tokenizers Library Not Found
```bash
pip install tokenizers
```

### HanLP Download Issues
```python
import hanlp
hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG', download=True)
```

### Out of Memory
- Reduce `vocab_size` parameter
- Process data in smaller batches
- Use `max_samples` parameter when training from corpus

### Slow Training
- Reduce dataset size for testing
- Increase `min_frequency` threshold
- Use `train_from_corpus_file` with `max_samples` parameter

## References

- BPE: [Sennrich et al., 2016 - Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/)
- HanLP: [HanLP Documentation](https://hanlp.hankcs.com/)
- Tokenizers: [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers)

## License

Same as the main project: MIT License
