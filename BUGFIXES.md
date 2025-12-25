# Bug Fixes for Transformer Training

## Critical Bugs Found

### 1. Fix Import Errors (train_transformer.py:33-34)

**Current (Broken):**
```python
from data.vocabulary_bpe_en import BPEVocabularyEN
from data.vocabulary_bpe_zh import BPEVocabularyZH
```

**Fixed:**
```python
from data.vocabulary_bpe import BPEVocabulary
```

### 2. Fix Vocabulary Initialization (train_transformer.py:293-297)

**Current (Broken):**
```python
vocab_zh = BPEVocabularyZH("Chinese")
vocab_en = BPEVocabularyEN("English")
```

**Fixed:**
```python
vocab_zh = BPEVocabulary("Chinese_BPE")
vocab_en = BPEVocabulary("English_BPE")
```

### 3. Fix Tokenization Method Call (train_transformer.py:205-206)

**Current (Broken):**
```python
zh_tokens = vocab_zh.segement_text(zh_text, max_length=max_length, is_pretokenized=False, output_ids=True)
en_tokens = vocab_en.segement_text(en_text, max_length=max_length, is_pretokenized=False, output_ids=True)
```

**Fixed:**
```python
# Use encode_batch for batch encoding
zh_tokens = vocab_zh.encode_batch(zh_text, max_length=max_length)
en_tokens = vocab_en.encode_batch(en_text, max_length=max_length)
```

### 4. Fix dataset_prepare Function (train_transformer.py:202-212)

**Current (Broken):**
```python
def dataset_prepare(data, vocab_zh, vocab_en, max_length=None, add_eos=False):
    "assume source is Chinese, target is English"
    zh_text, en_text = map(list, zip(*data))
    zh_tokens = vocab_zh.segement_text(zh_text, ...)
    en_tokens = vocab_en.segement_text(en_text, ...)
    return list(zip(zh_tokens, en_tokens))
```

**Fixed - Option A: Use Raw Text**
```python
def dataset_prepare(data, vocab_zh, vocab_en, max_length=None, add_eos=False):
    """
    Assume source is Chinese (zh), target is English (en)
    data: list of (zh_text, en_text) tuples
    """
    zh_texts, en_texts = zip(*data)

    # Encode using BPE
    zh_ids = vocab_zh.encode_batch(list(zh_texts), max_length=max_length)
    en_ids = vocab_en.encode_batch(list(en_texts), max_length=max_length)

    # Add BOS/EOS if needed
    if add_eos:
        zh_ids = [ids + [vocab_zh.eos_idx] for ids in zh_ids]
        en_ids = [[vocab_en.bos_idx] + ids + [vocab_en.eos_idx] for ids in en_ids]

    return list(zip(zh_ids, en_ids))
```

**Fixed - Option B: Load Raw Text Directly**
Modify the data loading to get raw text instead of pre-tokenized:

```python
# In main():
train_data_raw = []
with open(train_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        train_data_raw.append((item['zh'], item['en']))  # (zh_text, en_text)

# Then use dataset_prepare with raw text
train_data = dataset_prepare(train_data_raw, vocab_zh, vocab_en, ...)
```

### 5. Fix load_and_preprocess Calls (train_transformer.py:271-273)

**Current (Broken):**
```python
train_data = preprocessor.load_and_preprocess(train_path, return_text=True)
valid_data = preprocessor.load_and_preprocess(config.VALID_PATH, return_text=True)
test_data = preprocessor.load_and_preprocess(config.TEST_PATH, return_text=True)
```

**Fixed:**
```python
# Option 1: Load raw data for BPE tokenization
def load_raw_data(filepath):
    """Load raw text data for BPE tokenization"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append((item['zh'], item['en']))  # (zh_text, en_text)
    return data

train_data = load_raw_data(train_path)
valid_data = load_raw_data(config.VALID_PATH)
test_data = load_raw_data(config.TEST_PATH)
```

### 6. Complete Fixed dataset_prepare Implementation

Here's a complete, working implementation:

```python
def dataset_prepare(data, vocab_zh, vocab_en, max_length=None, add_eos=False):
    """
    Encode text data using BPE tokenizers

    Args:
        data: List of (zh_text, en_text) tuples (raw strings)
        vocab_zh: Chinese BPE vocabulary
        vocab_en: English BPE vocabulary
        max_length: Maximum sequence length
        add_eos: Whether to add EOS tokens

    Returns:
        List of (zh_ids, en_ids) tuples
    """
    # Unpack texts
    zh_texts, en_texts = zip(*data)

    # Encode using BPE tokenizers
    zh_ids_list = vocab_zh.encode_batch(list(zh_texts), max_length=max_length)
    en_ids_list = vocab_en.encode_batch(list(en_texts), max_length=max_length)

    # Add special tokens if needed
    if add_eos:
        # For Chinese: add EOS at the end
        zh_ids_list = [ids + [vocab_zh.eos_idx] for ids in zh_ids_list]

        # For English: add BOS at start and EOS at end
        en_ids_list = [[vocab_en.bos_idx] + ids + [vocab_en.eos_idx]
                       for ids in en_ids_list]

    return list(zip(zh_ids_list, en_ids_list))
```

## Additional Recommendations

### 1. Add Special Token Properties to BPEVocabulary

Make sure `BPEVocabulary` has these properties:

```python
class BPEVocabulary:
    ...

    @property
    def pad_idx(self):
        return self.tokenizer.token_to_id(self.pad_token)

    @property
    def unk_idx(self):
        return self.tokenizer.token_to_id(self.unk_token)

    @property
    def bos_idx(self):
        return self.tokenizer.token_to_id(self.bos_token)

    @property
    def eos_idx(self):
        return self.tokenizer.token_to_id(self.eos_token)
```

### 2. Use TokenizerManager (Recommended)

Instead of manually handling BPE vocabularies, use the provided `TokenizerManager`:

```python
from data.tokenizer_integration import TokenizerManager

# Initialize
manager = TokenizerManager()

# Train tokenizers
manager.train_tokenizers(
    corpus_path=train_path,
    en_vocab_size=10000,
    zh_vocab_size=10000
)

# Use in dataset_prepare
def dataset_prepare_with_manager(data, manager, max_length=None, add_eos=False):
    zh_texts, en_texts = zip(*data)

    # Encode
    zh_ids, en_ids = manager.encode_batch(
        zh_texts=list(zh_texts),
        en_texts=list(en_texts),
        max_length=max_length
    )

    # Add special tokens if needed
    if add_eos:
        special_ids = manager.get_special_token_ids()
        zh_ids = [ids + [special_ids['zh']['eos_idx']] for ids in zh_ids]
        en_ids = [[special_ids['en']['bos_idx']] + ids + [special_ids['en']['eos_idx']]
                  for ids in en_ids]

    return list(zip(zh_ids, en_ids))
```

### 3. Verify Training Loop

Ensure the training loop properly uses the encoded data:

```python
# In train_epoch:
for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(dataloader):
    src = src.to(config.DEVICE)
    tgt = tgt.to(config.DEVICE)
    src_len = src_len.to(config.DEVICE)

    # tgt is already encoded IDs, so we can use directly
    tgt_input = tgt[:, :-1]  # Remove last token
    tgt_output = tgt[:, 1:]  # Remove first token

    outputs = model(src, tgt_input)
    ...
```

## Testing the Fixes

After applying these fixes:

1. **Check that imports work**: No ImportError or AttributeError
2. **Verify tokenization**: Sample data should encode/decode correctly
3. **Run training**: Loss should decrease over epochs
4. **Check output**: Translations should be coherent (not garbled)
5. **Verify BLEU score**: Should be significantly higher (e.g., >20)

## Expected Improvement

With these fixes, you should see:
- **Proper tokenization**: Text encoded correctly as IDs
- **Successful training**: Loss decreases from ~10 to <3
- **Coherent translations**: English-like output sentences
- **Better BLEU score**: Expected ~20-30 (vs current ~0.06)

The current BLEU of 0.0654 is essentially random performance, indicating the model never learned anything due to the broken data pipeline.
