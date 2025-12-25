# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Chinese-English Machine Translation** project implementing both RNN-based and Transformer-based neural machine translation models. The project includes training scripts, evaluation tools, and comprehensive model architectures for comparing different NMT approaches.

## Common Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify dataset exists (should contain train_10k.jsonl, train_100k.jsonl, valid.jsonl, test.jsonl)
ls dataset/
```

### Train RNN Model
```bash
# Basic training (small dataset)
python train_rnn.py

# Different attention mechanisms
python train_rnn.py --attention dot
python train_rnn.py --attention multiplicative
python train_rnn.py --attention additive

# Different RNN cells
python train_rnn.py --cell LSTM
python train_rnn.py --cell GRU

# Teacher forcing strategies
python train_rnn.py --teacher-forcing 1.0  # Full teacher forcing
python train_rnn.py --teacher-forcing 0.0  # No teacher forcing
python train_rnn.py --teacher-forcing 0.5  # Scheduled sampling

# Train on large dataset
python train_rnn.py --use-large --epochs 20

# Custom hyperparameters
python train_rnn.py --batch-size 128 --lr 0.0005 --epochs 30
```

### Train Transformer Model
```bash
# Basic training
python train_transformer.py

# Architectural variations
python train_transformer.py --position-embedding sinusoidal
python train_transformer.py --position-embedding learned
python train_transformer.py --position-embedding relative

python train_transformer.py --norm-type LayerNorm
python train_transformer.py --norm-type RMSNorm

# Hyperparameter experiments
python train_transformer.py --batch-size 32
python train_transformer.py --batch-size 128
python train_transformer.py --lr 0.0001
python train_transformer.py --lr 0.0005
python train_transformer.py --d-model 256 --num-layers 2  # Small model
python train_transformer.py --d-model 512 --num-layers 6  # Large model

# Label smoothing with large dataset
python train_transformer.py --use-large --label-smoothing --epochs 20
```

### Evaluate Models
```bash
# Evaluate single model
python evaluate.py --rnn-checkpoint ./checkpoints/rnn/best_model_dot_LSTM.pt
python evaluate.py --transformer-checkpoint ./checkpoints/transformer/sinusoidal_LayerNorm_d512_l4_bs64_lr0.001/best_model.pt

# Compare RNN vs Transformer
python evaluate.py \
    --rnn-checkpoint ./checkpoints/rnn/best_model_dot_LSTM.pt \
    --transformer-checkpoint ./checkpoints/transformer/sinusoidal_LayerNorm_d512_l4_bs64_lr0.001/best_model.pt \
    --output comparison_results.json
```

### Finetune Pretrained Models
```bash
# Finetune T5 on small dataset
python finetune.py --model-name t5-base --batch-size 32 --lr 5e-5 --epochs 10

# Finetune mT5 (multilingual) on large dataset
python finetune.py --model-name google/mt5-base --use-large --batch-size 16 --epochs 20

# Finetune with frozen encoder
python finetune.py --model-name t5-base --freeze-encoder --lr 1e-4

# Resume finetuning from checkpoint
python finetune.py --model-name t5-base --checkpoint ./checkpoints/pretrained/t5-base_best.pt
```

### Data Loading

The dataset uses JSONL format with each line containing: `{"en": "...", "zh": "...", "index": ...}`

- **train_10k.jsonl** - Small training set (10k samples)
- **train_100k.jsonl** - Large training set (100k samples)
- **valid.jsonl** - Validation set (500 samples)
- **test.jsonl** - Test set (200 samples)

## Code Architecture

### High-Level Structure

```
hw3_code/
├── config.py                    # Configuration with all hyperparameters
├── data/                        # Data processing utilities
│   ├── preprocessor.py         # Tokenization and preprocessing
│   ├── vocabulary.py           # Vocabulary management
│   └── dataloader.py           # PyTorch DataLoader
├── models/                      # Model architectures
│   ├── rnn/                    # RNN-based Seq2Seq models
│   │   ├── attention.py        # Attention mechanisms (dot, multiplicative, additive)
│   │   ├── encoder.py          # RNN encoder
│   │   ├── decoder.py          # RNN decoder
│   │   ├── seq2seq.py          # Complete Seq2Seq model
│   │   └── __init__.py
│   └── transformer/            # Transformer models
│       ├── attention.py        # Multi-head attention
│       ├── positional_encoding.py  # Positional embeddings
│       ├── normalization.py    # LayerNorm and RMSNorm
│       ├── encoder.py          # Transformer encoder
│       ├── decoder.py          # Transformer decoder
│       ├── transformer.py      # Complete Transformer model
│       └── __init__.py
├── utils/                       # Training and evaluation utilities
│   ├── metrics.py              # BLEU-4 and precision-n metrics
│   ├── beam_search.py          # Greedy and beam search decoding
│   └── training_utils.py       # Checkpointing, logging, warmup scheduler
├── train_rnn.py                # RNN training script
├── train_transformer.py        # Transformer training script
├── evaluate.py                 # Model evaluation script
└── dataset/                     # Training/validation/test data
```

### Key Model Components

**RNN Models (Seq2Seq):**
- `models/rnn/seq2seq.py:12` - Main Seq2Seq class combining encoder-decoder
- `models/rnn/encoder.py` - Bidirectional RNN encoder (LSTM/GRU)
- `models/rnn/decoder.py` - RNN decoder with attention mechanism
- `models/rnn/attention.py:1` - Three attention types: dot, multiplicative, additive

**Transformer Models:**
- `models/transformer/transformer.py:12` - Main Transformer class
- `models/transformer/encoder.py` - Multi-layer Transformer encoder
- `models/transformer/decoder.py` - Multi-layer Transformer decoder
- `models/transformer/attention.py` - Multi-head self-attention
- `models/transformer/positional_encoding.py:1` - Support for sinusoidal, learned, and relative position embeddings
- `models/transformer/normalization.py:1` - LayerNorm and RMSNorm implementations

**Data Pipeline:**
- `data/preprocessor.py:1` - Text tokenization using jieba for Chinese
- `data/vocabulary.py:1` - Vocabulary building and management
- `data/dataloader.py:1` - PyTorch DataLoader with batching

**Training Infrastructure:**
- `utils/training_utils.py:1` - Checkpoint saving/loading, parameter counting, training meters
- `utils/beam_search.py:1` - Greedy and beam search decoding algorithms
- `utils/metrics.py:1` - BLEU-4 and precision-n calculations

### Configuration

All hyperparameters are centralized in `config.py:5`:
- Dataset paths (train_10k.jsonl, train_100k.jsonl, valid.jsonl, test.jsonl)
- RNN parameters (embed_dim=256, hidden_dim=512, num_layers=2, dropout=0.3)
- Transformer parameters (d_model=512, nhead=8, num_layers=4, dim_feedforward=2048)
- Training settings (batch_size=64, lr=0.001, epochs=30, grad_clip=5.0)
- Device configuration (defaults to CUDA)

### Checkpoint Structure

Training outputs are saved to:
- `./checkpoints/rnn/` - RNN model checkpoints
- `./checkpoints/transformer/` - Transformer model checkpoints

Each checkpoint directory contains:
- `best_model.pt` - Best model based on validation BLEU
- `checkpoint_epoch_*.pt` - Periodic checkpoints
- `vocab_zh.pkl` - Chinese vocabulary
- `vocab_en.pkl` - English vocabulary
- `results.txt` - Evaluation results and sample translations

### Core Training Loops

**RNN Training (`train_rnn.py:30`):**
- Teacher forcing configurable via `--teacher-forcing` flag
- Supports scheduled sampling strategies
- Uses attention mechanisms for alignment
- Gradient clipping at 5.0

**Transformer Training (`train_transformer.py:67`):**
- Label smoothing loss implementation (`train_transformer.py:30`)
- Warmup scheduler for learning rate
- Position embedding and normalization variations
- Parallel training (no teacher forcing needed)

**Evaluation (`evaluate.py:27`):**
- Computes BLEU-4 and precision-n (n=1,2,3,4) metrics
- Compares greedy vs beam search decoding
- Measures inference time per sentence
- Generates sample translations for qualitative analysis

## Key Implementation Details

### Special Tokens
- `<pad>` - Padding token
- `<unk>` - Unknown token
- `<bos>` - Beginning of sentence
- `<eos>` - End of sentence

### Attention Mechanisms
1. **Dot-product** - Simple attention via dot product
2. **Multiplicative** - Scaled dot-product attention
3. **Additive (Bahdanau)** - Additive attention with non-linearities

### Decoding Strategies
- **Greedy decoding** - Select argmax at each step (fast)
- **Beam search** - Search with beam size (better quality, slower)

### Vocabulary
- Built from training data with frequency threshold (MIN_FREQ=2)
- Maximum size: 50,000 tokens
- Separate vocabularies for Chinese and English

## Experimental Variations

**RNN Experiments:**
- Attention type comparison (dot, multiplicative, additive)
- Cell type comparison (LSTM vs GRU)
- Teacher forcing ratios (0.0, 0.5, 1.0)
- Decoding strategies (greedy vs beam search)

**Transformer Experiments:**
- Position embeddings (sinusoidal, learned, relative)
- Normalization (LayerNorm, RMSNorm)
- Model scales (small: 2 layers, medium: 4 layers, large: 6 layers)
- Hyperparameter sensitivity (batch size, learning rate)

## Development Notes

### Modifying Models
- To add new attention mechanisms: edit `models/rnn/attention.py:1`
- To add positional embeddings: edit `models/transformer/positional_encoding.py:1`
- To modify model hyperparameters: edit `config.py:5`

### Finetuning Pretrained Models
The `finetune.py` script enables finetuning of pretrained Transformer models (T5, mT5, etc.) for NMT:
- Loads pretrained tokenizer and model from Hugging Face
- Supports encoder freezing for parameter-efficient finetuning
- Uses AdamW optimizer with warmup scheduler
- Evaluates on validation/test sets with BLEU and precision metrics
- Saves best model based on validation BLEU

**Key differences from `train_transformer.py`**:
- Uses pretrained tokenizer instead of custom vocabulary
- Leverages pretrained weights for faster convergence
- Lower learning rates (5e-5) recommended for finetuning
- Supports model checkpointing and resuming

### Key Implementation Patterns
- RNN models use teacher forcing during training (configurable ratio)
- Transformer models use label smoothing loss for regularization
- Both models support greedy and beam search decoding
- Checkpoints include vocabularies for reproducible inference
- Relative positional encoding uses einsum for efficient computation

## Metrics and Evaluation

- **BLEU-4**: Standard MT metric measuring n-gram precision
- **Precision-n**: Unigram to 4-gram precision scores
- **Inference time**: Average decoding time per sentence
- **Qualitative analysis**: Sample translations saved to `results.txt`

## Dependencies

- `torch>=2.0.0` - Deep learning framework
- `jieba>=0.42.1` - Chinese text segmentation
- `tqdm>=4.65.0` - Progress bars
