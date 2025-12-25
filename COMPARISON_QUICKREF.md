# Quick Reference: Model Comparison

## Files Created

1. **`compare_models.py`** - Main comparison script
2. **`COMPARISON_GUIDE.md`** - Detailed usage documentation
3. **`run_comparison_example.py`** - Example workflow script

## Quick Start

```bash
# Basic comparison
python compare_models.py \
    --rnn-checkpoint ./checkpoints/rnn/best_model.pt \
    --transformer-checkpoint ./checkpoints/transformer/best_model.pt

# With custom output directory
python compare_models.py \
    --rnn-checkpoint ./checkpoints/rnn/best_model.pt \
    --transformer-checkpoint ./checkpoints/transformer/best_model.pt \
    --output-dir ./my_results
```

## What Gets Analyzed

| Dimension | Metrics |
|-----------|---------|
| **Architecture** | Parameter count, complexity, parallelism |
| **Training Efficiency** | Epochs, validation BLEU, training time |
| **Performance** | BLEU-4, Precision-1/2/3/4, inference time |
| **Scalability** | Performance by sentence length (short/medium/long) |
| **Speed** | Greedy/beam search latency, throughput |
| **Memory** | Usage at different batch sizes |
| **Qualitative** | Sample translations |

## Output Files

- `comparison_report.txt` - Human-readable comprehensive report
- `detailed_results.json` - Machine-readable detailed data

## Key Features

### ✅ Automatic Analysis
- Sentence length buckets (≤10, 11-30, 31-50 tokens)
- Memory profiling at multiple batch sizes
- Speed profiling (min, max, avg, std)
- Sample translation generation

### ✅ Comprehensive Report Includes
1. Architecture comparison
2. Model complexity analysis
3. Training efficiency metrics
4. Translation performance (greedy + beam search)
5. Scalability across sentence lengths
6. Practical trade-offs summary
7. Recommendations
8. Sample translations

### ✅ Practical Trade-offs Analyzed
- Model size vs quality
- Speed vs accuracy
- Memory vs batch size
- Simplicity vs performance

## Example Workflow

```bash
# 1. Train models
python train_rnn.py --attention dot --epochs 20
python train_transformer.py --epochs 20

# 2. Run comparison
python compare_models.py \
    --rnn-checkpoint ./checkpoints/rnn/best_model_dot_LSTM.pt \
    --transformer-checkpoint ./checkpoints/transformer/best_model.pt

# 3. View results
cat ./comparison_results/comparison_report.txt
```

## Or Use the Example Script

```bash
# Complete workflow (training + comparison)
python run_comparison_example.py

# Skip training, use existing models
python run_comparison_example.py --skip-training

# Specify custom checkpoints
python run_comparison_example.py \
    --rnn-checkpoint ./checkpoints/rnn/custom.pt \
    --transformer-checkpoint ./checkpoints/transformer/custom.pt
```

## Understanding the Report

### BLEU Scores Interpretation
- 30+ = Excellent
- 20-30 = Good
- 10-20 = Fair
- <10 = Poor

### Speed Analysis
- Lower inference time = faster
- Compare tokens/second for throughput
- Consider both greedy and beam search

### Length Analysis
- RNN typically degrades on long sentences
- Transformer maintains more consistent performance
- Look for degradation patterns

### Recommendations Section
The report automatically suggests which model to use based on:
- Available computational resources
- Required translation quality
- Sentence length characteristics
- Deployment constraints

## Common Use Cases

### 1. Model Selection
```bash
# Compare trained models to choose best one
python compare_models.py --rnn-checkpoint <path1> --transformer-checkpoint <path2>
```

### 2. Architecture Comparison
```bash
# Compare different RNN attention mechanisms
python compare_models.py \
    --rnn-checkpoint ./checkpoints/rnn/dot_att.pt \
    --transformer-checkpoint ./checkpoints/rnn/additive_att.pt
```

### 3. Hyperparameter Analysis
```bash
# Compare different model scales
python compare_models.py \
    --rnn-checkpoint ./checkpoints/rnn/small.pt \
    --transformer-checkpoint ./checkpoints/rnn/large.pt
```

## Integration with Existing Scripts

The comparison script works with existing checkpoints:
- Uses same vocabulary format
- Compatible with all training configurations
- Works with both small and large datasets

## Customization

Modify `compare_models.py` to add:
- Custom metrics
- Domain-specific analysis
- Attention visualization
- Additional length buckets

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Checkpoint not found | Verify path exists, check filename |
| Out of memory | Reduce batch size, use CPU |
| CUDA error | Script auto-falls back to CPU |
| Missing training info | Optional, script still runs |

## Dependencies

```bash
pip install numpy pandas matplotlib
```

## Support

For detailed information, see `COMPARISON_GUIDE.md`

---

**Created by**: Claude Code
**Purpose**: Comprehensive comparison of RNN vs Transformer NMT models
**Compatible with**: All checkpoints from train_rnn.py and train_transformer.py
