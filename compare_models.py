"""
Comprehensive comparison script for RNN-based and Transformer-based NMT models
Evaluates multiple dimensions: architecture, efficiency, performance, scalability, and trade-offs
"""

import torch
import os
import sys
import argparse
import time
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.preprocessor import Preprocessor
from data.vocabulary import Vocabulary
from data.dataloader import get_dataloader
from models.rnn.seq2seq import Seq2Seq
from models.transformer.transformer import Transformer
from utils.metrics import calculate_all_metrics
from utils.training_utils import load_checkpoint, count_parameters
from utils.beam_search import greedy_decode, beam_search_decode


class ModelComparator:
    """Comprehensive model comparison framework"""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(
            config.DEVICE if torch.cuda.is_available() else 'cpu'
        )
        self.config.DEVICE = self.device
        print(f"Using device: {self.device}\n")

    def load_rnn_model(self, checkpoint_path: str):
        """Load RNN model from checkpoint"""
        checkpoint = load_checkpoint(checkpoint_path, None, device=self.device)
        vocab_zh = checkpoint.get('vocab_src')
        vocab_en = checkpoint.get('vocab_tgt')

        model = Seq2Seq(
            src_vocab_size=len(vocab_zh),
            tgt_vocab_size=len(vocab_en),
            embed_dim=self.config.RNN_EMBED_DIM,
            hidden_dim=self.config.RNN_HIDDEN_DIM,
            num_layers=self.config.RNN_NUM_LAYERS,
            dropout=self.config.RNN_DROPOUT,
            cell_type=self.config.RNN_CELL_TYPE,
            attention_type=self.config.ATTENTION_TYPE,
            pad_idx=vocab_en.pad_idx
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        # Extract training info if available
        training_info = {
            'epochs_trained': checkpoint.get('epoch', 0),
            'best_val_bleu': checkpoint.get('best_val_bleu', 0.0),
            'training_time': checkpoint.get('total_training_time', 0.0),
        }

        return model, vocab_zh, vocab_en, training_info

    def load_transformer_model(self, checkpoint_path: str):
        """Load Transformer model from checkpoint"""
        checkpoint = load_checkpoint(checkpoint_path, None, device=self.device)
        vocab_zh = checkpoint.get('vocab_src')
        vocab_en = checkpoint.get('vocab_tgt')

        model = Transformer(
            src_vocab_size=len(vocab_zh),
            tgt_vocab_size=len(vocab_en),
            d_model=self.config.TRANS_D_MODEL,
            num_heads=self.config.TRANS_NHEAD,
            num_encoder_layers=self.config.TRANS_NUM_ENCODER_LAYERS,
            num_decoder_layers=self.config.TRANS_NUM_DECODER_LAYERS,
            dim_feedforward=self.config.TRANS_DIM_FEEDFORWARD,
            dropout=self.config.TRANS_DROPOUT,
            position_embedding=self.config.POSITION_EMBEDDING,
            norm_type=self.config.NORM_TYPE,
            pad_idx=vocab_en.pad_idx
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        # Extract training info if available
        training_info = {
            'epochs_trained': checkpoint.get('epoch', 0),
            'best_val_bleu': checkpoint.get('best_val_bleu', 0.0),
            'training_time': checkpoint.get('total_training_time', 0.0),
        }

        return model, vocab_zh, vocab_en, training_info

    def analyze_by_length(self, model, dataloader, vocab_src, vocab_tgt,
                         model_type: str) -> Dict[str, Any]:
        """Analyze model performance across different sentence lengths"""
        model.eval()

        length_buckets = {
            'short': {'refs': [], 'hyps': [], 'times': []},   # <= 10
            'medium': {'refs': [], 'hyps': [], 'times': []},  # 11-30
            'long': {'refs': [], 'hyps': [], 'times': []},    # 31-50
        }

        with torch.no_grad():
            for src, tgt, src_len, tgt_len in tqdm(dataloader, desc=f"Analyzing by length ({model_type})"):
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                src_len = src_len.to(self.device)

                for i in range(src.size(0)):
                    src_seq = src[i:i+1]
                    src_len_seq = src_len[i:i+1]
                    ref_len = src_len_seq.item()

                    # Decode
                    start_time = time.time()
                    pred_tokens = beam_search_decode(
                        model, src_seq, src_len_seq, vocab_tgt,
                        self.config.BEAM_SIZE, self.config.MAX_DECODE_LENGTH,
                        self.device
                    )
                    decode_time = time.time() - start_time

                    # Get reference
                    ref_tokens = tgt[i].cpu().tolist()
                    ref_tokens = [t for t in ref_tokens if t not in [
                        vocab_tgt.pad_idx, vocab_tgt.bos_idx, vocab_tgt.eos_idx
                    ]]

                    # Convert to words
                    ref_words = vocab_tgt.decode(ref_tokens, skip_special=False)
                    pred_words = vocab_tgt.decode(pred_tokens, skip_special=False)

                    # Bucket by length
                    if ref_len <= 10:
                        bucket = 'short'
                    elif ref_len <= 30:
                        bucket = 'medium'
                    else:
                        bucket = 'long'

                    length_buckets[bucket]['refs'].append(ref_words)
                    length_buckets[bucket]['hyps'].append(pred_words)
                    length_buckets[bucket]['times'].append(decode_time)

        # Calculate metrics for each bucket
        results = {}
        for bucket_name, data in length_buckets.items():
            if data['refs']:
                metrics = calculate_all_metrics(data['refs'], data['hyps'])
                avg_time = np.mean(data['times'])
                results[bucket_name] = {
                    'bleu4': metrics['bleu4'],
                    'precision_1': metrics['precision_1'],
                    'precision_2': metrics['precision_2'],
                    'precision_3': metrics['precision_3'],
                    'precision_4': metrics['precision_4'],
                    'avg_time': avg_time,
                    'sample_count': len(data['refs'])
                }

        return results

    def analyze_memory_efficiency(self, model, batch_sizes: List[int]) -> Dict[str, float]:
        """Analyze memory usage at different batch sizes"""
        memory_usage = {}

        for batch_size in batch_sizes:
            # Create dummy input
            src = torch.randint(0, 100, (batch_size, 50), device=self.device)
            tgt = torch.randint(0, 100, (batch_size, 50), device=self.device)

            # Measure memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                start_mem = torch.cuda.memory_allocated()

                with torch.no_grad():
                    if isinstance(model, Seq2Seq):
                        # RNN forward pass
                        encoder_outputs, _ = model.encoder(src)
                        _ = model.decoder(tgt[:, :-1], encoder_outputs)
                    else:
                        # Transformer forward pass
                        _ = model(src, tgt[:, :-1])

                torch.cuda.synchronize()
                end_mem = torch.cuda.memory_allocated()
                mem_used = (end_mem - start_mem) / 1024 / 1024  # MB
            else:
                mem_used = 0.0

            memory_usage[f'batch_{batch_size}'] = mem_used

        return memory_usage

    def measure_inference_speed(self, model, dataloader, vocab_src, vocab_tgt,
                               model_type: str) -> Dict[str, float]:
        """Comprehensive inference speed analysis"""
        model.eval()

        greedy_times = []
        beam_times = []
        throughput_metrics = {'total_tokens': 0, 'total_time': 0}

        with torch.no_grad():
            for src, tgt, src_len, tgt_len in tqdm(dataloader, desc=f"Speed analysis ({model_type})"):
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                src_len = src_len.to(self.device)

                batch_start = time.time()

                for i in range(src.size(0)):
                    src_seq = src[i:i+1]
                    src_len_seq = src_len[i:i+1]
                    tgt_len_seq = tgt_len[i:i+1]

                    # Greedy decoding
                    start_time = time.time()
                    pred_tokens_greedy = greedy_decode(
                        model, src_seq, src_len_seq, vocab_tgt,
                        self.config.MAX_DECODE_LENGTH, self.device
                    )
                    greedy_times.append(time.time() - start_time)

                    # Beam search decoding
                    start_time = time.time()
                    pred_tokens_beam = beam_search_decode(
                        model, src_seq, src_len_seq, vocab_tgt,
                        self.config.BEAM_SIZE, self.config.MAX_DECODE_LENGTH,
                        self.device
                    )
                    beam_times.append(time.time() - start_time)

                    # Track throughput
                    tokens_decoded = len(pred_tokens_beam)
                    throughput_metrics['total_tokens'] += tokens_decoded

                batch_time = time.time() - batch_start
                throughput_metrics['total_time'] += batch_time

        return {
            'greedy': {
                'avg_time': np.mean(greedy_times),
                'std_time': np.std(greedy_times),
                'min_time': np.min(greedy_times),
                'max_time': np.max(greedy_times)
            },
            'beam': {
                'avg_time': np.mean(beam_times),
                'std_time': np.std(beam_times),
                'min_time': np.min(beam_times),
                'max_time': np.max(beam_times)
            },
            'throughput': {
                'tokens_per_second': throughput_metrics['total_tokens'] / throughput_metrics['total_time'],
                'sentences_per_second': len(greedy_times) / throughput_metrics['total_time']
            }
        }

    def qualitative_analysis(self, model, dataloader, vocab_src, vocab_tgt,
                           model_type: str, num_samples: int = 10) -> List[Dict]:
        """Generate sample translations for qualitative analysis"""
        model.eval()

        samples = []
        count = 0

        with torch.no_grad():
            for src, tgt, src_len, tgt_len in dataloader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                src_len = src_len.to(self.device)

                for i in range(src.size(0)):
                    if count >= num_samples:
                        break

                    src_seq = src[i:i+1]
                    src_len_seq = src_len[i:i+1]

                    # Decode
                    pred_tokens = beam_search_decode(
                        model, src_seq, src_len_seq, vocab_tgt,
                        self.config.BEAM_SIZE, self.config.MAX_DECODE_LENGTH,
                        self.device
                    )

                    # Get source and reference
                    src_tokens = src[i].cpu().tolist()
                    src_tokens = [t for t in src_tokens if t not in [
                        vocab_src.pad_idx, vocab_src.bos_idx, vocab_src.eos_idx
                    ]]
                    ref_tokens = tgt[i].cpu().tolist()
                    ref_tokens = [t for t in ref_tokens if t not in [
                        vocab_tgt.pad_idx, vocab_tgt.bos_idx, vocab_tgt.eos_idx
                    ]]

                    src_text = vocab_src.decode(src_tokens, skip_special=False)
                    ref_text = vocab_tgt.decode(ref_tokens, skip_special=False)
                    hyp_text = vocab_tgt.decode(pred_tokens, skip_special=False)

                    samples.append({
                        'source': src_text,
                        'reference': ref_text,
                        'hypothesis': hyp_text,
                        'source_length': len(src_tokens),
                        'reference_length': len(ref_tokens),
                        'hypothesis_length': len(pred_tokens)
                    })

                    count += 1

                if count >= num_samples:
                    break

        return samples

    def generate_comparison_report(self, rnn_results: Dict, trans_results: Dict,
                                 output_file: str):
        """Generate comprehensive comparison report"""

        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE MODEL COMPARISON REPORT")
        report.append("=" * 80)
        report.append("")

        # 1. ARCHITECTURE COMPARISON
        report.append("1. MODEL ARCHITECTURE")
        report.append("-" * 80)
        report.append("")
        report.append("RNN-based Seq2Seq Model:")
        report.append(f"  • Architecture: Encoder-Decoder with Attention")
        report.append(f"  • Encoder: {self.config.RNN_NUM_LAYERS}-layer bidirectional {self.config.RNN_CELL_TYPE}")
        report.append(f"  • Decoder: {self.config.RNN_NUM_LAYERS}-layer {self.config.RNN_CELL_TYPE} with {self.config.ATTENTION_TYPE} attention")
        report.append(f"  • Hidden size: {self.config.RNN_HIDDEN_DIM}")
        report.append(f"  • Embedding dim: {self.config.RNN_EMBED_DIM}")
        report.append(f"  • Computation: Sequential (step-by-step)")
        report.append(f"  • Parallelism: Limited (cannot parallelize across time)")
        report.append("")

        report.append("Transformer-based Model:")
        report.append(f"  • Architecture: Multi-layer Self-Attention")
        report.append(f"  • Encoder: {self.config.TRANS_NUM_ENCODER_LAYERS} layers, d_model={self.config.TRANS_D_MODEL}")
        report.append(f"  • Decoder: {self.config.TRANS_NUM_DECODER_LAYERS} layers with masked self-attention")
        report.append(f"  • Attention heads: {self.config.TRANS_NHEAD}")
        report.append(f"  • Position embedding: {self.config.POSITION_EMBEDDING}")
        report.append(f"  • Normalization: {self.config.NORM_TYPE}")
        report.append(f"  • Computation: Parallel (full sequence at once)")
        report.append(f"  • Parallelism: High (can parallelize across sequence)")
        report.append("")

        # 2. MODEL COMPLEXITY
        report.append("2. MODEL COMPLEXITY")
        report.append("-" * 80)
        report.append("")
        report.append(f"RNN Parameters: {rnn_results['params']:,}")
        report.append(f"Transformer Parameters: {trans_results['params']:,}")
        report.append(f"Parameter Ratio (Trans/RNN): {trans_results['params'] / rnn_results['params']:.2f}x")
        report.append("")

        # 3. TRAINING EFFICIENCY
        report.append("3. TRAINING EFFICIENCY")
        report.append("-" * 80)
        report.append("")

        if 'training_info' in rnn_results:
            report.append("RNN Training:")
            report.append(f"  • Epochs trained: {rnn_results['training_info']['epochs_trained']}")
            report.append(f"  • Best validation BLEU: {rnn_results['training_info']['best_val_bleu']:.4f}")
            if rnn_results['training_info']['training_time'] > 0:
                report.append(f"  • Total training time: {rnn_results['training_info']['training_time']:.2f}s")
            report.append("")

        if 'training_info' in trans_results:
            report.append("Transformer Training:")
            report.append(f"  • Epochs trained: {trans_results['training_info']['epochs_trained']}")
            report.append(f"  • Best validation BLEU: {trans_results['training_info']['best_val_bleu']:.4f}")
            if trans_results['training_info']['training_time'] > 0:
                report.append(f"  • Total training time: {trans_results['training_info']['training_time']:.2f}s")
            report.append("")

        report.append("Key Differences:")
        report.append("  • RNN: Sequential processing, slower training")
        report.append("  • Transformer: Parallel processing, faster training")
        report.append("  • RNN: Lower computational complexity O(n)")
        report.append("  • Transformer: Higher computational complexity O(n²) but parallelizable")
        report.append("")

        # 4. TRANSLATION PERFORMANCE
        report.append("4. TRANSLATION PERFORMANCE")
        report.append("-" * 80)
        report.append("")

        report.append("RNN Results (Beam Search):")
        for key, value in rnn_results['beam'].items():
            if key != 'avg_time':
                report.append(f"  • {key}: {value:.4f}")
        report.append(f"  • Avg inference time: {rnn_results['beam']['avg_time']:.4f}s")
        report.append("")

        report.append("Transformer Results (Beam Search):")
        for key, value in trans_results['beam'].items():
            if key != 'avg_time':
                report.append(f"  • {key}: {value:.4f}")
        report.append(f"  • Avg inference time: {trans_results['beam']['avg_time']:.4f}s")
        report.append("")

        # Performance comparison
        if 'bleu4' in rnn_results['beam'] and 'bleu4' in trans_results['beam']:
            improvement = (trans_results['beam']['bleu4'] - rnn_results['beam']['bleu4']) / rnn_results['beam']['bleu4'] * 100
            report.append(f"BLEU-4 Improvement: {improvement:+.2f}%")
            report.append("")

        # 5. SCALABILITY & GENERALIZATION
        report.append("5. SCALABILITY & GENERALIZATION")
        report.append("-" * 80)
        report.append("")

        if 'length_analysis' in rnn_results and 'length_analysis' in trans_results:
            report.append("Performance by Sentence Length:")
            report.append("")
            report.append("Short sentences (≤10 tokens):")
            if 'short' in rnn_results['length_analysis']:
                report.append(f"  RNN BLEU-4: {rnn_results['length_analysis']['short']['bleu4']:.4f}")
            if 'short' in trans_results['length_analysis']:
                report.append(f"  Transformer BLEU-4: {trans_results['length_analysis']['short']['bleu4']:.4f}")
            report.append("")

            report.append("Medium sentences (11-30 tokens):")
            if 'medium' in rnn_results['length_analysis']:
                report.append(f"  RNN BLEU-4: {rnn_results['length_analysis']['medium']['bleu4']:.4f}")
            if 'medium' in trans_results['length_analysis']:
                report.append(f"  Transformer BLEU-4: {trans_results['length_analysis']['medium']['bleu4']:.4f}")
            report.append("")

            report.append("Long sentences (31-50 tokens):")
            if 'long' in rnn_results['length_analysis']:
                report.append(f"  RNN BLEU-4: {rnn_results['length_analysis']['long']['bleu4']:.4f}")
            if 'long' in trans_results['length_analysis']:
                report.append(f"  Transformer BLEU-4: {trans_results['length_analysis']['long']['bleu4']:.4f}")
            report.append("")

        report.append("Key Observations:")
        report.append("  • RNN: Performance degrades with length (vanishing gradients)")
        report.append("  • Transformer: More consistent across lengths (self-attention)")
        report.append("")

        # 6. PRACTICAL TRADE-OFFS
        report.append("6. PRACTICAL TRADE-OFFS")
        report.append("-" * 80)
        report.append("")

        report.append("Model Size:")
        report.append(f"  • RNN: {rnn_results['params']:,} parameters")
        report.append(f"  • Transformer: {trans_results['params']:,} parameters")
        report.append(f"  • Winner: {'RNN' if rnn_results['params'] < trans_results['params'] else 'Transformer'} (smaller)")
        report.append("")

        report.append("Inference Speed:")
        if 'speed_analysis' in rnn_results and 'speed_analysis' in trans_results:
            report.append(f"  • RNN beam search: {rnn_results['speed_analysis']['beam']['avg_time']:.4f}s")
            report.append(f"  • Transformer beam search: {trans_results['speed_analysis']['beam']['avg_time']:.4f}s")
            speedup = rnn_results['speed_analysis']['beam']['avg_time'] / trans_results['speed_analysis']['beam']['avg_time']
            report.append(f"  • Speedup: {speedup:.2f}x")
            report.append(f"  • Winner: {'RNN' if speedup < 1 else 'Transformer'} (faster)")
            report.append("")

        report.append("Translation Quality:")
        if 'bleu4' in rnn_results['beam'] and 'bleu4' in trans_results['beam']:
            report.append(f"  • RNN BLEU-4: {rnn_results['beam']['bleu4']:.4f}")
            report.append(f"  • Transformer BLEU-4: {trans_results['beam']['bleu4']:.4f}")
            winner = 'Transformer' if trans_results['beam']['bleu4'] > rnn_results['beam']['bleu4'] else 'RNN'
            report.append(f"  • Winner: {winner} (higher BLEU)")
            report.append("")

        report.append("Memory Efficiency:")
        if 'memory' in rnn_results and 'memory' in trans_results:
            report.append("  Batch size 32 memory usage (MB):")
            for key in rnn_results['memory']:
                if 'batch_32' in key and key in trans_results['memory']:
                    rnn_mem = rnn_results['memory'][key]
                    trans_mem = trans_results['memory'][key]
                    report.append(f"    • RNN: {rnn_mem:.2f} MB")
                    report.append(f"    • Transformer: {trans_mem:.2f} MB")
                    report.append(f"    • Winner: {'RNN' if rnn_mem < trans_mem else 'Transformer'} (less memory)")
            report.append("")

        report.append("Implementation Complexity:")
        report.append("  • RNN: Easier to understand and implement")
        report.append("  • Transformer: More complex but more flexible")
        report.append("  • Winner: RNN (simpler)")
        report.append("")

        # 7. RECOMMENDATIONS
        report.append("7. RECOMMENDATIONS")
        report.append("-" * 80)
        report.append("")
        report.append("Use RNN when:")
        report.append("  • Model size is a critical constraint")
        report.append("  • Working with short sequences")
        report.append("  • Simplicity and interpretability are important")
        report.append("  • Limited computational resources")
        report.append("")
        report.append("Use Transformer when:")
        report.append("  • Translation quality is the top priority")
        report.append("  • Working with long sequences")
        report.append("  • Sufficient computational resources available")
        report.append("  • Need parallel training/inference")
        report.append("  • State-of-the-art performance required")
        report.append("")

        # 8. SAMPLE TRANSLATIONS
        if 'samples' in rnn_results and 'samples' in trans_results:
            report.append("8. SAMPLE TRANSLATIONS")
            report.append("-" * 80)
            report.append("")

            for i in range(min(5, len(rnn_results['samples']), len(trans_results['samples']))):
                report.append(f"Sample {i+1}:")
                report.append(f"  Source: {rnn_results['samples'][i]['source']}")
                report.append(f"  Reference: {rnn_results['samples'][i]['reference']}")
                report.append(f"  RNN: {rnn_results['samples'][i]['hypothesis']}")
                report.append(f"  Transformer: {trans_results['samples'][i]['hypothesis']}")
                report.append("")

        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))

        return report

    def compare_models(self, rnn_checkpoint: str, trans_checkpoint: str,
                      output_dir: str = "./comparison_results"):
        """Run comprehensive comparison between two models"""

        os.makedirs(output_dir, exist_ok=True)

        # Load models
        print("Loading models...")
        rnn_model, vocab_zh, vocab_en, rnn_training_info = self.load_rnn_model(rnn_checkpoint)
        trans_model, _, _, trans_training_info = self.load_transformer_model(trans_checkpoint)

        # Load test data
        print("\nLoading test data...")
        preprocessor = Preprocessor()
        test_data = preprocessor.load_and_preprocess(self.config.TEST_PATH)
        print(f"Test samples: {len(test_data)}\n")

        # Create dataloaders
        rnn_loader = get_dataloader(test_data, vocab_zh, vocab_en,
                                   self.config.BATCH_SIZE, shuffle=False)
        trans_loader = get_dataloader(test_data, vocab_zh, vocab_en,
                                     self.config.BATCH_SIZE, shuffle=False)

        # Run comprehensive evaluation
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)

        # Basic metrics
        print("\n1. Evaluating RNN model...")
        rnn_results = self._evaluate_single_model(
            rnn_model, rnn_loader, vocab_zh, vocab_en, "RNN",
            rnn_training_info
        )

        print("\n2. Evaluating Transformer model...")
        trans_results = self._evaluate_single_model(
            trans_model, trans_loader, vocab_zh, vocab_en, "Transformer",
            trans_training_info
        )

        # Generate and save report
        report_file = os.path.join(output_dir, "comparison_report.txt")
        report = self.generate_comparison_report(rnn_results, trans_results, report_file)

        # Save detailed results as JSON
        detailed_results = {
            'rnn': rnn_results,
            'transformer': trans_results,
            'config': {
                'rnn_embed_dim': self.config.RNN_EMBED_DIM,
                'rnn_hidden_dim': self.config.RNN_HIDDEN_DIM,
                'rnn_num_layers': self.config.RNN_NUM_LAYERS,
                'rnn_cell_type': self.config.RNN_CELL_TYPE,
                'rnn_attention': self.config.ATTENTION_TYPE,
                'trans_d_model': self.config.TRANS_D_MODEL,
                'trans_nhead': self.config.TRANS_NHEAD,
                'trans_num_layers': self.config.TRANS_NUM_ENCODER_LAYERS,
                'position_embedding': self.config.POSITION_EMBEDDING,
                'norm_type': self.config.NORM_TYPE
            }
        }

        json_file = os.path.join(output_dir, "detailed_results.json")
        with open(json_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"\n✓ Report saved to: {report_file}")
        print(f"✓ Detailed results saved to: {json_file}")

        # Print summary
        print("\n" + "="*60)
        print("QUICK SUMMARY")
        print("="*60)
        print(f"RNN Parameters: {rnn_results['params']:,}")
        print(f"Transformer Parameters: {trans_results['params']:,}")
        print(f"\nRNN BLEU-4: {rnn_results['beam']['bleu4']:.4f}")
        print(f"Transformer BLEU-4: {trans_results['beam']['bleu4']:.4f}")
        improvement = (trans_results['beam']['bleu4'] - rnn_results['beam']['bleu4']) / rnn_results['beam']['bleu4'] * 100
        print(f"Improvement: {improvement:+.2f}%")

        return detailed_results

    def _evaluate_single_model(self, model, dataloader, vocab_src, vocab_tgt,
                              model_type: str, training_info: Dict):
        """Evaluate a single model comprehensively"""
        results = {
            'params': count_parameters(model),
            'training_info': training_info
        }

        # Basic evaluation
        print(f"  Running basic evaluation...")
        basic_results = self._basic_evaluation(model, dataloader, vocab_src, vocab_tgt, model_type)
        results.update(basic_results)

        # Length analysis
        print(f"  Analyzing performance by sentence length...")
        length_results = self.analyze_by_length(model, dataloader, vocab_src, vocab_tgt, model_type)
        results['length_analysis'] = length_results

        # Speed analysis
        print(f"  Measuring inference speed...")
        speed_results = self.measure_inference_speed(model, dataloader, vocab_src, vocab_tgt, model_type)
        results['speed_analysis'] = speed_results

        # Memory analysis
        print(f"  Analyzing memory efficiency...")
        memory_results = self.analyze_memory_efficiency(model, [16, 32, 64])
        results['memory'] = memory_results

        # Qualitative analysis
        print(f"  Generating sample translations...")
        samples = self.qualitative_analysis(model, dataloader, vocab_src, vocab_tgt, model_type, num_samples=10)
        results['samples'] = samples

        return results

    def _basic_evaluation(self, model, dataloader, vocab_src, vocab_tgt, model_type: str):
        """Basic model evaluation (from original evaluate.py)"""
        model.eval()

        references = []
        hypotheses_greedy = []
        hypotheses_beam = []
        greedy_times = []
        beam_times = []

        with torch.no_grad():
            for src, tgt, src_len, tgt_len in tqdm(dataloader, desc=f"Evaluating {model_type}"):
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                src_len = src_len.to(self.device)

                for i in range(src.size(0)):
                    src_seq = src[i:i+1]
                    src_len_seq = src_len[i:i+1]

                    # Greedy decoding
                    start_time = time.time()
                    pred_tokens_greedy = greedy_decode(
                        model, src_seq, src_len_seq, vocab_tgt,
                        self.config.MAX_DECODE_LENGTH, self.device
                    )
                    greedy_times.append(time.time() - start_time)

                    # Beam search decoding
                    start_time = time.time()
                    pred_tokens_beam = beam_search_decode(
                        model, src_seq, src_len_seq, vocab_tgt,
                        self.config.BEAM_SIZE, self.config.MAX_DECODE_LENGTH,
                        self.device
                    )
                    beam_times.append(time.time() - start_time)

                    # Get reference tokens
                    ref_tokens = tgt[i].cpu().tolist()
                    ref_tokens = [t for t in ref_tokens if t not in [
                        vocab_tgt.pad_idx, vocab_tgt.bos_idx, vocab_tgt.eos_idx
                    ]]

                    # Convert to words
                    ref_words = vocab_tgt.decode(ref_tokens, skip_special=False)
                    pred_words_greedy = vocab_tgt.decode(pred_tokens_greedy, skip_special=False)
                    pred_words_beam = vocab_tgt.decode(pred_tokens_beam, skip_special=False)

                    references.append(ref_words)
                    hypotheses_greedy.append(pred_words_greedy)
                    hypotheses_beam.append(pred_words_beam)

        # Calculate metrics
        metrics_greedy = calculate_all_metrics(references, hypotheses_greedy)
        metrics_beam = calculate_all_metrics(references, hypotheses_beam)

        return {
            'greedy': {
                'metrics': metrics_greedy,
                'avg_time': np.mean(greedy_times) if greedy_times else 0
            },
            'beam': {
                'metrics': metrics_beam,
                'avg_time': np.mean(beam_times) if beam_times else 0
            }
        }


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive comparison of RNN and Transformer NMT models"
    )
    parser.add_argument('--rnn-checkpoint', type=str, required=True,
                       help='Path to RNN model checkpoint')
    parser.add_argument('--transformer-checkpoint', type=str, required=True,
                       help='Path to Transformer model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./comparison_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Run comparison
    config = Config()
    comparator = ModelComparator(config)
    results = comparator.compare_models(
        args.rnn_checkpoint,
        args.transformer_checkpoint,
        args.output_dir
    )

    print("\n✓ Comparison complete!")


if __name__ == "__main__":
    main()
