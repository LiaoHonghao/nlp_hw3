"""
Comprehensive evaluation script for trained models
Evaluates RNN and Transformer models and compares their performance
"""

import torch
import os
import sys
import argparse
import time
from tqdm import tqdm
import json

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


def evaluate_model(model, dataloader, vocab_src, vocab_tgt, config, model_type='rnn'):
    """Evaluate a model and return metrics, translations, and timing info"""
    model.eval()

    references = []
    hypotheses_greedy = []
    hypotheses_beam = []

    greedy_times = []
    beam_times = []

    with torch.no_grad():
        for src, tgt, src_len, tgt_len in tqdm(dataloader, desc=f"Evaluating {model_type}"):
            src = src.to(config.DEVICE)
            tgt = tgt.to(config.DEVICE)
            src_len = src_len.to(config.DEVICE)

            for i in range(src.size(0)):
                src_seq = src[i:i+1]
                src_len_seq = src_len[i:i+1]

                # Greedy decoding
                start_time = time.time()
                pred_tokens_greedy = greedy_decode(
                    model, src_seq, src_len_seq, vocab_tgt,
                    config.MAX_DECODE_LENGTH, config.DEVICE
                )
                greedy_times.append(time.time() - start_time)

                # Beam search decoding
                start_time = time.time()
                pred_tokens_beam = beam_search_decode(
                    model, src_seq, src_len_seq, vocab_tgt,
                    config.BEAM_SIZE, config.MAX_DECODE_LENGTH, config.DEVICE
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

    # Add timing information
    avg_greedy_time = sum(greedy_times) / len(greedy_times) if greedy_times else 0
    avg_beam_time = sum(beam_times) / len(beam_times) if beam_times else 0

    return {
        'greedy': {
            'metrics': metrics_greedy,
            'avg_time': avg_greedy_time
        },
        'beam': {
            'metrics': metrics_beam,
            'avg_time': avg_beam_time
        },
        'references': references,
        'hypotheses_greedy': hypotheses_greedy,
        'hypotheses_beam': hypotheses_beam
    }


def load_rnn_model(checkpoint_path, device):
    """Load RNN model from checkpoint"""
    checkpoint = load_checkpoint(checkpoint_path, None, device=device)

    vocab_zh = checkpoint.get('vocab_src')
    vocab_en = checkpoint.get('vocab_tgt')

    # Extract model config from checkpoint
    model_state = checkpoint['model_state_dict']

    # Infer configuration from state dict
    config = Config()

    model = Seq2Seq(
        src_vocab_size=len(vocab_zh),
        tgt_vocab_size=len(vocab_en),
        embed_dim=config.RNN_EMBED_DIM,
        hidden_dim=config.RNN_HIDDEN_DIM,
        num_layers=config.RNN_NUM_LAYERS,
        dropout=config.RNN_DROPOUT,
        cell_type=config.RNN_CELL_TYPE,
        attention_type=config.ATTENTION_TYPE,
        pad_idx=vocab_en.pad_idx
    ).to(device)

    model.load_state_dict(model_state)

    return model, vocab_zh, vocab_en


def load_transformer_model(checkpoint_path, device):
    """Load Transformer model from checkpoint"""
    checkpoint = load_checkpoint(checkpoint_path, None, device=device)

    vocab_zh = checkpoint.get('vocab_src')
    vocab_en = checkpoint.get('vocab_tgt')

    config = Config()

    model = Transformer(
        src_vocab_size=len(vocab_zh),
        tgt_vocab_size=len(vocab_en),
        d_model=config.TRANS_D_MODEL,
        num_heads=config.TRANS_NHEAD,
        num_encoder_layers=config.TRANS_NUM_ENCODER_LAYERS,
        num_decoder_layers=config.TRANS_NUM_DECODER_LAYERS,
        dim_feedforward=config.TRANS_DIM_FEEDFORWARD,
        dropout=config.TRANS_DROPOUT,
        position_embedding=config.POSITION_EMBEDDING,
        norm_type=config.NORM_TYPE,
        pad_idx=vocab_en.pad_idx
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    return model, vocab_zh, vocab_en


def main(args):
    config = Config()
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    config.DEVICE = device

    print(f"Using device: {device}\n")

    # Load test data
    print("Loading test data...")
    preprocessor = Preprocessor()
    test_data = preprocessor.load_and_preprocess(config.TEST_PATH)
    print(f"Test samples: {len(test_data)}\n")

    results = {}

    # Evaluate RNN model if checkpoint provided
    if args.rnn_checkpoint:
        print("="*60)
        print("Evaluating RNN Model")
        print("="*60)

        model, vocab_zh, vocab_en = load_rnn_model(args.rnn_checkpoint, device)
        print(f"Model parameters: {count_parameters(model):,}")

        test_loader = get_dataloader(test_data, vocab_zh, vocab_en, config.BATCH_SIZE, shuffle=False)

        rnn_results = evaluate_model(model, test_loader, vocab_zh, vocab_en, config, 'RNN')

        results['rnn'] = {
            'params': count_parameters(model),
            'greedy': {
                'bleu4': rnn_results['greedy']['metrics']['bleu4'],
                'precision_1': rnn_results['greedy']['metrics']['precision_1'],
                'precision_2': rnn_results['greedy']['metrics']['precision_2'],
                'precision_3': rnn_results['greedy']['metrics']['precision_3'],
                'precision_4': rnn_results['greedy']['metrics']['precision_4'],
                'avg_time': rnn_results['greedy']['avg_time']
            },
            'beam': {
                'bleu4': rnn_results['beam']['metrics']['bleu4'],
                'precision_1': rnn_results['beam']['metrics']['precision_1'],
                'precision_2': rnn_results['beam']['metrics']['precision_2'],
                'precision_3': rnn_results['beam']['metrics']['precision_3'],
                'precision_4': rnn_results['beam']['metrics']['precision_4'],
                'avg_time': rnn_results['beam']['avg_time']
            }
        }

        print("\nRNN Results (Greedy):")
        for key, value in rnn_results['greedy']['metrics'].items():
            print(f"  {key}: {value:.4f}")
        print(f"  Avg inference time: {rnn_results['greedy']['avg_time']:.4f}s")

        print("\nRNN Results (Beam Search):")
        for key, value in rnn_results['beam']['metrics'].items():
            print(f"  {key}: {value:.4f}")
        print(f"  Avg inference time: {rnn_results['beam']['avg_time']:.4f}s")

    # Evaluate Transformer model if checkpoint provided
    if args.transformer_checkpoint:
        print("\n" + "="*60)
        print("Evaluating Transformer Model")
        print("="*60)

        model, vocab_zh, vocab_en = load_transformer_model(args.transformer_checkpoint, device)
        print(f"Model parameters: {count_parameters(model):,}")

        test_loader = get_dataloader(test_data, vocab_zh, vocab_en, config.BATCH_SIZE, shuffle=False)

        trans_results = evaluate_model(model, test_loader, vocab_zh, vocab_en, config, 'Transformer')

        results['transformer'] = {
            'params': count_parameters(model),
            'greedy': {
                'bleu4': trans_results['greedy']['metrics']['bleu4'],
                'precision_1': trans_results['greedy']['metrics']['precision_1'],
                'precision_2': trans_results['greedy']['metrics']['precision_2'],
                'precision_3': trans_results['greedy']['metrics']['precision_3'],
                'precision_4': trans_results['greedy']['metrics']['precision_4'],
                'avg_time': trans_results['greedy']['avg_time']
            },
            'beam': {
                'bleu4': trans_results['beam']['metrics']['bleu4'],
                'precision_1': trans_results['beam']['metrics']['precision_1'],
                'precision_2': trans_results['beam']['metrics']['precision_2'],
                'precision_3': trans_results['beam']['metrics']['precision_3'],
                'precision_4': trans_results['beam']['metrics']['precision_4'],
                'avg_time': trans_results['beam']['avg_time']
            }
        }

        print("\nTransformer Results (Greedy):")
        for key, value in trans_results['greedy']['metrics'].items():
            print(f"  {key}: {value:.4f}")
        print(f"  Avg inference time: {trans_results['greedy']['avg_time']:.4f}s")

        print("\nTransformer Results (Beam Search):")
        for key, value in trans_results['beam']['metrics'].items():
            print(f"  {key}: {value:.4f}")
        print(f"  Avg inference time: {trans_results['beam']['avg_time']:.4f}s")

    # Comparison
    if 'rnn' in results and 'transformer' in results:
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)

        print("\nModel Complexity:")
        print(f"  RNN parameters: {results['rnn']['params']:,}")
        print(f"  Transformer parameters: {results['transformer']['params']:,}")
        print(f"  Ratio: {results['transformer']['params'] / results['rnn']['params']:.2f}x")

        print("\nBLEU-4 Scores (Beam Search):")
        print(f"  RNN: {results['rnn']['beam']['bleu4']:.4f}")
        print(f"  Transformer: {results['transformer']['beam']['bleu4']:.4f}")
        improvement = (results['transformer']['beam']['bleu4'] - results['rnn']['beam']['bleu4']) / results['rnn']['beam']['bleu4'] * 100
        print(f"  Improvement: {improvement:+.2f}%")

        print("\nInference Speed (Beam Search):")
        print(f"  RNN: {results['rnn']['beam']['avg_time']:.4f}s")
        print(f"  Transformer: {results['transformer']['beam']['avg_time']:.4f}s")
        speedup = results['rnn']['beam']['avg_time'] / results['transformer']['beam']['avg_time']
        print(f"  Speedup: {speedup:.2f}x")

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare translation models")
    parser.add_argument('--rnn-checkpoint', type=str,
                       help='Path to RNN model checkpoint')
    parser.add_argument('--transformer-checkpoint', type=str,
                       help='Path to Transformer model checkpoint')
    parser.add_argument('--output', type=str, default='comparison_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    if not args.rnn_checkpoint and not args.transformer_checkpoint:
        print("Error: Please provide at least one checkpoint to evaluate")
        parser.print_help()
        sys.exit(1)

    main(args)
