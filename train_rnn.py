"""
Training script for RNN-based machine translation
Supports different attention mechanisms and teacher forcing strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.preprocessor import Preprocessor
from data.vocabulary import Vocabulary
from data.dataloader import get_dataloader
from models.rnn.seq2seq import Seq2Seq
from utils.metrics import calculate_all_metrics
from utils.training_utils import (
    save_checkpoint, load_checkpoint, count_parameters,
    AverageMeter, Timer, print_model_info, get_lr
)
from utils.beam_search import greedy_decode, beam_search_decode


def train_epoch(model, dataloader, optimizer, criterion, config, epoch, vocab_tgt):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    timer = Timer()
    timer.start()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(pbar):
        # Move to device
        src = src.to(config.DEVICE)
        tgt = tgt.to(config.DEVICE)
        src_len = src_len.to(config.DEVICE)

        # Forward pass
        outputs, _ = model(src, src_len, tgt, teacher_forcing_ratio=config.TEACHER_FORCING_RATIO)
        # outputs: (batch_size, tgt_len, vocab_size)

        # Compute loss (ignore first token which is <bos>)
        output_dim = outputs.size(-1)
        outputs = outputs[:, 1:, :].reshape(-1, output_dim)  # (batch_size * (tgt_len-1), vocab_size)
        tgt = tgt[:, 1:].reshape(-1)  # (batch_size * (tgt_len-1))

        loss = criterion(outputs, tgt)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

        optimizer.step()

        # Update metrics
        losses.update(loss.item(), src.size(0))

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'lr': f'{get_lr(optimizer):.6f}'
        })

    elapsed = timer.stop()
    return losses.avg, elapsed


def evaluate(model, dataloader, criterion, config, vocab_src, vocab_tgt, use_beam_search=False):
    """Evaluate on validation/test set"""
    model.eval()
    losses = AverageMeter()

    references = []
    hypotheses = []

    with torch.no_grad():
        for src, tgt, src_len, tgt_len in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            src = src.to(config.DEVICE)
            tgt = tgt.to(config.DEVICE)
            src_len = src_len.to(config.DEVICE)

            # Forward pass for loss
            outputs, _ = model(src, src_len, tgt, teacher_forcing_ratio=0.0)

            # Compute loss
            output_dim = outputs.size(-1)
            outputs_flat = outputs[:, 1:, :].reshape(-1, output_dim)
            tgt_flat = tgt[:, 1:].reshape(-1)
            loss = criterion(outputs_flat, tgt_flat)
            losses.update(loss.item(), src.size(0))

            # Generate translations for BLEU
            for i in range(src.size(0)):
                src_seq = src[i:i+1]
                src_len_seq = src_len[i:i+1]

                if use_beam_search:
                    pred_tokens = beam_search_decode(
                        model, src_seq, src_len_seq, vocab_tgt,
                        config.BEAM_SIZE, config.MAX_DECODE_LENGTH, config.DEVICE
                    )
                else:
                    pred_tokens = greedy_decode(
                        model, src_seq, src_len_seq, vocab_tgt,
                        config.MAX_DECODE_LENGTH, config.DEVICE
                    )

                # Get reference tokens (skip <bos> and <eos>)
                ref_tokens = tgt[i].cpu().tolist()
                ref_tokens = [t for t in ref_tokens if t not in [
                    vocab_tgt.pad_idx, vocab_tgt.bos_idx, vocab_tgt.eos_idx
                ]]

                # Convert to words
                ref_words = vocab_tgt.decode(ref_tokens, skip_special=False)
                pred_words = vocab_tgt.decode(pred_tokens, skip_special=False)

                references.append(ref_words)
                hypotheses.append(pred_words)

    # Calculate metrics
    metrics = calculate_all_metrics(references, hypotheses)
    metrics['loss'] = losses.avg

    return metrics, references, hypotheses


def main(args):
    config = Config()

    # Override config with command line arguments
    if args.attention:
        config.ATTENTION_TYPE = args.attention
    if args.cell:
        config.RNN_CELL_TYPE = args.cell
    if args.teacher_forcing:
        config.TEACHER_FORCING_RATIO = args.teacher_forcing
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.epochs:
        config.NUM_EPOCHS = args.epochs

    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    config.DEVICE = device
    print(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = config.RNN_CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    preprocessor = Preprocessor()

    train_path = config.TRAIN_LARGE_PATH if args.use_large else config.TRAIN_SMALL_PATH
    train_data = preprocessor.load_and_preprocess(train_path)
    valid_data = preprocessor.load_and_preprocess(config.VALID_PATH)
    test_data = preprocessor.load_and_preprocess(config.TEST_PATH)

    print(f"Train samples: {len(train_data)}")
    print(f"Valid samples: {len(valid_data)}")
    print(f"Test samples: {len(test_data)}")

    # Build vocabularies
    print("\nBuilding vocabularies...")
    zh_tokens = [zh for zh, en in train_data]
    en_tokens = [en for zh, en in train_data]

    vocab_zh = Vocabulary("Chinese")
    vocab_zh.build_vocab(zh_tokens, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE)

    vocab_en = Vocabulary("English")
    vocab_en.build_vocab(en_tokens, min_freq=config.MIN_FREQ, max_size=config.MAX_VOCAB_SIZE)

    # Save vocabularies
    vocab_zh.save(os.path.join(checkpoint_dir, 'vocab_zh.pkl'))
    vocab_en.save(os.path.join(checkpoint_dir, 'vocab_en.pkl'))

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader = get_dataloader(train_data, vocab_zh, vocab_en, config.BATCH_SIZE, shuffle=True)
    valid_loader = get_dataloader(valid_data, vocab_zh, vocab_en, config.BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(test_data, vocab_zh, vocab_en, config.BATCH_SIZE, shuffle=False)

    # Create model
    print("\nCreating model...")
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

    print_model_info(model, f"RNN-{config.RNN_CELL_TYPE} ({config.ATTENTION_TYPE} attention)")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_en.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop
    print("\nStarting training...")
    best_bleu = 0.0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        print(f"{'='*50}")

        # Train
        train_loss, train_time = train_epoch(model, train_loader, optimizer, criterion, config, epoch, vocab_en)
        print(f"Train Loss: {train_loss:.4f} | Time: {train_time:.2f}s")

        # Validate
        print("\nValidating...")
        valid_metrics, _, _ = evaluate(model, valid_loader, criterion, config, vocab_zh, vocab_en, use_beam_search=False)

        print(f"Valid Loss: {valid_metrics['loss']:.4f}")
        print(f"Valid BLEU-4: {valid_metrics['bleu4']:.4f}")
        for n in range(1, 5):
            print(f"Valid Precision-{n}: {valid_metrics[f'precision_{n}']:.4f}")

        # Save best model
        if valid_metrics['bleu4'] > best_bleu:
            best_bleu = valid_metrics['bleu4']
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_{config.ATTENTION_TYPE}_{config.RNN_CELL_TYPE}.pt')
            save_checkpoint(model, optimizer, epoch, 0, valid_metrics['loss'], checkpoint_path, vocab_zh, vocab_en)
            print(f"New best model saved! BLEU: {best_bleu:.4f}")

        # Save periodic checkpoint
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_{config.ATTENTION_TYPE}_{config.RNN_CELL_TYPE}.pt')
            save_checkpoint(model, optimizer, epoch, 0, valid_metrics['loss'], checkpoint_path, vocab_zh, vocab_en)

    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)

    # Load best model
    best_checkpoint = os.path.join(checkpoint_dir, f'best_model_{config.ATTENTION_TYPE}_{config.RNN_CELL_TYPE}.pt')
    load_checkpoint(best_checkpoint, model, device=device)

    # Test with greedy decoding
    print("\nGreedy Decoding:")
    test_metrics_greedy, refs, hyps_greedy = evaluate(model, test_loader, criterion, config, vocab_zh, vocab_en, use_beam_search=False)
    print(f"Test BLEU-4: {test_metrics_greedy['bleu4']:.4f}")
    for n in range(1, 5):
        print(f"Test Precision-{n}: {test_metrics_greedy[f'precision_{n}']:.4f}")

    # Test with beam search
    print(f"\nBeam Search Decoding (beam_size={config.BEAM_SIZE}):")
    test_metrics_beam, _, hyps_beam = evaluate(model, test_loader, criterion, config, vocab_zh, vocab_en, use_beam_search=True)
    print(f"Test BLEU-4: {test_metrics_beam['bleu4']:.4f}")
    for n in range(1, 5):
        print(f"Test Precision-{n}: {test_metrics_beam[f'precision_{n}']:.4f}")

    # Save results
    results_path = os.path.join(checkpoint_dir, f'results_{config.ATTENTION_TYPE}_{config.RNN_CELL_TYPE}.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"RNN-{config.RNN_CELL_TYPE} with {config.ATTENTION_TYPE} attention\n")
        f.write(f"{'='*50}\n\n")
        f.write("Greedy Decoding:\n")
        for metric, value in test_metrics_greedy.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write(f"\nBeam Search Decoding (beam_size={config.BEAM_SIZE}):\n")
        for metric, value in test_metrics_beam.items():
            f.write(f"  {metric}: {value:.4f}\n")

        f.write("\n\nSample Translations (first 10):\n")
        f.write("="*50 + "\n")
        for i in range(min(10, len(refs))):
            f.write(f"\nExample {i+1}:\n")
            f.write(f"Reference: {' '.join(refs[i])}\n")
            f.write(f"Greedy:    {' '.join(hyps_greedy[i])}\n")
            f.write(f"Beam:      {' '.join(hyps_beam[i])}\n")

    print(f"\nResults saved to {results_path}")
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN-based NMT")
    parser.add_argument('--attention', type=str, choices=['dot', 'multiplicative', 'additive'],
                       help='Attention mechanism')
    parser.add_argument('--cell', type=str, choices=['LSTM', 'GRU'],
                       help='RNN cell type')
    parser.add_argument('--teacher-forcing', type=float,
                       help='Teacher forcing ratio (0.0-1.0)')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size')
    parser.add_argument('--lr', type=float,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int,
                       help='Number of epochs')
    parser.add_argument('--use-large', action='store_true',
                       help='Use large training set (100k)')

    args = parser.parse_args()
    main(args)
