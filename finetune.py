"""
Finetuning script for pretrained Transformer models (T5, mT5, etc.) on NMT tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.preprocessor import Preprocessor
from data.vocabulary import Vocabulary
from data.dataloader import get_dataloader
from utils.metrics import calculate_all_metrics
from utils.training_utils import (
    save_checkpoint, load_checkpoint, count_parameters,
    AverageMeter, Timer, print_model_info, WarmupScheduler
)
from utils.beam_search import greedy_decode, beam_search_decode


def train_epoch(model, dataloader, optimizer, scheduler, config, epoch, tokenizer):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    timer = Timer()
    timer.start()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(pbar):
        # Decode token IDs back to text
        src_texts = [tokenizer.decode(s, skip_special_tokens=True) for s in src]
        tgt_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in tgt]

        # Tokenize with pretrained tokenizer
        src_encoded = tokenizer(src_texts, padding=True, truncation=True,
                               max_length=config.MAX_LEN, return_tensors='pt')
        tgt_encoded = tokenizer(tgt_texts, padding=True, truncation=True,
                               max_length=config.MAX_LEN, return_tensors='pt')

        # Move to device
        input_ids = src_encoded['input_ids'].to(config.DEVICE)
        attention_mask = src_encoded['attention_mask'].to(config.DEVICE)
        labels = tgt_encoded['input_ids'].to(config.DEVICE)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

        optimizer.step()
        scheduler.step()

        # Update metrics
        losses.update(loss.item(), input_ids.size(0))

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    elapsed = timer.stop()
    return losses.avg, elapsed


def evaluate(model, dataloader, config, tokenizer, vocab_tgt):
    """Evaluate on validation/test set"""
    model.eval()
    losses = AverageMeter()

    references = []
    hypotheses = []

    with torch.no_grad():
        for src, tgt, src_len, tgt_len in tqdm(dataloader, desc="Evaluating"):
            # Decode token IDs back to text
            src_texts = [tokenizer.decode(s, skip_special_tokens=True) for s in src]
            tgt_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in tgt]

            # Tokenize
            src_encoded = tokenizer(src_texts, padding=True, truncation=True,
                                   max_length=config.MAX_LEN, return_tensors='pt')
            tgt_encoded = tokenizer(tgt_texts, padding=True, truncation=True,
                                   max_length=config.MAX_LEN, return_tensors='pt')

            input_ids = src_encoded['input_ids'].to(config.DEVICE)
            attention_mask = src_encoded['attention_mask'].to(config.DEVICE)
            labels = tgt_encoded['input_ids'].to(config.DEVICE)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            losses.update(loss.item(), input_ids.size(0))

            # Generate translations
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.MAX_LEN,
                num_beams=config.BEAM_SIZE,
                early_stopping=True
            )

            # Decode predictions and references
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            hypotheses.extend(preds)
            references.extend(refs)

    # Calculate metrics
    bleu, precision = calculate_all_metrics(hypotheses, references, vocab_tgt)

    return losses.avg, bleu, precision


def main():
    parser = argparse.ArgumentParser(description='Finetune pretrained model for NMT')
    parser.add_argument('--model-name', type=str, default='t5-base',
                       help='Pretrained model name (t5-base, mt5-base, etc.)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--warmup-steps', type=int, default=500,
                       help='Warmup steps')
    parser.add_argument('--use-large', action='store_true',
                       help='Use large dataset (100k)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--freeze-encoder', action='store_true',
                       help='Freeze encoder parameters')

    args = parser.parse_args()

    # Setup
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.LR = args.lr
    config.EPOCHS = args.epochs
    config.WARMUP_STEPS = args.warmup_steps

    os.makedirs('./checkpoints/pretrained', exist_ok=True)

    # Load tokenizer and model
    print(f"Loading pretrained model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model = model.to(config.DEVICE)

    # Freeze encoder if requested
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen")

    print(f"Model parameters: {count_parameters(model):,}")

    # Load data
    print("Loading data...")
    preprocessor = Preprocessor()
    vocab_src = Vocabulary()
    vocab_tgt = Vocabulary()

    dataset_type = 'train_100k' if args.use_large else 'train_10k'
    train_dataloader = get_dataloader(
        dataset_type, config.BATCH_SIZE, vocab_src, vocab_tgt, preprocessor
    )
    valid_dataloader = get_dataloader(
        'valid', config.BATCH_SIZE, vocab_src, vocab_tgt, preprocessor
    )
    test_dataloader = get_dataloader(
        'test', config.BATCH_SIZE, vocab_src, vocab_tgt, preprocessor
    )

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.LR)
    total_steps = len(train_dataloader) * config.EPOCHS
    scheduler = WarmupScheduler(optimizer, config.WARMUP_STEPS, total_steps)

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    # Training loop
    best_bleu = 0
    for epoch in range(start_epoch, config.EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print(f"{'='*60}")

        # Train
        train_loss, train_time = train_epoch(
            model, train_dataloader, optimizer, scheduler, config, epoch + 1, tokenizer
        )
        print(f"Train Loss: {train_loss:.4f} | Time: {train_time:.2f}s")

        # Validate
        valid_loss, valid_bleu, valid_precision = evaluate(
            model, valid_dataloader, config, tokenizer, vocab_tgt
        )
        print(f"Valid Loss: {valid_loss:.4f} | BLEU: {valid_bleu:.4f}")
        print(f"Precision-1: {valid_precision[0]:.4f} | Precision-4: {valid_precision[3]:.4f}")

        # Save checkpoint
        checkpoint_path = f'./checkpoints/pretrained/{args.model_name}_epoch_{epoch + 1}.pt'
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        # Save best model
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            best_path = f'./checkpoints/pretrained/{args.model_name}_best.pt'
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_path)
            print(f"Best model saved with BLEU: {best_bleu:.4f}")

    # Test
    print(f"\n{'='*60}")
    print("Testing on test set...")
    print(f"{'='*60}")
    test_loss, test_bleu, test_precision = evaluate(
        model, test_dataloader, config, tokenizer, vocab_tgt
    )
    print(f"Test Loss: {test_loss:.4f} | BLEU: {test_bleu:.4f}")
    print(f"Precision-1: {test_precision[0]:.4f} | Precision-4: {test_precision[3]:.4f}")


if __name__ == '__main__':
    main()
