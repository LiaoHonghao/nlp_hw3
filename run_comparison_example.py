#!/usr/bin/env python3
"""
Example script demonstrating how to use compare_models.py
This script shows the complete workflow: training, evaluation, and comparison
"""

import os
import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a shell command and print status"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        return False

    return True


def check_checkpoint(path, model_type):
    """Check if checkpoint exists"""
    if not os.path.exists(path):
        print(f"Warning: {model_type} checkpoint not found at {path}")
        return False
    print(f"✓ {model_type} checkpoint found: {path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Example workflow for comparing RNN and Transformer models"
    )
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training steps (assume models already trained)')
    parser.add_argument('--rnn-checkpoint', type=str,
                       help='Custom path to RNN checkpoint')
    parser.add_argument('--transformer-checkpoint', type=str,
                       help='Custom path to Transformer checkpoint')
    parser.add_argument('--output-dir', type=str, default='./comparison_results',
                       help='Output directory for comparison results')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON WORKFLOW")
    print("="*80)

    # Step 1: Training (optional)
    if not args.skip_training:
        print("\nStep 1: Training Models")
        print("-" * 80)

        # Train RNN model
        print("\nTraining RNN model...")
        rnn_cmd = "python train_rnn.py --attention dot --cell LSTM --epochs 20"
        if not run_command(rnn_cmd, "Training RNN Model"):
            print("RNN training failed. Continuing with existing checkpoints...")

        # Train Transformer model
        print("\nTraining Transformer model...")
        trans_cmd = "python train_transformer.py --position-embedding sinusoidal --norm-type LayerNorm --epochs 20"
        if not run_command(trans_cmd, "Training Transformer Model"):
            print("Transformer training failed. Continuing with existing checkpoints...")

    else:
        print("\n✓ Skipping training (using existing models)")

    # Step 2: Locate checkpoints
    print("\n\nStep 2: Locating Model Checkpoints")
    print("-" * 80)

    # Use custom paths or find latest checkpoints
    if args.rnn_checkpoint:
        rnn_checkpoint = args.rnn_checkpoint
    else:
        # Find latest RNN checkpoint
        rnn_dir = "./checkpoints/rnn"
        if os.path.exists(rnn_dir):
            checkpoints = [f for f in os.listdir(rnn_dir) if f.endswith('.pt') and 'best' in f]
            if checkpoints:
                rnn_checkpoint = os.path.join(rnn_dir, sorted(checkpoints)[-1])
            else:
                rnn_checkpoint = None
        else:
            rnn_checkpoint = None

    if args.transformer_checkpoint:
        trans_checkpoint = args.transformer_checkpoint
    else:
        # Find latest Transformer checkpoint
        trans_dir = "./checkpoints/transformer"
        if os.path.exists(trans_dir):
            checkpoints = [f for f in os.listdir(trans_dir) if f.endswith('.pt') and 'best' in f]
            if checkpoints:
                trans_checkpoint = os.path.join(trans_dir, sorted(checkpoints)[-1])
            else:
                trans_checkpoint = None
        else:
            trans_checkpoint = None

    # Verify checkpoints
    rnn_exists = check_checkpoint(rnn_checkpoint, "RNN") if rnn_checkpoint else False
    trans_exists = check_checkpoint(trans_checkpoint, "Transformer") if trans_checkpoint else False

    if not rnn_exists or not trans_exists:
        print("\n" + "!"*60)
        print("ERROR: Could not find required checkpoints")
        print("!"*60)
        print("\nPlease ensure you have trained both models or specify checkpoints manually:")
        print("  --rnn-checkpoint <path>")
        print("  --transformer-checkpoint <path>")
        print("\nExample:")
        print("  python run_comparison_example.py \\")
        print("    --rnn-checkpoint ./checkpoints/rnn/best_model.pt \\")
        print("    --transformer-checkpoint ./checkpoints/transformer/best_model.pt")
        sys.exit(1)

    # Step 3: Run comparison
    print("\n\nStep 3: Running Comprehensive Comparison")
    print("-" * 80)

    compare_cmd = (
        f"python compare_models.py "
        f"--rnn-checkpoint {rnn_checkpoint} "
        f"--transformer-checkpoint {trans_checkpoint} "
        f"--output-dir {args.output_dir}"
    )

    if not run_command(compare_cmd, "Comprehensive Model Comparison"):
        print("\nComparison failed!")
        sys.exit(1)

    # Step 4: Display results
    print("\n\nStep 4: Comparison Results")
    print("="*80)

    report_file = os.path.join(args.output_dir, "comparison_report.txt")
    json_file = os.path.join(args.output_dir, "detailed_results.json")

    if os.path.exists(report_file):
        print(f"\n✓ Comprehensive report saved to:")
        print(f"  {report_file}")

        # Print first few lines of report
        print("\n" + "-"*80)
        print("REPORT PREVIEW:")
        print("-"*80)
        with open(report_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:50]):  # Print first 50 lines
                print(line.rstrip())
            if len(lines) > 50:
                print("\n... (see full report for complete analysis)")

    if os.path.exists(json_file):
        print(f"\n✓ Detailed results (JSON) saved to:")
        print(f"  {json_file}")

    # Summary
    print("\n\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print(f"  1. Review the full report: {report_file}")
    print(f"  2. Analyze detailed metrics: {json_file}")
    print(f"  3. Compare different model configurations")
    print(f"  4. Experiment with hyperparameters")
    print("\nTo compare different configurations:")
    print("  python compare_models.py \\")
    print("    --rnn-checkpoint <path> \\")
    print("    --transformer-checkpoint <path> \\")
    print("    --output-dir <output_dir>")

    # Additional experiments suggestions
    print("\n" + "-"*80)
    print("SUGGESTED EXPERIMENTS:")
    print("-"*80)
    print("\n1. Compare different attention mechanisms:")
    print("   python train_rnn.py --attention dot")
    print("   python train_rnn.py --attention additive")
    print("   python compare_models.py --rnn-checkpoint <dot> --rnn-checkpoint <additive>")
    print("\n2. Compare different RNN cells:")
    print("   python train_rnn.py --cell LSTM")
    print("   python train_rnn.py --cell GRU")
    print("\n3. Compare Transformer architectures:")
    print("   python train_transformer.py --position-embedding sinusoidal")
    print("   python train_transformer.py --position-embedding learned")
    print("\n4. Compare model scales:")
    print("   python train_transformer.py --d-model 256 --num-layers 2")
    print("   python train_transformer.py --d-model 512 --num-layers 6")
    print("\n5. Analyze length sensitivity:")
    print("   - The comparison script automatically analyzes this!")
    print("   - Check 'Scalability & Generalization' section in report")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
