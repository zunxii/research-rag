"""
Entry point for LoRA evaluation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.evaluation.lora.evaluator import LoRAEvaluator
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/evaluation/lora")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    evaluator = LoRAEvaluator(
        output_dir=args.output_dir,
        device=args.device
    )
    
    evaluator.run_evaluation()
    evaluator.save_results()


if __name__ == "__main__":
    main()