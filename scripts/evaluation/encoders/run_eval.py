"""
Entry point for encoder evaluation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.evaluation.encoders.evaluator import EncoderEvaluator
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/evaluation/encoders")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    evaluator = EncoderEvaluator(
        output_dir=args.output_dir,
        device=args.device
    )
    
    evaluator.run_all_tests()
    evaluator.save_results()


if __name__ == "__main__":
    main()