"""
Entry point for counterfactual evaluation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.evaluation.counterfactual.evaluator import CounterfactualEvaluator
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-dir", default="outputs/kb/kb_final_v2")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output-dir", 
                       default="outputs/evaluation/counterfactual")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    evaluator = CounterfactualEvaluator(
        kb_dir=args.kb_dir,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=args.device
    )
    
    evaluator.run_evaluation()
    evaluator.save_results()


if __name__ == "__main__":
    main()
