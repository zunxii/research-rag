"""
Entry point for retrieval evaluation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.evaluation.retrieval.evaluator import RetrievalEvaluator
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-dir", default="outputs/kb/kb_final_v2")
    parser.add_argument("--output-dir", default="outputs/evaluation/retrieval")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    evaluator = RetrievalEvaluator(
        kb_dir=args.kb_dir,
        output_dir=args.output_dir,
        device=args.device,
        query_csv="data/raw/external_queries.csv"
    )
    
    evaluator.run_all_evaluations()
    evaluator.save_results()


if __name__ == "__main__":
    main()
