"""
Main pipeline orchestrator
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.pipeline.orchestrator import PipelineOrchestrator
from scripts.pipeline.config import load_config
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config YAML")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--stages", nargs="+")
    args = parser.parse_args()
    
    config = load_config(args.config)
    orchestrator = PipelineOrchestrator(config)
    
    if args.stages:
        orchestrator.run_stages(args.stages)
    else:
        orchestrator.run_full_pipeline(skip_training=args.skip_training)


if __name__ == "__main__":
    main()
