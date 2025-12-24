#!/usr/bin/env python3
"""
Main Entry Point for Research Pipeline
Just a convenience CLI wrapper - all logic in separate scripts
"""

import argparse
import sys
import subprocess
from pathlib import Path


def run_script(script_path, args_list=None):
    """Execute a script with optional arguments."""
    cmd = [sys.executable, str(script_path)]
    if args_list:
        cmd.extend(args_list)
    
    result = subprocess.run(cmd, cwd=Path.cwd())
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Medical Multimodal Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train LoRA adapters
  python main.py train-lora
  
  # Train fusion module  
  python main.py train-fusion
  
  # Build full knowledge base
  python main.py build-kb
  
  # Build KB smoke test
  python main.py build-kb-smoke
  
  # Run single query inference
  python main.py infer --query-text "..." --query-image "path/to/img.jpg"
  
  # Run counterfactual reasoning
  python main.py reason --query-text "..." --query-image "path/to/img.jpg"
  
  # Evaluate retrieval performance
  python main.py evaluate-retrieval
  
  # Run all tests
  python main.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # ========================================
    # TRAINING COMMANDS
    # ========================================
    subparsers.add_parser(
        "train-lora",
        help="Train LoRA adapters (calls scripts/training/train_lora.py)"
    )
    
    subparsers.add_parser(
        "train-fusion",
        help="Train fusion module (calls scripts/training/train_fusion.py)"
    )
    
    # ========================================
    # KB BUILDING COMMANDS
    # ========================================
    subparsers.add_parser(
        "build-kb",
        help="Build full knowledge base (calls scripts/kb/build_kb.py)"
    )
    
    subparsers.add_parser(
        "build-kb-smoke",
        help="Build smoke test KB (calls scripts/kb/build_kb_smoke.py)"
    )
    
    subparsers.add_parser(
        "build-kb-dry",
        help="Build dry run KB (calls scripts/kb/build_kb_dry.py)"
    )
    
    # ========================================
    # INFERENCE COMMANDS
    # ========================================
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run single query inference"
    )
    infer_parser.add_argument("--query-text", required=True)
    infer_parser.add_argument("--query-image", required=False)
    infer_parser.add_argument("--kb-dir", default="outputs/kb/kb_final_v2")
    infer_parser.add_argument("--top-k", type=int, default=5)
    
    reason_parser = subparsers.add_parser(
        "reason",
        help="Run counterfactual reasoning"
    )
    reason_parser.add_argument("--query-text", required=True)
    reason_parser.add_argument("--query-image", required=True)
    reason_parser.add_argument("--kb-dir", default="outputs/kb/kb_final_v2")
    
    subparsers.add_parser(
        "reason-full",
        help="Run full pipeline with Gemini explanation"
    )
    
    # ========================================
    # EVALUATION COMMANDS
    # ========================================
    subparsers.add_parser(
        "evaluate-retrieval",
        help="Evaluate retrieval performance"
    )
    
    # ========================================
    # TESTING COMMANDS
    # ========================================
    test_parser = subparsers.add_parser(
        "test",
        help="Run test suite"
    )
    test_parser.add_argument("--verbose", "-v", action="store_true")
    test_parser.add_argument("--module", help="Specific test module to run")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # ========================================
    # Execute Commands (just call existing scripts)
    # ========================================
    if args.command == "train-lora":
        print(" Running: scripts/training/train_lora.py")
        return run_script("scripts/training/train_lora.py")
        
    elif args.command == "train-fusion":
        print(" Running: scripts/training/train_fusion.py")
        return run_script("scripts/training/train_fusion.py")
        
    elif args.command == "build-kb":
        print(" Running: scripts/kb/build_kb.py")
        return run_script("scripts/kb/build_kb.py")
        
    elif args.command == "build-kb-smoke":
        print(" Running: scripts/kb/build_kb_smoke.py")
        return run_script("scripts/kb/build_kb_smoke.py")
        
    elif args.command == "build-kb-dry":
        print(" Running: scripts/kb/build_kb_dry.py")
        return run_script("scripts/kb/build_kb_dry.py")
        
    elif args.command == "infer":
        print(" Running: scripts/inference/run_single_query.py")
        # Pass through arguments
        script_args = []
        if args.query_text:
            script_args.extend(["--query-text", args.query_text])
        if args.query_image:
            script_args.extend(["--query-image", args.query_image])
        if args.kb_dir:
            script_args.extend(["--kb-dir", args.kb_dir])
        if args.top_k:
            script_args.extend(["--top-k", str(args.top_k)])
        
        return run_script("scripts/inference/run_single_query.py", script_args)
        
    elif args.command == "reason":
        print(" Running: scripts/inference/run_counterfactuals.py")
        script_args = [
            "--query-text", args.query_text,
            "--query-image", args.query_image,
            "--kb-dir", args.kb_dir
        ]
        return run_script("scripts/inference/run_counterfactuals.py", script_args)
        
    elif args.command == "reason-full":
        print(" Running: scripts/inference/run_counterfactual_pipeline.py")
        return run_script("scripts/inference/run_counterfactual_pipeline.py")
        
    elif args.command == "evaluate-retrieval":
        print(" Running: scripts/evaluation/run_retrieval_eval.py")
        return run_script("scripts/evaluation/run_retrieval_eval.py")
        
    elif args.command == "test":
        print(" Running test suite")
        import pytest
        
        pytest_args = []
        if args.verbose:
            pytest_args.append("-v")
        if args.module:
            pytest_args.append(f"tests/test_{args.module}.py")
        else:
            pytest_args.append("tests/")
        
        return pytest.main(pytest_args)


if __name__ == "__main__":
    sys.exit(main())