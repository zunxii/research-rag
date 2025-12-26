"""
Pipeline orchestration logic
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import json


class PipelineOrchestrator:
    """Orchestrates pipeline stages"""
    
    def __init__(self, config):
        self.config = config
        self.results = {
            "timestamp": config.timestamp,
            "stages": {}
        }
        
        self.stage_scripts = {
            "train_lora": "scripts/training/train_lora.py",
            "train_fusion": "scripts/training/train_fusion.py",
            "build_kb": "scripts/kb/build_kb.py",
            "eval_retrieval": "scripts/evaluation/retrieval/run_eval.py",
            "eval_encoders": "scripts/evaluation/encoders/run_eval.py",
            "eval_counterfactual": "scripts/evaluation/counterfactual/run_eval.py",
            "eval_lora": "scripts/evaluation/lora/run_eval.py"
        }
    
    def run_stage(self, stage_name: str, script_path: str, 
                  args: List[str] = None) -> bool:
        """Run a single stage"""
        print(f"\n{'='*70}")
        print(f"STAGE: {stage_name}")
        print(f"{'='*70}\n")
        
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
        
        start_time = datetime.now()
        result = subprocess.run(cmd, cwd=Path.cwd())
        end_time = datetime.now()
        
        success = result.returncode == 0
        duration = (end_time - start_time).total_seconds()
        
        self.results["stages"][stage_name] = {
            "success": success,
            "duration_seconds": duration,
            "timestamp": start_time.isoformat()
        }
        
        status = "✓" if success else "✗"
        print(f"\n{status} {stage_name} - {duration:.2f}s")
        return success
    
    def run_full_pipeline(self, skip_training: bool = False):
        """Run complete pipeline"""
        print(f"\n{'#'*70}")
        print(f"# PIPELINE START: {self.config.timestamp}")
        print(f"{'#'*70}\n")
        
        self.config.save()
        
        # Training
        if not skip_training:
            if self.config.get("training.lora.enabled"):
                if not self.run_stage("LoRA Training", 
                                     self.stage_scripts["train_lora"]):
                    return self._finish(False)
            
            if self.config.get("training.fusion.enabled"):
                if not self.run_stage("Fusion Training",
                                     self.stage_scripts["train_fusion"]):
                    return self._finish(False)
        
        # KB Building
        if self.config.get("kb_building.enabled"):
            if not self.run_stage("KB Building",
                                 self.stage_scripts["build_kb"]):
                return self._finish(False)
        
        # Evaluation
        self._run_evaluations()
        
        return self._finish(True)
    
    def run_stages(self, stage_names: List[str]):
        """Run specific stages"""
        stage_map = {
            "train": ["train_lora", "train_fusion"],
            "kb": ["build_kb"],
            "eval": ["eval_retrieval", "eval_encoders", 
                    "eval_counterfactual", "eval_lora"]
        }
        
        for stage_name in stage_names:
            if stage_name in stage_map:
                for substage in stage_map[stage_name]:
                    if substage in self.stage_scripts:
                        self.run_stage(substage, self.stage_scripts[substage])
        
        self._finish(True)
    
    def _run_evaluations(self):
        """Run all evaluation stages"""
        eval_config = self.config.config.get("evaluation", {})
        
        for eval_name in ["retrieval", "encoders", "counterfactual", "lora"]:
            if eval_config.get(eval_name, {}).get("enabled"):
                script_key = f"eval_{eval_name}"
                self.run_stage(f"{eval_name.title()} Evaluation",
                              self.stage_scripts[script_key])
    
    def _finish(self, success: bool):
        """Finish pipeline and save results"""
        results_path = self.config.output_root / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'#'*70}")
        print(f"# PIPELINE {'SUCCESS' if success else 'FAILED'}")
        print(f"# Results: {self.config.output_root}")
        print(f"{'#'*70}\n")
        
        return success