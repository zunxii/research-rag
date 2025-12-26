"""
Counterfactual stability evaluation orchestrator
"""
from pathlib import Path
from datetime import datetime
import json
import random

from scripts.evaluation.counterfactual.stability_tester import StabilityTester
from scripts.evaluation.counterfactual.robustness_analyzer import RobustnessAnalyzer


class CounterfactualEvaluator:
    """Orchestrates counterfactual evaluation"""
    
    def __init__(self, kb_dir: str, num_samples: int, 
                 output_dir: str, device: str = "cpu"):
        self.kb_dir = Path(kb_dir)
        self.num_samples = num_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": num_samples,
            "stability_tests": []
        }
        
        self.stability_tester = StabilityTester(kb_dir, device)
        self.robustness_analyzer = RobustnessAnalyzer()
    
    def run_evaluation(self):
        """Run counterfactual evaluation"""
        print("\n" + "="*70)
        print("COUNTERFACTUAL STABILITY EVALUATION")
        print("="*70 + "\n")
        
        # Get sample indices
        metadata = self.stability_tester.get_metadata()
        sample_indices = random.sample(
            range(len(metadata)), 
            min(self.num_samples, len(metadata))
        )
        
        print(f" Testing {len(sample_indices)} samples...")
        
        # Run stability tests
        for idx in sample_indices:
            result = self.stability_tester.test_sample(idx)
            self.results["stability_tests"].append(result)
        
        # Analyze results
        print(" Analyzing robustness...")
        self.results["analysis"] = \
            self.robustness_analyzer.analyze(
                self.results["stability_tests"]
            )
    
    def save_results(self):
        """Save evaluation results"""
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")
        
        summary_path = self.output_dir / "summary.txt"
        self._save_summary(summary_path)
        print(f"✓ Summary saved to {summary_path}")
    
    def _save_summary(self, path: Path):
        """Save human-readable summary"""
        analysis = self.results["analysis"]
        
        with open(path, 'w') as f:
            f.write("COUNTERFACTUAL STABILITY SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Samples tested: {self.num_samples}\n\n")
            
            f.write("Average JS Divergences:\n")
            for pert, div in analysis["avg_divergences"].items():
                f.write(f"  {pert}: {div:.4f}\n")
            
            f.write(f"\nRobustness level distribution:\n")
            for level, count in analysis["robustness_distribution"].items():
                f.write(f"  {level}: {count}\n")

