"""
Retrieval evaluation orchestrator
"""
from pathlib import Path
from datetime import datetime
import json

from scripts.evaluation.retrieval.metrics import MetricsCalculator
from scripts.evaluation.retrieval.modes import ModeEvaluator
from scripts.evaluation.retrieval.analysis import ResultsAnalyzer


class RetrievalEvaluator:
    """Orchestrates retrieval evaluation"""
    
    def __init__(self, kb_dir: str, output_dir: str, device: str = "cpu"):
        self.kb_dir = Path(kb_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "kb_dir": str(kb_dir),
            "modes": {}
        }
        
        self.metrics_calc = MetricsCalculator()
        self.mode_eval = ModeEvaluator(kb_dir, device)
        self.analyzer = ResultsAnalyzer()
    
    def run_all_evaluations(self):
        """Run all retrieval evaluations"""
        print("\n" + "="*70)
        print("RETRIEVAL EVALUATION")
        print("="*70 + "\n")
        
        # Evaluate each mode
        for mode in ["text", "image", "fusion"]:
            print(f"\n Evaluating {mode.upper()} mode...")
            results = self.mode_eval.evaluate_mode(mode)
            self.results["modes"][mode] = results
        
        # Run analysis
        print("\n Analyzing results...")
        self.results["analysis"] = self.analyzer.analyze(self.results["modes"])
    
    def save_results(self):
        """Save evaluation results"""
        # Save main results
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")
        
        # Save summary
        summary_path = self.output_dir / "summary.txt"
        self._save_summary(summary_path)
        print(f"✓ Summary saved to {summary_path}")
    
    def _save_summary(self, path: Path):
        """Save human-readable summary"""
        with open(path, 'w') as f:
            f.write("RETRIEVAL EVALUATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            for mode, results in self.results["modes"].items():
                f.write(f"{mode.upper()} Mode:\n")
                f.write(f"  R@1:  {results['metrics']['R@1']:.4f}\n")
                f.write(f"  R@5:  {results['metrics']['R@5']:.4f}\n")
                f.write(f"  R@10: {results['metrics']['R@10']:.4f}\n")
                f.write(f"  MRR:  {results['metrics']['MRR']:.4f}\n")
                f.write(f"  MAP:  {results['metrics']['MAP']:.4f}\n\n")

