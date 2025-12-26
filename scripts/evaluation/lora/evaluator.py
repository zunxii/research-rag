"""
LoRA fine-tuning impact evaluation
"""
from pathlib import Path
from datetime import datetime
import json

from scripts.evaluation.lora.embedding_comparison import EmbeddingComparator
from scripts.evaluation.lora.alignment_improvement import AlignmentAnalyzer


class LoRAEvaluator:
    """Evaluates LoRA fine-tuning impact"""
    
    def __init__(self, output_dir: str, device: str = "cpu"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # Check if LoRA is available
        lora_path = Path("outputs/models/trained_lora")
        if not lora_path.exists():
            self.has_lora = False
            print("⚠ LoRA model not found. Skipping evaluation.")
        else:
            self.has_lora = True
            self.embedding_comparator = EmbeddingComparator(device)
            self.alignment_analyzer = AlignmentAnalyzer(device)
    
    def run_evaluation(self):
        """Run LoRA evaluation"""
        if not self.has_lora:
            self.results["status"] = "lora_not_available"
            return
        
        print("\n" + "="*70)
        print("LORA FINE-TUNING EVALUATION")
        print("="*70 + "\n")
        
        # Test 1: Embedding comparison
        print(" Comparing base vs LoRA embeddings...")
        self.results["tests"]["embedding_comparison"] = \
            self.embedding_comparator.compare()
        
        # Test 2: Alignment improvement
        print(" Analyzing alignment improvement...")
        self.results["tests"]["alignment_improvement"] = \
            self.alignment_analyzer.analyze()
    
    def save_results(self):
        """Save evaluation results"""
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")
        
        if self.has_lora:
            summary_path = self.output_dir / "summary.txt"
            self._save_summary(summary_path)
            print(f"✓ Summary saved to {summary_path}")
    
    def _save_summary(self, path: Path):
        """Save human-readable summary"""
        with open(path, 'w') as f:
            f.write("LORA EVALUATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            # Embedding comparison
            ec = self.results["tests"]["embedding_comparison"]
            f.write("Embedding Comparison:\n")
            f.write(f"  Distance (L2): {ec['distance_l2']:.4f}\n")
            f.write(f"  Cosine similarity: {ec['cosine_similarity']:.4f}\n\n")
            
            # Alignment
            aa = self.results["tests"]["alignment_improvement"]
            if "improvement" in aa:
                f.write("Alignment Improvement:\n")
                f.write(f"  Base similarity: {aa['base_similarity']:.4f}\n")
                f.write(f"  LoRA similarity: {aa['lora_similarity']:.4f}\n")
                f.write(f"  Improvement: {aa['improvement']:.4f}\n")

