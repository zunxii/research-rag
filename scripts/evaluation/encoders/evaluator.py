"""
Encoder evaluation orchestrator
"""
from pathlib import Path
from datetime import datetime
import json

from scripts.evaluation.encoders.embedding_quality import EmbeddingQualityTester
from scripts.evaluation.encoders.fusion_analysis import FusionAnalyzer
from scripts.evaluation.encoders.modality_alignment import ModalityAlignmentTester


class EncoderEvaluator:
    """Orchestrates encoder evaluation"""
    
    def __init__(self, output_dir: str, device: str = "cpu"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # Initialize testers
        self.embedding_tester = EmbeddingQualityTester(device)
        self.fusion_analyzer = FusionAnalyzer(device)
        self.alignment_tester = ModalityAlignmentTester(device)
    
    def run_all_tests(self):
        """Run all encoder tests"""
        print("\n" + "="*70)
        print("ENCODER & FUSION EVALUATION")
        print("="*70 + "\n")
        
        # Test 1: Embedding Quality
        print(" Testing embedding quality...")
        self.results["tests"]["embedding_quality"] = \
            self.embedding_tester.test()
        
        # Test 2: Fusion Analysis
        print(" Analyzing fusion behavior...")
        self.results["tests"]["fusion_analysis"] = \
            self.fusion_analyzer.analyze()
        
        # Test 3: Modality Alignment
        print(" Testing modality alignment...")
        self.results["tests"]["modality_alignment"] = \
            self.alignment_tester.test()
    
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
        with open(path, 'w') as f:
            f.write("ENCODER EVALUATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            # Embedding quality
            eq = self.results["tests"]["embedding_quality"]
            f.write("Embedding Quality:\n")
            f.write(f"  Normalization: {eq['normalization_check']}\n")
            f.write(f"  Deterministic: {eq['deterministic_check']}\n")
            f.write(f"  Dimension: {eq['dimension']}\n\n")
            
            # Fusion
            fa = self.results["tests"]["fusion_analysis"]
            f.write("Fusion Analysis:\n")
            f.write(f"  Gate mean: {fa['gate_statistics']['mean']:.4f}\n")
            f.write(f"  Output normalized: {fa['output_normalized']}\n\n")

