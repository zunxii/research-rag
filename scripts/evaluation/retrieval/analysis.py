"""
Result analysis and insights
"""
from collections import Counter
from typing import Dict


class ResultsAnalyzer:
    """Analyzes evaluation results"""
    
    def analyze(self, mode_results: Dict) -> Dict:
        """Analyze results across modes"""
        analysis = {}
        
        # Mode comparison
        analysis["mode_comparison"] = self._compare_modes(mode_results)
        
        # Best mode per metric
        analysis["best_modes"] = self._find_best_modes(mode_results)
        
        # Improvement analysis
        if "fusion" in mode_results and "text" in mode_results:
            analysis["fusion_improvement"] = self._compute_improvement(
                mode_results["text"]["metrics"],
                mode_results["fusion"]["metrics"]
            )
        
        return analysis
    
    def _compare_modes(self, mode_results: Dict) -> Dict:
        """Compare metrics across modes"""
        comparison = {}
        metrics = ["R@1", "R@5", "R@10", "MRR", "MAP"]
        
        for metric in metrics:
            comparison[metric] = {
                mode: results["metrics"].get(metric, 0.0)
                for mode, results in mode_results.items()
            }
        
        return comparison
    
    def _find_best_modes(self, mode_results: Dict) -> Dict:
        """Find best mode for each metric"""
        best_modes = {}
        metrics = ["R@1", "R@5", "R@10", "MRR", "MAP"]
        
        for metric in metrics:
            best_mode = max(
                mode_results.keys(),
                key=lambda m: mode_results[m]["metrics"].get(metric, 0.0)
            )
            best_modes[metric] = {
                "mode": best_mode,
                "value": mode_results[best_mode]["metrics"].get(metric, 0.0)
            }
        
        return best_modes
    
    def _compute_improvement(self, baseline: Dict, improved: Dict) -> Dict:
        """Compute improvement percentages"""
        improvements = {}
        for metric in baseline:
            if metric in improved:
                base_val = baseline[metric]
                imp_val = improved[metric]
                if base_val > 0:
                    improvements[metric] = {
                        "absolute": imp_val - base_val,
                        "relative_pct": ((imp_val - base_val) / base_val) * 100
                    }
        return improvements


