"""
Robustness analysis
"""
from collections import Counter
from typing import List, Dict
import numpy as np


class RobustnessAnalyzer:
    """Analyzes robustness from stability tests"""
    
    def analyze(self, stability_tests: List[Dict]) -> Dict:
        """Analyze stability test results"""
        return {
            "avg_divergences": self._compute_avg_divergences(stability_tests),
            "robustness_distribution": self._compute_robustness_dist(stability_tests),
            "per_diagnosis_stability": self._analyze_per_diagnosis(stability_tests)
        }
    
    def _compute_avg_divergences(self, tests: List[Dict]) -> Dict:
        """Compute average JS divergences"""
        divergences = {
            "no_text": [],
            "no_image": [],
            "noisy": []
        }
        
        for test in tests:
            js_divs = test["stability"]["js_divergence"]
            for key in divergences:
                if key in js_divs:
                    divergences[key].append(js_divs[key])
        
        return {
            key: np.mean(vals) if vals else 0.0
            for key, vals in divergences.items()
        }
    
    def _compute_robustness_dist(self, tests: List[Dict]) -> Dict:
        """Compute robustness level distribution"""
        levels = [
            test["stability"]["robustness_level"]
            for test in tests
        ]
        return dict(Counter(levels))
    
    def _analyze_per_diagnosis(self, tests: List[Dict]) -> Dict:
        """Analyze stability per diagnosis"""
        by_diagnosis = {}
        
        for test in tests:
            diag = test["diagnosis"]
            if diag not in by_diagnosis:
                by_diagnosis[diag] = []
            
            by_diagnosis[diag].append(
                test["stability"]["robustness_level"]
            )
        
        return {
            diag: {
                "count": len(levels),
                "high": levels.count("high"),
                "medium": levels.count("medium"),
                "low": levels.count("low")
            }
            for diag, levels in by_diagnosis.items()
        }
