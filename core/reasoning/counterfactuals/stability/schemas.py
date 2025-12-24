from dataclasses import dataclass
from typing import Dict

@dataclass
class DistributionResult:
    distribution: Dict[str, float]
    num_clusters: int

@dataclass
class StabilityReport:
    js_divergence: Dict[str, float]
    robustness_level: str
