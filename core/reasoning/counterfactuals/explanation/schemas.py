from dataclasses import dataclass
from typing import List

@dataclass
class Explanation:
    primary_hypothesis: str
    confidence_level: str
    reasoning: List[str]
    uncertainty_notes: List[str]
    rejected_hypotheses: List[str]
