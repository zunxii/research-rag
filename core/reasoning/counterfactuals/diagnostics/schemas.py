from dataclasses import dataclass

@dataclass
class HypothesisScore:
    diagnosis: str
    retention_score: float
    modality_dependency: str
    base_support: float
    final_score: float
