from .evidence_retention import evidence_retention
from .modality_dependency import modality_dependency
from .ranking import stability_rank
from .schemas import HypothesisScore

class CounterfactualScorer:
    def score(self, stability_output: dict):
        baseline = stability_output["baseline"]
        no_text = stability_output["no_text"]
        no_image = stability_output["no_image"]
        noisy = stability_output["noisy"]

        retention = evidence_retention(baseline, no_text, no_image, noisy)
        modality = modality_dependency(no_text, no_image)

        ranked = stability_rank(retention, modality, baseline)

        return [
            HypothesisScore(
                diagnosis=diag,
                retention_score=retention.get(diag, 0.0),
                modality_dependency=modality.get(diag, "unstable"),
                base_support=baseline["distribution"].get(diag, 0.0),
                final_score=score,
            )
            for diag, score in ranked.items()
        ]
