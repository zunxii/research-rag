# core/reasoning/constraints/extractor.py
from .cluster_distribution import cluster_distribution_constraint
from .modality_consistency import modality_consistency_constraint
from .boundary_analysis import boundary_analysis_constraint
from .evidence_diversity import evidence_diversity_constraint
from .distribution_check import distribution_check_constraint

class ConstraintExtractor:
    def extract(
        self,
        retrieved_metadata: list,
        img_emb,
        txt_emb,
        centroid_distances: dict,
        query_distance: float,
        percentile_95: float
    ):
        return {
            "cluster_distribution": cluster_distribution_constraint(
                retrieved_metadata
            ),
            "modality_consistency": modality_consistency_constraint(
                img_emb, txt_emb
            ),
            "boundary_analysis": boundary_analysis_constraint(
                centroid_distances
            ),
            "evidence_diversity": evidence_diversity_constraint(
                retrieved_metadata
            ),
            "distribution_check": distribution_check_constraint(
                query_distance, percentile_95
            ),
        }
