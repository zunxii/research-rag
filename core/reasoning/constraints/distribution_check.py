# core/reasoning/constraints/distribution_check.py
import numpy as np
from .base import Constraint

def distribution_check_constraint(
    query_distance: float,
    percentile_95: float
) -> Constraint:
    return {
        "in_distribution": query_distance <= percentile_95,
        "distance": round(query_distance, 4),
        "threshold": round(percentile_95, 4),
    }
