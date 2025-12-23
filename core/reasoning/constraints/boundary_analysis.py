# core/reasoning/constraints/boundary_analysis.py
import numpy as np
from .base import Constraint

def boundary_analysis_constraint(
    centroid_distances: dict
) -> Constraint:
    # centroid_distances = {label: distance}
    sorted_items = sorted(centroid_distances.items(), key=lambda x: x[1])

    if len(sorted_items) < 2:
        return {"near_boundary": False, "margin": None}

    d1 = sorted_items[0][1]
    d2 = sorted_items[1][1]
    margin = abs(d2 - d1)

    return {
        "near_boundary": margin < 0.15,
        "margin": round(margin, 4),
        "top_labels": [sorted_items[0][0], sorted_items[1][0]],
    }
