from typing import Dict

def boundary_analysis_constraint(centroid_distances: dict) -> Dict:
    """
    Checks whether query is near decision boundary between clusters.
    """

    if len(centroid_distances) < 2:
        return {
            "near_boundary": False,
            "margin": None,
            "top_labels": []
        }

    sorted_items = sorted(centroid_distances.items(), key=lambda x: float(x[1]))

    d1 = float(sorted_items[0][1])
    d2 = float(sorted_items[1][1])
    margin = abs(d2 - d1)

    return {
        "near_boundary": bool(margin < 0.15),
        "margin": round(margin, 4),
        "top_labels": [sorted_items[0][0], sorted_items[1][0]],
    }
