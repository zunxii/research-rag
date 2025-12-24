from typing import Dict

def evidence_diversity_constraint(retrieved_metadata: list) -> Dict:
    images = {m["image_path"] for m in retrieved_metadata}
    cases = {m["case_id"] for m in retrieved_metadata}

    if len(images) < 5:
        level = "low"
    elif len(images) < 12:
        level = "medium"
    else:
        level = "high"

    return {
        "level": level,
        "unique_images": int(len(images)),
        "unique_cases": int(len(cases)),
    }
