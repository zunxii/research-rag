def evidence_diversity_constraint(
    retrieved_metadata: list
):
    images = set(
        m["metadata"]["image_path"]
        for m in retrieved_metadata
        if "metadata" in m
    )

    cases = set(
        m["metadata"]["case_id"]
        for m in retrieved_metadata
        if "metadata" in m
    )

    if len(images) < 5:
        level = "low"
    elif len(images) < 12:
        level = "medium"
    else:
        level = "high"

    return {
        "level": level,
        "unique_images": len(images),
        "unique_cases": len(cases),
    }
