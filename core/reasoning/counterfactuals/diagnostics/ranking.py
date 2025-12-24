def stability_rank(retention, modality, baseline):
    scores = {}

    for diag in baseline["distribution"]:
        base = baseline["distribution"][diag]
        retain = retention.get(diag, 0.0)

        modality_bonus = {
            "multimodal": 1.0,
            "image_dominant": 0.7,
            "text_dominant": 0.7,
            "unstable": 0.3,
        }.get(modality.get(diag, "unstable"), 0.3)

        scores[diag] = round(base * retain * modality_bonus, 4)

    return scores
