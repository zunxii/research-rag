def evidence_retention(baseline, no_text, no_image, noisy):
    all_dists = {
        "baseline": baseline,
        "no_text": no_text,
        "no_image": no_image,
        "noisy": noisy,
    }

    diagnoses = set()
    for d in all_dists.values():
        diagnoses |= set(d["distribution"].keys())

    scores = {}
    for diag in diagnoses:
        present = sum(
            diag in dist["distribution"]
            for dist in all_dists.values()
        )
        scores[diag] = present / len(all_dists)

    return scores
