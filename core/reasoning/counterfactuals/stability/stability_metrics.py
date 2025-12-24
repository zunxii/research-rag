from .base import normalize, js_divergence

def stability_report(baseline: dict, variants: dict):
    base = normalize(baseline["distribution"])

    divergences = {
        k: round(js_divergence(base, normalize(v["distribution"])), 4)
        for k, v in variants.items()
    }

    max_div = max(divergences.values())

    return {
        "js_divergence": divergences,
        "robustness_level": (
            "high" if max_div < 0.15
            else "medium" if max_div < 0.3
            else "low"
        ),
    }
