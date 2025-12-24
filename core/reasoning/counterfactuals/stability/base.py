import numpy as np

def normalize(dist: dict) -> dict:
    total = sum(dist.values()) + 1e-9
    return {k: v / total for k, v in dist.items()}

def js_divergence(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    p_vec = np.array([p.get(k, 0.0) for k in keys])
    q_vec = np.array([q.get(k, 0.0) for k in keys])

    m = 0.5 * (p_vec + q_vec)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log((a[mask] + 1e-9) / (b[mask] + 1e-9)))

    return float(0.5 * kl(p_vec, m) + 0.5 * kl(q_vec, m))
