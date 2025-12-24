import torch
from .perturbations import remove_text, remove_image, add_noise
from .distribution import cluster_distribution
from .stability_metrics import stability_report

class StabilityRunner:
    def __init__(self, retriever, fusion):
        self.retriever = retriever
        self.fusion = fusion.eval()

    def _run(self, img_emb, txt_emb):
        with torch.no_grad():
            fused = self.fusion(img_emb, txt_emb)
        retrieved = self.retriever.retrieve(fused)
        return cluster_distribution(retrieved)

    def run(self, img_emb, txt_emb):
        baseline = self._run(img_emb, txt_emb)

        i, t = remove_text(img_emb, txt_emb)
        no_text = self._run(i, t)

        i, t = remove_image(img_emb, txt_emb)
        no_image = self._run(i, t)

        noisy = self._run(add_noise(img_emb), txt_emb)

        stability = stability_report(
            baseline,
            {
                "no_text": no_text,
                "no_image": no_image,
                "noisy": noisy,
            }
        )

        return {
            "baseline": baseline,
            "no_text": no_text,
            "no_image": no_image,
            "noisy": noisy,
            "stability": stability,
        }
