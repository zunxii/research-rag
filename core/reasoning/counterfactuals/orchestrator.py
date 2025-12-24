class CounterfactualReasoner:
    def __init__(self, stability_runner, scorer, explainer=None):
        self.stability_runner = stability_runner
        self.scorer = scorer
        self.explainer = explainer  # Gemini (optional)

    def run(self, img_emb, txt_emb):
        stability = self.stability_runner.run(img_emb, txt_emb)
        ranked = self.scorer.score(stability)

        output = {
            "stability": stability,
            "ranked_hypotheses": [h.__dict__ for h in ranked],
        }

        if self.explainer:
            explanation = self.explainer.explain(output)
            output["explanation"] = explanation.__dict__

        return output
