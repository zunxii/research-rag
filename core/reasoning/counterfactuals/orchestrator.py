class CounterfactualOrchestrator:
    def __init__(self, stability_runner, scorer):
        self.stability_runner = stability_runner
        self.scorer = scorer

    def run(self, img_emb, txt_emb):
        stability = self.stability_runner.run(img_emb, txt_emb)
        scores = self.scorer.score(stability)

        return {
            "stability": stability,
            "scores": scores,
        }
