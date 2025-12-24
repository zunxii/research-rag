import json
import torch
from core.reasoning.counterfactuals.stability.runner import StabilityRunner
from core.reasoning.counterfactuals.stability.retrieval import CounterfactualRetriever
from core.reasoning.counterfactuals.diagnostics.scorer import CounterfactualScorer
from core.reasoning.counterfactuals.orchestrator import CounterfactualOrchestrator

# assume index, metadata, fusion already loaded
retriever = CounterfactualRetriever(index, metadata)
stability_runner = StabilityRunner(retriever, fusion)
scorer = CounterfactualScorer()

orchestrator = CounterfactualOrchestrator(stability_runner, scorer)

output = orchestrator.run(img_emb, txt_emb)

print(json.dumps({
    "stability": output["stability"],
    "scores": [s.__dict__ for s in output["scores"]],
}, indent=2))
