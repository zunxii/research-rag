import faiss
import numpy as np

class StabilityRetriever:
    def __init__(self, index, metadata):
        self.index = index
        self.metadata = metadata

    def retrieve(self, emb, top_k=10):
        q = emb.cpu().numpy().astype("float32")
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)

        return [
            {
                "diagnosis_label": self.metadata[i]["diagnosis_label"],
                "score": float(s),
            }
            for s, i in zip(scores[0], idxs[0])
        ]
