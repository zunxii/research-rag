import faiss
import json
import numpy as np
from pathlib import Path


class KBRetriever:
    def __init__(self, kb_dir: str):
        kb_dir = Path(kb_dir)

        self.embeddings = np.load(kb_dir / "embeddings.npy")
        self.index = faiss.read_index(str(kb_dir / "index.faiss"))

        with open(kb_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        assert self.index.ntotal == len(self.metadata)

    def search(self, query_embedding: np.ndarray, top_k: int):
        query_embedding = query_embedding.astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)
        return scores[0], indices[0]
