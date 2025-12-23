# core/kb/index.py
import faiss
import numpy as np

class KBIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray):
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def save(self, path: str):
        faiss.write_index(self.index, str(path))
