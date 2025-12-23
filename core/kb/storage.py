# core/kb/storage.py
import json
import numpy as np
from pathlib import Path

class KBStorage:
    def save_embeddings(self, embeddings, path):
        np.save(path, embeddings)

    def save_metadata(self, metadata, path):
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
