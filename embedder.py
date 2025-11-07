#!/usr/bin/env python3
# sentence-transformers wrapper (L2-normalized embeddings).

from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        """Return L2-normalized embeddings for a list of texts or a single text."""
        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True
        vecs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # normalize for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        vecs = vecs / norms
        return vecs[0] if single else vecs

    def dim(self):
        """Return the dimensionality of the embedding vectors."""
        # Using a simple text to get dimensionality is fine since it's constant
        v = self.model.encode(['hello'], convert_to_numpy=True)
        return v.shape[1]

if __name__ == '__main__':
    e = Embedder()
    print('dim=', e.dim())
