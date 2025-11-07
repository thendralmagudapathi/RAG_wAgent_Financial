#!/usr/bin/env python3
# per-company FAISS index manager + metadata save/load.

import os
import faiss
import numpy as np
import pickle

class CompanyIndex:
    """Per-company FAISS index plus metadata storage."""
    def __init__(self, dim: int, company_name: str, store_dir: str = 'vector_store'):
        self.dim = dim
        self.company = company_name.replace(' ', '_')
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        # index will be IndexFlatIP since we use normalized vectors for cosine
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []  # list of dicts per vector id

    def add(self, vectors: np.ndarray, metadatas: list):
        assert vectors.shape[1] == self.dim
        self.index.add(vectors.astype('float32'))
        self.metadata.extend(metadatas)

    def save(self):
        idx_path = os.path.join(self.store_dir, f'{self.company}.index')
        meta_path = os.path.join(self.store_dir, f'{self.company}.meta.pkl')
        faiss.write_index(self.index, idx_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, company_name: str, dim: int, store_dir: str = 'vector_store'):
        inst = cls(dim, company_name, store_dir)
        idx_path = os.path.join(store_dir, f'{inst.company}.index')
        meta_path = os.path.join(store_dir, f'{inst.company}.meta.pkl')
        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            raise FileNotFoundError('Index or metadata not found for ' + company_name)
        inst.index = faiss.read_index(idx_path)
        with open(meta_path, 'rb') as f:
            inst.metadata = pickle.load(f)
        return inst

    def search(self, qvec: np.ndarray, top_k: int = 5):
        # qvec shape (dim,), needs to be (1,dim)
        q = qvec.reshape(1, -1).astype('float32')
        scores, ids = self.index.search(q, top_k)
        ids = ids[0].tolist()
        scores = scores[0].tolist()
        results = []
        for i, s in zip(ids, scores):
            if i < 0 or i >= len(self.metadata):
                continue
            m = dict(self.metadata[i])
            m['_score'] = float(s)
            results.append(m)
        return results

    def centroid(self):
        # compute centroid of all vectors in index
        if self.index.ntotal == 0:
            return None
        # read vectors from index
        vecs = np.zeros((self.index.ntotal, self.dim), dtype='float32')
        self.index.reconstruct_n(0, self.index.ntotal, vecs)
        c = vecs.mean(axis=0)
        # normalize
        norm = np.linalg.norm(c)
        if norm == 0:
            return c
        return c / norm

    def __len__(self):
        return self.index.ntotal
#per-company FAISS index manager + metadata save/load.

import os
import faiss
import numpy as np
import pickle

class CompanyIndex:
    """Per-company FAISS index plus metadata storage."""
    def __init__(self, dim: int, company_name: str, store_dir: str = 'vector_store'):
        self.dim = dim
        self.company = company_name.replace(' ', '_')
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        # index will be IndexFlatIP since we use normalized vectors for cosine
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []  # list of dicts per vector id

    def add(self, vectors: np.ndarray, metadatas: list):
        assert vectors.shape[1] == self.dim
        self.index.add(vectors.astype('float32'))
        self.metadata.extend(metadatas)

    def save(self):
        idx_path = os.path.join(self.store_dir, f'{self.company}.index')
        meta_path = os.path.join(self.store_dir, f'{self.company}.meta.pkl')
        faiss.write_index(self.index, idx_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, company_name: str, dim: int, store_dir: str = 'vector_store'):
        inst = cls(dim, company_name, store_dir)
        idx_path = os.path.join(store_dir, f'{inst.company}.index')
        meta_path = os.path.join(store_dir, f'{inst.company}.meta.pkl')
        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            raise FileNotFoundError('Index or metadata not found for ' + company_name)
        inst.index = faiss.read_index(idx_path)
        with open(meta_path, 'rb') as f:
            inst.metadata = pickle.load(f)
        return inst

    def search(self, qvec: np.ndarray, top_k: int = 5):
        # qvec shape (dim,), needs to be (1,dim)
        q = qvec.reshape(1, -1).astype('float32')
        scores, ids = self.index.search(q, top_k)
        ids = ids[0].tolist()
        scores = scores[0].tolist()
        results = []
        for i, s in zip(ids, scores):
            if i < 0 or i >= len(self.metadata):
                continue
            m = dict(self.metadata[i])
            m['_score'] = float(s)
            results.append(m)
        return results

    def centroid(self):
        # compute centroid of all vectors in index
        if self.index.ntotal == 0:
            return None
        # read vectors from index
        vecs = np.zeros((self.index.ntotal, self.dim), dtype='float32')
        self.index.reconstruct_n(0, self.index.ntotal, vecs)
        c = vecs.mean(axis=0)
        # normalize
        norm = np.linalg.norm(c)
        if norm == 0:
            return c
        return c / norm

    def __len__(self):
        return self.index.ntotal
