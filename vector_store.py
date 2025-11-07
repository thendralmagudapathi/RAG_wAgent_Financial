#!/usr/bin/env python3
# per-company FAISS index manager + metadata save/load.

import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Optional, Union

class CompanyIndex:
    """Per-company FAISS index plus metadata storage.
    
    This class manages a FAISS index for a single company's document chunks,
    along with associated metadata. Uses IndexFlatIP (exact inner product) 
    since we work with L2-normalized vectors for cosine similarity.
    """
    def __init__(self, dim: int, company_name: str, store_dir: str = 'vector_store'):
        """Initialize a new company index.
        
        Args:
            dim: Dimensionality of the vectors to store
            company_name: Name of the company (will be sanitized for filenames)
            store_dir: Directory to store index and metadata files
        """
        self.dim = dim
        self.company = company_name.replace(' ', '_')
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)
        # index will be IndexFlatIP since we use normalized vectors for cosine
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []  # list of dicts per vector id

    def add(self, vectors: np.ndarray, metadatas: List[Dict]) -> None:
        """Add vectors and their metadata to the index.
        
        Args:
            vectors: Array of shape (n, dim) containing L2-normalized vectors
            metadatas: List of metadata dicts, one per vector
        
        Raises:
            AssertionError: If vectors shape doesn't match expected dim
            ValueError: If lengths of vectors and metadata don't match
        """
        if len(vectors) != len(metadatas):
            raise ValueError("Number of vectors and metadata entries must match")
        assert vectors.shape[1] == self.dim, f"Expected dim={self.dim}, got {vectors.shape[1]}"
        self.index.add(vectors.astype('float32'))
        self.metadata.extend(metadatas)

    def save(self) -> None:
        """Save both the FAISS index and metadata to disk."""
        idx_path = os.path.join(self.store_dir, f'{self.company}.index')
        meta_path = os.path.join(self.store_dir, f'{self.company}.meta.pkl')
        faiss.write_index(self.index, idx_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, company_name: str, dim: int, store_dir: str = 'vector_store') -> 'CompanyIndex':
        """Load a company index from disk.
        
        Args:
            company_name: Name of the company to load
            dim: Expected vector dimensionality
            store_dir: Directory containing the stored files
        
        Returns:
            CompanyIndex: Loaded index instance
            
        Raises:
            FileNotFoundError: If index or metadata files don't exist
        """
        inst = cls(dim, company_name, store_dir)
        idx_path = os.path.join(store_dir, f'{inst.company}.index')
        meta_path = os.path.join(store_dir, f'{inst.company}.meta.pkl')
        if not os.path.exists(idx_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f'Index or metadata not found for {company_name} in {store_dir}')
        inst.index = faiss.read_index(idx_path)
        with open(meta_path, 'rb') as f:
            inst.metadata = pickle.load(f)
        return inst

    def search(self, qvec: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar vectors and return their metadata.
        
        Args:
            qvec: Query vector of shape (dim,)
            top_k: Number of results to return
        
        Returns:
            List of metadata dicts, each with added '_score' field
        """
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

    def centroid(self) -> Optional[np.ndarray]:
        """Compute the normalized centroid of all vectors in the index.
        
        Returns:
            L2-normalized centroid vector, or None if index is empty
        """
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

    def __len__(self) -> int:
        """Return number of vectors in the index."""
        return self.index.ntotal
