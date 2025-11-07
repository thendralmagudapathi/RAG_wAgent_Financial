#!/usr/bin/env python3
# builder (parses data_input/*.htm) and retriever (company detection + search).

#!/usr/bin/env python3
# builder (parses data_input/*.htm) and retriever (company detection + search).

import os
import glob
from parser import extract_text_from_html, guess_company_name, chunk_text
from embedder import Embedder
from vector_store import CompanyIndex
import numpy as np

class RAGBuilder:
    def __init__(self, model_name='all-MiniLM-L6-v2', store_dir='vector_store'):
        self.embedder = Embedder(model_name)
        self.store_dir = store_dir

    def build_from_dir(self, data_dir: str):
        files = glob.glob(os.path.join(data_dir, '*.htm*'))
        indexes = {}
        for f in files:
            raw_html = open(f, 'r', encoding='utf-8', errors='ignore').read()
            company = guess_company_name(f, raw_html)
            text = extract_text_from_html(f)
            chunks = chunk_text(text, max_chars=1000)
            if not chunks:
                continue
            if company not in indexes:
                indexes[company] = CompanyIndex(self.embedder.dim(), company, self.store_dir)
            emb = self.embedder.embed(chunks)
            metadatas = [{'company': company, 'text': c, 'source': f, 'chunk_id': i} for i, c in enumerate(chunks)]
            indexes[company].add(emb.astype('float32'), metadatas)
        # save all
        for c, idx in indexes.items():
            idx.save()
        return list(indexes.keys())

class RAGRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2', store_dir='vector_store'):
        self.embedder = Embedder(model_name)
        self.store_dir = store_dir
        # load all company indices available
        self.company_indices = {}
        # autodetect meta files
        for p in glob.glob(os.path.join(store_dir, '*.meta.pkl')):
            name = os.path.basename(p).replace('.meta.pkl', '')
            try:
                idx = CompanyIndex.load(name, self.embedder.dim(), store_dir)
                self.company_indices[name] = idx
            except Exception:
                pass

    def detect_company(self, question: str):
        qvec = self.embedder.embed(question)
        # compute similarity to company centroids
        best = None
        best_score = -1.0
        for name, idx in self.company_indices.items():
            c = idx.centroid()
            if c is None:
                continue
            score = float(np.dot(qvec, c))
            if score > best_score:
                best_score = score
                best = name
        return best, best_score

    def retrieve(self, question: str, top_k: int = 5):
        company, score = self.detect_company(question)
        if company is None:
            return None, []
        qvec = self.embedder.embed(question)
        results = self.company_indices[company].search(qvec, top_k=top_k)
        return company, results

if __name__ == '__main__':
    b = RAGBuilder()
    print('Use RAGBuilder.build_from_dir to create indexes.')
import glob
from parser import extract_text_from_html, guess_company_name, chunk_text
from embedder import Embedder
from vector_store import CompanyIndex
import numpy as np

class RAGBuilder:
    def __init__(self, model_name='all-MiniLM-L6-v2', store_dir='vector_store'):
        self.embedder = Embedder(model_name)
        self.store_dir = store_dir

    def build_from_dir(self, data_dir: str):
        files = glob.glob(os.path.join(data_dir, '*.htm*'))
        indexes = {}
        for f in files:
            raw_html = open(f, 'r', encoding='utf-8', errors='ignore').read()
            company = guess_company_name(f, raw_html)
            text = extract_text_from_html(f)
            chunks = chunk_text(text, max_chars=1000)
            if not chunks:
                continue
            if company not in indexes:
                indexes[company] = CompanyIndex(self.embedder.dim(), company, self.store_dir)
            emb = self.embedder.embed(chunks)
            metadatas = [{'company': company, 'text': c, 'source': f, 'chunk_id': i} for i, c in enumerate(chunks)]
            indexes[company].add(emb.astype('float32'), metadatas)
        # save all
        for c, idx in indexes.items():
            idx.save()
        return list(indexes.keys())

class RAGRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2', store_dir='vector_store'):
        self.embedder = Embedder(model_name)
        self.store_dir = store_dir
        # load all company indices available
        self.company_indices = {}
        # autodetect meta files
        for p in glob.glob(os.path.join(store_dir, '*.meta.pkl')):
            name = os.path.basename(p).replace('.meta.pkl', '')
            try:
                idx = CompanyIndex.load(name, self.embedder.dim(), store_dir)
                self.company_indices[name] = idx
            except Exception:
                pass

    def detect_company(self, question: str):
        qvec = self.embedder.embed(question)
        # compute similarity to company centroids
        best = None
        best_score = -1.0
        for name, idx in self.company_indices.items():
            c = idx.centroid()
            if c is None:
                continue
            score = float(np.dot(qvec, c))
            if score > best_score:
                best_score = score
                best = name
        return best, best_score

    def retrieve(self, question: str, top_k: int = 5):
        company, score = self.detect_company(question)
        if company is None:
            return None, []
        qvec = self.embedder.embed(question)
        results = self.company_indices[company].search(qvec, top_k=top_k)
        return company, results

if __name__ == '__main__':
    b = RAGBuilder()
    print('Use RAGBuilder.build_from_dir to create indexes.')
#builder (parses data_input/.htm) and retriever (company detection + search).

import os
import glob
from .parser import extract_text_from_html, guess_company_name, chunk_text
from .embedder import Embedder
from .vector_store import CompanyIndex
import numpy as np

class RAGBuilder:
    def __init__(self, model_name='all-MiniLM-L6-v2', store_dir='vector_store'):
        self.embedder = Embedder(model_name)
        self.store_dir = store_dir

    def build_from_dir(self, data_dir: str):
        files = glob.glob(os.path.join(data_dir, '*.htm*'))
        indexes = {}
        for f in files:
            raw_html = open(f, 'r', encoding='utf-8', errors='ignore').read()
            company = guess_company_name(f, raw_html)
            text = extract_text_from_html(f)
            chunks = chunk_text(text, max_chars=1000)
            if not chunks:
                continue
            if company not in indexes:
                indexes[company] = CompanyIndex(self.embedder.dim(), company, self.store_dir)
            emb = self.embedder.embed(chunks)
            metadatas = [{'company': company, 'text': c, 'source': f, 'chunk_id': i} for i, c in enumerate(chunks)]
            indexes[company].add(emb.astype('float32'), metadatas)
        # save all
        for c, idx in indexes.items():
            idx.save()
        return list(indexes.keys())

class RAGRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2', store_dir='vector_store'):
        self.embedder = Embedder(model_name)
        self.store_dir = store_dir
        # load all company indices available
        self.company_indices = {}
        # autodetect meta files
        for p in glob.glob(os.path.join(store_dir, '*.meta.pkl')):
            name = os.path.basename(p).replace('.meta.pkl', '')
            try:
                idx = CompanyIndex.load(name, self.embedder.dim(), store_dir)
                self.company_indices[name] = idx
            except Exception:
                pass

    def detect_company(self, question: str):
        qvec = self.embedder.embed(question)
        # compute similarity to company centroids
        best = None
        best_score = -1.0
        for name, idx in self.company_indices.items():
            c = idx.centroid()
            if c is None:
                continue
            score = float(np.dot(qvec, c))
            if score > best_score:
                best_score = score
                best = name
        return best, best_score

    def retrieve(self, question: str, top_k: int = 5):
        company, score = self.detect_company(question)
        if company is None:
            return None, []
        qvec = self.embedder.embed(question)
        results = self.company_indices[company].search(qvec, top_k=top_k)
        return company, results

if __name__ == '__main__':
    b = RAGBuilder()
    print('Use RAGBuilder.build_from_dir to create indexes.')