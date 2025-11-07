#!/usr/bin/env python3
# main CLI to build indices and ask questions.

import argparse
import os
from retriever_clean import RAGBuilder, RAGRetriever
from llm import call_ollama
from agent import choose_companies_llm


def build(data_dir: str, store_dir: str, model: str):
    b = RAGBuilder(model_name=model, store_dir=store_dir)
    companies = b.build_from_dir(data_dir)
    print('Built indices for:', companies)


def ask(question: str, store_dir: str, model: str, ollama_model: str, k: int = 5, no_llm: bool = False):
    r = RAGRetriever(model_name=model, store_dir=store_dir)
    if not r.company_indices:
        print('No company indices found. Build indexes first.')
        return

    # Prepare short contexts per company (few snippet extracts)
    contexts = {}
    for cname, idx in r.company_indices.items():
        snippets = []
        for m in idx.metadata[:3]:
            snippets.append(m.get('text', ''))
        contexts[cname] = snippets if snippets else ['']

    # Have the local LLM decide which companies the question pertains to
    try:
        selected = choose_companies_llm(question, contexts, model=ollama_model)
    except Exception as e:
        print('Error during company selection with LLM:', e)
        selected = []

    if not selected:
        print('LLM selected no company (NEITHER). Showing top contexts for all companies:')
        selected = list(r.company_indices.keys())

    if no_llm:
        for cname in selected:
            print(f'--- Top passages for {cname} ---')
            qvec = r.embedder.embed(question)
            results = r.company_indices[cname].search(qvec, top_k=k)
            for res in results:
                print(f"[score={res['_score']:.3f}] {res['text'][:400]}... (source={res.get('source')})\n")
        return

    # If one company selected, retrieve and answer using its contexts
    if len(selected) == 1:
        cname = selected[0]
        qvec = r.embedder.embed(question)
        results = r.company_indices[cname].search(qvec, top_k=k)
        print(f'Question seems about company: {cname} (retrieved {len(results)} chunks)')
        context = '\n\n---\n\n'.join([f"[score={res['_score']:.3f}] {res['text']}" for res in results])
        prompt = f"Answer the user question based only on the following context from {cname}. If unknown, say you don't know.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer concisely."
    else:
        # Comparative: collect contexts per company and label them
        all_contexts = []
        for cname in selected:
            qvec = r.embedder.embed(question)
            results = r.company_indices[cname].search(qvec, top_k=k)
            chunk = '\n\n---\n\n'.join([f"[{cname} score={res['_score']:.3f}] {res['text']}" for res in results])
            all_contexts.append(chunk)
        context = '\n\n==== COMPANY SEPARATOR ====\n\n'.join(all_contexts)
        prompt = f"Answer the user question using the following contexts from multiple companies. Label which company each piece came from and base your answer only on this context. If the answer is unknown, say you don't know.\n\nCONTEXTS:\n{context}\n\nQUESTION:\n{question}\n\nAnswer concisely and make comparative points explicit."

    print('\n---PROMPT to Ollama---\n')
    try:
        resp = call_ollama(prompt, model=ollama_model)
        print('\n---LLM ANSWER---\n')
        print(resp)
    except Exception as e:
        print('Error calling Ollama:', e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--build', action='store_true', help='Build vector store from HTML files')
    p.add_argument('--data-dir', default='data_input', help='Directory with HTML files')
    p.add_argument('--store-dir', default='vector_store', help='Where to save vector indices')
    p.add_argument('--model', default='all-MiniLM-L6-v2', help='Sentence-transformers model')
    p.add_argument('--ask', type=str, help='Ask a question')
    p.add_argument('--ollama-model', default='mistral', help='Ollama model name to use for generation')
    p.add_argument('--top-k', type=int, default=5, help='Number of passages to retrieve')
    p.add_argument('--no-llm', action='store_true', help='Do retrieval only, do not call the LLM')
    args = p.parse_args()

    if args.build:
        build(args.data_dir, args.store_dir, args.model)
    elif args.ask:
        ask(args.ask, args.store_dir, args.model, args.ollama_model, k=args.top_k, no_llm=args.no_llm)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
