# RAG_wAgent

Lightweight RAG (Retrieval-Augmented Generation) demo using a local LLM (Ollama + tinyllama), a simple vector store, and a Streamlit UI. This repo includes helper debug scripts to exercise routing and model calls.
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/82a34331-100a-4513-af73-7346eb1a2230" />

## Contents
- `rag_local_streamlit.py` — Streamlit web UI and controls (model, temperature, top-k per company, routing, verbosity).
- `llm.py` — Wrapper for calling Ollama CLI / model.
- `vector_store.py` — Vector store utilities (indexing / retrieval).
- `agent.py` — Routing logic (LLM-based and centroid-based options).
- `debug_call.py`, `debug_agent.py`, `debug_meta.py`, `debug_parse.py` — Standalone debug scripts for testing components.
- `vector_store/` — Per-company vector metadata and index files (created by indexing routines).

---

## Setup (Windows / PowerShell)
1. Install Ollama and the required local models (follow Ollama's docs for your OS):
   - Install Ollama: https://ollama.com/docs
   - Pull the tinyllama model (example):

   ```powershell
   ollama pull tinyllama
   ```

   Confirm the model is available:

   ```powershell
   ollama list
   ollama status
   ```

2. Create and activate a Python virtual environment (recommended):

   ```powershell
   cd 'C:\Users\HP\Desktop\RAG_wAgent'
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install Python dependencies.
   - If the repo provides `requirements.txt`, install with:

   ```powershell
   pip install -r requirements.txt
   ```

   Add other packages as needed (faiss, sentence-transformers, scikit-learn, etc.) depending on your indexing/retrieval implementation.

4. (Optional) Ensure you have any indexing artifacts in `vector_store/`. If not, run the project's indexing script (if present) or the scripts that populate the vector store for each company.

---

## How to run
1. Start Ollama (if required by your Ollama install method).
2. Start the Streamlit UI (from the repo root, with the venv activated):

```powershell
# using the activated venv
python -m streamlit run .\rag_local_streamlit.py
```

3. Open your browser at http://localhost:8501.
4. Use the UI to pick a model (e.g., `tinyllama`), set `Top-K` (per-company), toggle routing mode (LLM-based vs centroid-based), and set verbosity.

Debug / manual tests (useful when the UI is slow or blocked):
- Test the LLM call directly:

```powershell
python .\debug_call.py
```

- Inspect vector store metadata:

```powershell
python .\debug_meta.py
```

- Test routing logic (LLM-based routing):

```powershell
python .\debug_parse.py
```

- Run `debug_agent.py` to exercise the full call path used by agent logic.

---

## Design approach (short)
This project implements a straightforward RAG flow with the following pieces:

- Ingestion / vector store: Documents are chunked per company and stored with metadata (in `vector_store/`). Each company has its own index/metadata file.

- Retrieval: For a query, the system retrieves the top-K chunks from each company's index independently ("top-K per company"), then either:
  - Combine the retrieved chunks and call the LLM to answer (centroid-based routing), or
  - Use a dedicated routing step where an LLM determines which company(s) the question is about, then only retrieve and answer using those companies' chunks (LLM-based routing).

- LLM calls: Performed via `llm.py` which wraps calls to Ollama (local model). Some models (e.g., `tinyllama`) have limited flag/temperature support; the UI disables incompatible controls for those models.

- Streaming UI: The Streamlit UI simulates word-by-word streaming for a better user experience (the code may intentionally yield words slowly to create a streaming effect). For debugging/performance tests, run the debug scripts that call the LLM directly (no UI streaming).

Design tradeoffs:
- Using tinyllama or other CPU-friendly models reduces memory and GPU requirements but can be slower on single-core CPU inference.
- Per-company top-K provides more balanced retrieval for multi-company queries but increases overall retrieval work.
- LLM-based routing can reduce LLM calls and retrieval bandwidth when routing is accurate; however, it adds an extra LLM invocation for the routing step.

---

## Example queries
- "Compare the market focus of DEVVSTREAM CORP and NCR VOYIX CORPORATION"
- "Is DEVVSTREAM CORP likely to provide carbon market infrastructure or retail POS solutions?"
- "Show me the most relevant policies for NCR VOYIX CORPORATION in Q3 2025 filings"
- "Which company is more likely to offer payment microservices?"

When testing, try:
- `Top-K = 1` or `2` (per-company) to reduce retrieval cost.
- `Verbosity = Minimal` to keep LLM prompt/response shorter.
- If `tinyllama` appears slow, temporarily switch to a faster model (if available) to isolate whether the bottleneck is the model or the retrieval flow.

---

## Troubleshooting tips
- Streamlit is slow or UI shows no progress:
  - Open the terminal where you launched Streamlit and look for errors or long blocking logs.
  - Run `debug_call.py` to ensure `llm.py` and Ollama respond quickly.
  - If CPU utilization drops and the app is still "generating", the app might be waiting on an external process or blocked by synchronous I/O (network, disk). Check Ollama status and model readiness.

- Ollama errors / model not found:
  - Re-run `ollama list` and `ollama pull tinyllama`.
  - Ensure the Ollama daemon/service is running; restart if needed.

- Routing produces unexpected results:
  - Run `python .\debug_parse.py` to see the raw routing decision and the prompt sent to the router LLM.

- If you change or recreate the vector store, restart the Streamlit app so it picks up the new files.

---

## Files of interest for debugging
- `llm.py` — check how the Ollama CLI is invoked and how flags (temperature, streaming) are managed.
- `agent.py` — routing and per-company retrieval decisions.
- `rag_local_streamlit.py` — the UI: model selection, streaming behaviour, and controls for top-k/verbosity.
- `debug_*.py` — quick entry points to test isolated functionality.
