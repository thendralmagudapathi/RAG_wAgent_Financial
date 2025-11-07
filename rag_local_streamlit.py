#!/usr/bin/env python3
"""Streamlit app for local RAG using Ollama and FAISS.

Features:
- Enter a question
- Shows retrieved chunks with provenance (source file and start_line)
- Streams the LLM answer as it's generated

Run: streamlit run rag_local_streamlit.py
"""

import streamlit as st
from retriever_clean import RAGRetriever
from agent import choose_companies_llm
from llm import stream_ollama, list_ollama_models

st.set_page_config(page_title='Local RAG', layout='wide')

st.title('Multi-Company Financial RAG System')

def pick_default_model(available_models: list[str]) -> str:
    """Pick a lightweight default model name from available models using heuristics."""
    if not available_models:
        return 'mistral'
    low = [m.lower() for m in available_models]
    # heuristics for light models: prefer names containing these substrings
    prefs = ['tiny', 'mini', 'small', 'lite', 'llama-mini', 'oasst-mini', 'alpaca']
    for p in prefs:
        for m, lm in zip(available_models, low):
            if p in lm:
                return m
    # fallback to first available
    return available_models[0]

available_models = list_ollama_models()
# Always add 'tinyllama' to the dropdown for fast CPU inference
# if 'tinyllama' not in [m.lower() for m in available_models]:
#     available_models.append('tinyllama')
model_default = pick_default_model(available_models)

with st.sidebar:
    if available_models:
        # present only the model short names for nicer display (strip tags after colon)
        display_models = [m.split(':')[0] for m in available_models]
        model_idx = available_models.index(model_default) if model_default in available_models else 0
        sel = st.selectbox('Choose a model', options=display_models, index=model_idx)
        # map selection back to full model identifier
        model = available_models[display_models.index(sel)]
    else:
        model = st.text_input('Ollama model', value=model_default, help='No local Ollama models detected; enter the model name you have installed or a light model name you prefer.')
    top_k = st.number_input('Top-k per company', min_value=1, max_value=20, value=4)
    # Disable temperature when the selected model is llama3 because that model
    # (or the Ollama runtime for it) doesn't accept the --temperature flag.
    temperature_disabled = 'llama3' in model.lower()
    if temperature_disabled:
        st.info('Temperature control disabled for the selected model (llama3).')
        # set deterministically to 0.0 when disabled
        temperature = 0.0
        # show a non-interactive slider for clarity (disabled param supported in Streamlit >=1.18)
        try:
            st.slider('Temperature (for mistral)', min_value=0.0, max_value=1.0, value=0.0, step=0.01, disabled=True)
        except TypeError:
            # older streamlit versions may not support `disabled`; just show text instead
            st.write('Temperature: 0.0 (disabled for this model)')
    else:
        temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=0.0, step=0.01, help='Sampling temperature for the LLM (0.0 = deterministic).')
    verbosity = st.selectbox('Verbosity', options=['Minimal', 'Normal', 'Debug'], index=1, help='Minimal: concise UI; Normal: show routing + retrieved; Debug: show prompt and raw model stream')
    store_dir = st.text_input('Vector store dir', value='vector_store')
    use_llm_routing = st.checkbox('Use LLM for company routing', value=True)
    if st.button('Reload indices'):
        st.session_state['retriever'] = RAGRetriever(store_dir=store_dir)

if 'retriever' not in st.session_state:
    st.session_state['retriever'] = RAGRetriever(store_dir=store_dir)

retriever: RAGRetriever = st.session_state['retriever']

q = st.text_input('Question')
if st.button('Ask') and q.strip():
    with st.spinner('Detecting company and retrieving...'):
        # Get short contexts per company for routing
        contexts = {}
        for name, idx in retriever.company_indices.items():
            # build a short snippet from first vector metadata
            if idx.metadata:
                contexts[name] = idx.metadata[0]['text'][:500]
            else:
                contexts[name] = ''
        if use_llm_routing:
            # routing currently does not accept temperature; keep deterministic routing
            choice = choose_companies_llm(q, contexts, model=model)
            st.write('Routing choice from LLM:', choice)
            selected = choice if isinstance(choice, list) else [choice]
        else:
            # fallback: use centroid-based detection
            cname, score = retriever.detect_company(q)
            selected = [cname] if cname else []
            st.write('Routing choice (centroid):', selected, 'score=', score)

        # Retrieve per selected company
        all_results = []
        for comp in selected:
            if comp not in retriever.company_indices:
                continue
            qvec = retriever.embedder.embed(q)
            res = retriever.company_indices[comp].search(qvec, top_k=top_k)
            for meta in res:
                score = meta.get('_score', 0.0)
                all_results.append((comp, score, meta))

    # Show retrieved chunks and provenance
    st.subheader('Retrieved chunks')
    for comp, score, meta in all_results:
        with st.expander(f"{comp} — {meta.get('source','')} (line {meta.get('start_line')}) — score={score:.3f}"):
            st.write(meta.get('text',''))

    st.subheader('Answer (streaming)')
    out_area = st.empty()
    out_text = ''
    show_debug = verbosity == 'Debug'
    if show_debug:
        st.markdown('**Prompt (debug):**')
        st.code('Building prompt...')
    prompt = f"Use the following retrieved snippets to answer the question:\nQuestion: {q}\n\n"
    for comp, score, meta in all_results:
        prompt += f"--- {comp} (source={meta.get('source')} start_line={meta.get('start_line')}) ---\n{meta.get('text')}\n\n"
    prompt += "Answer concisely and cite source files and line numbers where relevant."

    try:
        if show_debug:
            st.code(prompt)
        # stream with temperature
        for chunk in stream_ollama(prompt, model=model, temperature=temperature, verbose=(verbosity == 'Debug')):
            out_text += chunk
            # use f-string to inject the growing output into the HTML safely
            out_area.markdown(f"<div style='font-family:monospace; white-space:pre-wrap'>{out_text}</div>", unsafe_allow_html=True)
            # if debug, also show raw chunk as it arrives
            if show_debug:
                st.write('--- raw chunk ---')
                st.write(repr(chunk))
    except Exception as e:
        st.error(f"LLM streaming error: {e}")
