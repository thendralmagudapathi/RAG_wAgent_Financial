#!/usr/bin/env python3
"""Agent to route queries to company data using the local LLM (Ollama).

This module asks the LLM to decide which company (or both/neither) a question pertains to,
based on short retrieved contexts from each company. It returns a list of selected company keys.
"""

from typing import Dict, List
import re
from llm import call_ollama


def choose_companies_llm(question: str, contexts: Dict[str, List[str]], model: str = 'mistral') -> List[str]:
    """Ask the LLM to choose which companies the question is about.

    contexts: mapping company_name -> list of short context strings (already retrieved)
    Returns a list of company names to use (one or more), or empty list if none.
    """
    # Build a compact prompt that lists companies and a few context snippets for each.
    lines = [
        "You are an assistant that must decide which of the given companies a user question is about.",
        "Respond on the first line with a single tag: CHOICE: <company name>|BOTH|NEITHER",
        "Then on following lines give a 1-2 sentence reason. Do not output any other text.",
        "",
        "Companies and short contexts:",
    ]
    for cname, snippets in contexts.items():
        lines.append(f"[{cname}]:")
        for s in snippets[:3]:
            # keep snippets short
            short = s.replace('\n', ' ').strip()
            if len(short) > 300:
                short = short[:297] + '...'
            lines.append(f"- {short}")
        lines.append("")

    lines.append("Question:")
    lines.append(question)
    prompt = "\n".join(lines)

    resp = call_ollama(prompt, model=model)
    if not resp:
        return []
    # parse first line
    first_line = resp.splitlines()[0].strip()
    if first_line.startswith('CHOICE:'):
        val = first_line[len('CHOICE:'):].strip()
        # handle common separators (comma or pipe) and tolerant values
        parts = [p.strip() for p in re.split(r'[,\|]', val) if p.strip()]
        up = [p.upper() for p in parts]
        if 'BOTH' in up:
            return list(contexts.keys())
        if 'NEITHER' in up:
            # If the LLM nevertheless discusses both companies in the following text,
            # treat as BOTH (models sometimes output NEITHER then compare both).
            tail = '\n'.join(resp.splitlines()[1:]).lower()
            if 'both' in tail or 'compare' in tail or any(c.lower() in tail for c in contexts.keys()):
                return list(contexts.keys())
            return []
        # maybe comma/pipe-separated names
        # validate against provided contexts
        out = [p for p in parts if p in contexts]
        # if none matched, attempt case-insensitive matching
        if not out:
            lowered = {k.lower(): k for k in contexts.keys()}
            for p in parts:
                lp = p.lower()
                if lp in lowered:
                    out.append(lowered[lp])
        return out
    # fallback: simple heuristic — if model returned a company name anywhere, match it
    text = resp.lower()
    selected = [c for c in contexts.keys() if c.lower() in text]
    # if contains words like both or compare, return all
    if 'both' in text or 'compare' in text or 'comparison' in text:
        return list(contexts.keys())
    return selected
