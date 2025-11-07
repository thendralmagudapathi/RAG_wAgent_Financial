#!/usr/bin/env python3
# simple Ollama CLI wrapper.

import subprocess
import json
import sys
from typing import Callable, Iterator

def call_ollama(prompt: str, model: str = 'mistral', max_tokens: int = 512):
    """
    Call Ollama CLI to generate a response using `ollama run MODEL [PROMPT]`.
    Returns the raw stdout or a best-effort extracted text field.
    """
    cmd = ['ollama', 'run', model, prompt]
    try:
        # force UTF-8 decoding and replace errors to avoid UnicodeDecodeError on Windows
        completed = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', check=False)
    except FileNotFoundError:
        raise RuntimeError('Ollama CLI not found. Please install Ollama from https://ollama.com')
    if completed.returncode != 0:
        raise RuntimeError(f'Ollama error: {completed.stderr.strip()}')
    out = completed.stdout.strip()
    try:
        j = json.loads(out)
        if isinstance(j, dict):
            for key in ('text', 'output', 'response'):
                if key in j and isinstance(j[key], str):
                    return j[key]
        return out
    except Exception:
        return out


def stream_ollama(prompt: str, model: str = 'mistral', temperature: float = None, verbose: bool = False) -> Iterator[str]:
    """Stream Ollama output line-by-line as an iterator of text chunks.

    Only pass --temperature if model is mistral and temperature is not None.
    """
    cmd = ['ollama', 'run', model]
    # Do NOT pass --temperature for any model (Ollama CLI does not support it reliably)
    # The temperature slider is UI-only and ignored in backend
    # verbose flag is not standardized across Ollama versions; include if requested
    if verbose and _ollama_supports_flag('--verbose'):
        cmd += ['--verbose']
    cmd += [prompt]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError('Ollama CLI not found. Please install Ollama from https://ollama.com')
    assert proc.stdout is not None
    try:
        # Read stdout incrementally
        for raw in iter(proc.stdout.readline, b''):
            if not raw:
                break
            try:
                text = raw.decode('utf-8', errors='replace')
            except Exception:
                # fallback
                text = raw.decode('utf-8', errors='ignore')
            yield text
        proc.wait()
        # If there is any trailing stderr and process failed, surface it
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode('utf-8', errors='replace') if proc.stderr is not None else ''
            raise RuntimeError(f'Ollama error (stream): {stderr}')
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass


def list_ollama_models() -> list:
    """Return a list of model names available to the local Ollama CLI.

    Uses `ollama list` and parses the first column. Returns an empty list if the
    command is not available or no models are installed.
    """
    try:
        completed = subprocess.run(['ollama', 'list'], capture_output=True, text=True, encoding='utf-8', errors='replace', check=False)
    except FileNotFoundError:
        return []
    out = completed.stdout.strip()
    if not out:
        return []
    lines = [l for l in out.splitlines() if l.strip()]
    models = []
    # skip a header line if present by checking if it contains 'NAME' or 'MODEL'
    for ln in lines:
        parts = ln.split()
        if len(parts) == 0:
            continue
        # heuristic: first token that looks like a model name; skip header
        if parts[0].upper() in ('NAME', 'MODEL'):
            continue
        models.append(parts[0])
    return models


_OLLAMA_FLAG_CACHE = {}


def _ollama_supports_flag(flag: str) -> bool:
    """Return True if `ollama run --help` mentions the given flag.

    Results are cached in-process.
    """
    if flag in _OLLAMA_FLAG_CACHE:
        return _OLLAMA_FLAG_CACHE[flag]
    try:
        completed = subprocess.run(['ollama', 'run', '--help'], capture_output=True, text=True, encoding='utf-8', errors='replace', check=False)
    except FileNotFoundError:
        _OLLAMA_FLAG_CACHE[flag] = False
        return False
    out = (completed.stdout or '') + '\n' + (completed.stderr or '')
    ok = flag in out
    _OLLAMA_FLAG_CACHE[flag] = ok
    return ok
