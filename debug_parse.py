from llm import call_ollama
import re
prompt = '''You are an assistant that must decide which of the given companies a user question is about.
Respond on the first line with a single tag: CHOICE: <company name>|BOTH|NEITHER
Then on following lines give a 1-2 sentence reason. Do not output any other text.

Companies and short contexts:
[DEVVSTREAM_CORP]:
- DevvStream provides sustainability software and carbon market services. They focus on greenhouse gas reporting and market infrastructure.

[ncr-20250930]:
- NCR Voyix provides digital commerce and POS software, microservices, and payment solutions for retail and hospitality.

Question:
Compare the market focus of DEVVSTREAM CORP and NCR VOYIX CORPORATION
'''
resp = call_ollama(prompt, model='mistral')
print('RAW RESP:\n', resp)
first_line = resp.splitlines()[0].strip()
print('FIRST LINE:', repr(first_line))
if first_line.startswith('CHOICE:'):
    val = first_line[len('CHOICE:'):].strip()
    parts = [p.strip() for p in re.split(r'[,\\|]', val) if p.strip()]
    print('PARSED PARTS:', parts)
    up = [p.upper() for p in parts]
    print('UP:', up)
    if 'BOTH' in up:
        print('DECISION: BOTH')
    elif 'NEITHER' in up:
        tail = '\n'.join(resp.splitlines()[1:]).lower()
        if 'both' in tail or 'compare' in tail or any(c.lower() in tail for c in ['DEVVSTREAM_CORP','ncr-20250930']):
            print('DECISION (tail): BOTH')
        else:
            print('DECISION: NEITHER')
else:
    print('FIRST LINE did not start with CHOICE:')
