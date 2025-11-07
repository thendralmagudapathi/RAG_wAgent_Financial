from llm import call_ollama
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
print('Prompt sent:')
print(prompt)
print('---response---')
resp = call_ollama(prompt, model='mistral')
print(resp)
