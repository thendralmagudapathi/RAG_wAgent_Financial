from agent import choose_companies_llm
contexts = {
    'DEVVSTREAM_CORP': 'DevvStream provides sustainability software and carbon market services. They focus on greenhouse gas reporting and market infrastructure.',
    'ncr-20250930': 'NCR Voyix provides digital commerce and POS software, microservices, and payment solutions for retail and hospitality.'
}
q = "Compare the market focus of DEVVSTREAM CORP and NCR VOYIX CORPORATION"
print('Question:', q)
print('LLM choice:')
print(choose_companies_llm(q, contexts, model='mistral'))
