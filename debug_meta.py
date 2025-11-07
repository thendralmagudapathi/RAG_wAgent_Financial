import pickle, os
for p in ['vector_store/DEVVSTREAM_CORP.meta.pkl','vector_store/ncr-20250930.meta.pkl']:
    if os.path.exists(p):
        with open(p,'rb') as f:
            meta = pickle.load(f)
        print(p, 'count=', len(meta))
        if meta:
            print('first keys=', list(meta[0].keys()))
            print('start_line=', meta[0].get('start_line'))
            print('text_preview=', repr(meta[0].get('text','')[:200]))
    else:
        print(p, 'MISSING')
