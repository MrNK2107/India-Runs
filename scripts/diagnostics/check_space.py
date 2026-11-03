import sys
import time

import requests

for i in range(12):
    try:
        r = requests.get('https://mrphantom07-redrob-ranker.hf.space/api/v1/health', timeout=10)
        data = r.json()
        idx = data.get('index_size', 0)
        embed = data.get('models_loaded', {}).get('embedding', False)
        ce = data.get('models_loaded', {}).get('cross_encoder', False)
        print(f'Check {i+1}: index_size={idx}, embed={embed}, ce={ce}, status={data.get("status")}')
        if idx > 0 and embed:
            print('=== READY ===')
            sys.exit(0)
    except Exception as e:
        print(f'Check {i+1}: error={e}')
    time.sleep(30)
print('Not ready after 6 minutes')
