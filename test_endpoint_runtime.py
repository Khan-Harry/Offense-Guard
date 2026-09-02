import sys
sys.stdout.reconfigure(encoding='utf-8')
from app import app
import json

with app.test_client() as client:
    res = client.post('/api/runtime_check', json={'text': 'tu gadha hai bilkul'})
    data = res.get_json()
    print('Status:', res.status_code)
    print('is_offensive:', data['is_offensive'])
    print('flagged_word:', data['flagged_word'])
    print('models_scores keys count:', len(data['models_scores']))
    for k, v in data['models_scores'].items():
        print(f"  - {v['name']}: {v['prediction']} ({v['confidence']*100:.1f}%)")
