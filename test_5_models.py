import sys
sys.stdout.reconfigure(encoding='utf-8')
from app import predict_text_multi_model

r = predict_text_multi_model('tu gadha hai bilkul')
print('Prediction:', r['prediction'])
print('Engine Used:', r['model_used'])
print('Models count in dict:', len(r['models_scores']))
for k, v in r['models_scores'].items():
    print(f"  * {k:14} -> {v['name']:25}: {v['prediction']:14} ({v['confidence']*100:.1f}%)")
