import sys
sys.stdout.reconfigure(encoding='utf-8')
from app import predict_text_multi_model

test_phrases = [
    'tum sher ki tarah bahadur ho',
    'tum sher ho',
    'mera sher beta',
    'kutta ik acha janwar hai',
    'meri billi ko kutte ne maar diya',
    'tu gadha hai bilkul',
    'tum aik kutte aur kameenay insan ho',
    'teri aisi ki taisi jahil'
]

print('='*85)
for phrase in test_phrases:
    res = predict_text_multi_model(phrase)
    print(f"\nSentence : \"{phrase}\"")
    print(f"Overall  : {res['prediction'].upper()} (Confidence: {res['confidence']*100:.1f}%)")
    print(f"Engine   : {res['model_used']}")
    print("Models Breakdown (All 5):")
    for k, v in res['models_scores'].items():
        icon = '⚠️' if v['prediction'] == 'offensive' else '✅'
        print(f"  {icon} {v['name']:25}: {v['prediction']:14} (Confidence: {v['confidence']*100:.1f}%)")
print('='*85)
