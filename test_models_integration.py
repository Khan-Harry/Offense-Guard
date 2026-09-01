import sys
sys.stdout.reconfigure(encoding='utf-8')
from app import predict_text_multi_model

test_sentences = [
    'kutta aik boht wafadar janwar hai',
    'tum aik kutte aur kameenay insan ho',
    'Assalam o Alaikum, aap kaise hain?',
    'teri aisi ki taisi jahil'
]

for s in test_sentences:
    res = predict_text_multi_model(s)
    print('='*60)
    print('Input Text   :', s)
    print('Final Verdict:', res['prediction'].upper(), f"({res['confidence']*100:.1f}%)")
    print('Engine       :', res['model_used'])
    print('All 5 Model Scores:')
    for m_key, m_val in res['models_scores'].items():
        print(f"   * {m_val['name']:24}: {m_val['prediction']:14} ({m_val['confidence']*100:.1f}%)")
