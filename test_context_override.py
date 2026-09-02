import sys
sys.stdout.reconfigure(encoding='utf-8')
from app import predict_text_multi_model

test_cases = [
    # Praise & Compliments (Tareef)
    ('tum sher ki tarah bahadur ho', 'NON-OFFENSIVE'),
    ('tum sher ki tarah larhte ho', 'NON-OFFENSIVE'),
    ('tum sher ki tarah larte ho', 'NON-OFFENSIVE'),
    ('mera sher beta hai', 'NON-OFFENSIVE'),
    ('tum cheetay ho yaar', 'NON-OFFENSIVE'),

    # Safe Animal & Zoological Contexts
    ('kutta ik acha janwar hai', 'NON-OFFENSIVE'),
    ('meri billi ko kutte ne maar diya', 'NON-OFFENSIVE'),
    ('kutta aik boht wafadar janwar hai', 'NON-OFFENSIVE'),
    ('Assalam o Alaikum, aap kaise hain?', 'NON-OFFENSIVE'),
    
    # Metaphorical Insults & Explicit Abuse (Must be OFFENSIVE)
    ('tum aik kutte aur kameenay insan ho', 'OFFENSIVE'),
    ('teri aisi ki taisi jahil', 'OFFENSIVE'),
    ('tu gadha hai bilkul', 'OFFENSIVE'),
    ('yeh banda kuttay se bura hai', 'OFFENSIVE'),
    ('teri maa ki', 'OFFENSIVE'),
]

print('='*75)
print(f"{'Input Text':<42} {'Expected':<15} {'Got':<15} {'Match'}")
print('='*75)
correct = 0
for text, expected in test_cases:
    res = predict_text_multi_model(text)
    got = res['prediction'].upper().replace('-', '_')
    expected_norm = expected.replace('-', '_')
    match = '✅' if got == expected_norm else '❌'
    if got == expected_norm:
        correct += 1
    short_text = text[:40] + '..' if len(text) > 40 else text
    print(f"{short_text:<42} {expected:<15} {res['prediction'].upper():<15} {match}")

print('='*75)
print(f"Final Accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")

# Also check 5 models breakdown on praise sample:
sample_res = predict_text_multi_model('tum sher ki tarah bahadur ho')
print('\nSample Models Scores Count:', len(sample_res['models_scores']))
for k, v in sample_res['models_scores'].items():
    print(f"  - {v['name']}: {v['prediction']} ({v['confidence']*100:.1f}%)")
