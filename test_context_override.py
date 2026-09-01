import sys
sys.stdout.reconfigure(encoding='utf-8')
from app import predict_text_multi_model

test_cases = [
    # Should be NON-OFFENSIVE (animal in positive context)
    ('kutta aik boht wafadar janwar hai', 'NON-OFFENSIVE'),
    ('meri billi boht pyari hai', 'NON-OFFENSIVE'),
    ('ghadha boht madadgar janwar hota hai', 'NON-OFFENSIVE'),
    ('Assalam o Alaikum, aap kaise hain?', 'NON-OFFENSIVE'),
    ('Pakistan zindabad, hum sab aik hain', 'NON-OFFENSIVE'),
    # Should be OFFENSIVE (person-directed insults)
    ('tum aik kutte aur kameenay insan ho', 'OFFENSIVE'),
    ('teri aisi ki taisi jahil', 'OFFENSIVE'),
    ('yeh banda kuttay se bura hai', 'OFFENSIVE'),
    ('tu ghadha hai bilkul', 'OFFENSIVE'),
    ('teri maa ki', 'OFFENSIVE'),
]

print('='*70)
print(f"{'Input Text':<40} {'Expected':<14} {'Got':<14} {'Match'}")
print('='*70)
correct = 0
for text, expected in test_cases:
    res = predict_text_multi_model(text)
    got = res['prediction'].upper().replace('-', '_')
    expected_norm = expected.replace('-', '_')
    match = '✅' if got == expected_norm else '❌'
    if got == expected_norm:
        correct += 1
    short_text = text[:38] + '..' if len(text) > 38 else text
    override = ' [Override]' if res.get('context_override') else ''
    print(f"{short_text:<40} {expected:<14} {res['prediction'].upper():<14} {match}{override}")

print('='*70)
print(f"Accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.0f}%)")
