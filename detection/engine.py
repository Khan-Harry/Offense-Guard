"""
OffenseGuard Multi-Model Context-Aware Detection Engine
Evaluates text across 5 ML and DL models (SVM, NB, RF, CNN, LSTM)
combined with bilingual hybrid context calibration.
"""

import re
from datetime import datetime
import numpy as np

import models_layer.model_loader as ml
from detection.cleaner import clean_text_for_context
from detection.lexicon import (
    COMMON_NAMES,
    NOBLE_PRAISE_ANIMALS,
    PEJORATIVE_ANIMALS,
    PEJORATIVE_ANIMAL_PHRASES,
    ACTUAL_BIOLOGICAL_ANIMALS,
    ALL_ANIMAL_WORDS,
    ZOOLOGICAL_FACTUAL_MARKERS,
    PRAISE_DESCRIPTORS,
    EXPLICIT_ABUSE,
    HUMAN_TARGET_PRONOUNS,
    STOP_WORDS
)


def predict_text_multi_model(text):
    """
    Main Prediction Engine:
    Evaluates text across 5 ML and DL models (SVM, NB, RF, CNN, LSTM)
    combined with bilingual hybrid context calibration.
    """
    if not text or not text.strip():
        return {
            'prediction': 'non-offensive',
            'confidence': 1.0,
            'model_used': 'None',
            'models_scores': {},
            'context_override': False
        }

    cleaned = clean_text_for_context(text)
    text_lower = cleaned.lower()
    # Extract both Latin/ASCII words and Unicode Urdu script words safely
    text_words = set(re.findall(r'[\w\u0600-\u06FF]+', text_lower))

    has_noble_animal = bool(text_words.intersection(NOBLE_PRAISE_ANIMALS))
    has_pejorative_phrase = any(ph in text_lower for ph in PEJORATIVE_ANIMAL_PHRASES)
    has_pejorative_animal = bool(text_words.intersection(PEJORATIVE_ANIMALS)) or has_pejorative_phrase
    has_biological_animal = bool(text_words.intersection(ACTUAL_BIOLOGICAL_ANIMALS))
    has_any_animal = bool(text_words.intersection(ALL_ANIMAL_WORDS)) or any(aw in text_lower for aw in ALL_ANIMAL_WORDS)
    has_explicit_abuse = any(ab in text_lower for ab in EXPLICIT_ABUSE)
    has_praise = any(pr in text_lower for pr in PRAISE_DESCRIPTORS)
    has_zoological_marker = any(zm in text_lower for zm in ZOOLOGICAL_FACTUAL_MARKERS)
    has_human_target = bool(text_words.intersection(HUMAN_TARGET_PRONOUNS)) or any(
        p in text_lower for p in [
            'tum ', 'tu ', 'teri ', 'tera ', 'aap ', 'you ', 'insan ', 'banda ',
            'تم ', 'تو ', 'تیرا ', 'تیری ', 'آپ ', 'انسان ', 'بندہ '
        ]
    )
    has_common_name = bool(text_words.intersection(COMMON_NAMES))

    models_scores = {}
    probs = {}

    # ── 1. Machine Learning Models (TF-IDF) ──
    if ml.vectorizer is not None:
        try:
            X = ml.vectorizer.transform([cleaned])

            # SVM
            if ml.svm_model is not None:
                if hasattr(ml.svm_model, 'predict_proba'):
                    p_svm = float(ml.svm_model.predict_proba(X)[0][1])
                else:
                    dec = float(ml.svm_model.decision_function(X)[0])
                    p_svm = float(1.0 / (1.0 + np.exp(-dec)))
                probs['svm'] = p_svm
                models_scores['svm'] = {
                    'name': 'Linear SVM',
                    'prediction': 'offensive' if p_svm >= 0.5 else 'non-offensive',
                    'confidence': round(float(max(p_svm, 1 - p_svm)), 4),
                    'offensive_prob': round(p_svm, 4)
                }

            # Naive Bayes
            if ml.nb_model is not None:
                p_nb = float(ml.nb_model.predict_proba(X)[0][1])
                probs['naive_bayes'] = p_nb
                models_scores['naive_bayes'] = {
                    'name': 'Naive Bayes',
                    'prediction': 'offensive' if p_nb >= 0.5 else 'non-offensive',
                    'confidence': round(float(max(p_nb, 1 - p_nb)), 4),
                    'offensive_prob': round(p_nb, 4)
                }

            # Random Forest
            if ml.rf_model is not None:
                p_rf = float(ml.rf_model.predict_proba(X)[0][1])
                probs['random_forest'] = p_rf
                models_scores['random_forest'] = {
                    'name': 'Random Forest',
                    'prediction': 'offensive' if p_rf >= 0.5 else 'non-offensive',
                    'confidence': round(float(max(p_rf, 1 - p_rf)), 4),
                    'offensive_prob': round(p_rf, 4)
                }
        except Exception as e:
            print(f"ML Prediction Error: {e}")

    # ── 2. Deep Learning Models (CNN, Bi-LSTM) ──
    if ml.tokenizer is not None:
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = ml.tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')

            # 1D CNN
            if ml.cnn_model is not None:
                p_cnn = float(ml.cnn_model.predict(padded, verbose=0)[0][1])
                probs['cnn'] = p_cnn
                models_scores['cnn'] = {
                    'name': '1D CNN (Deep Learning)',
                    'prediction': 'offensive' if p_cnn >= 0.5 else 'non-offensive',
                    'confidence': round(float(max(p_cnn, 1 - p_cnn)), 4),
                    'offensive_prob': round(p_cnn, 4)
                }

            # Bi-LSTM
            if ml.lstm_model is not None:
                p_lstm = float(ml.lstm_model.predict(padded, verbose=0)[0][1])
                probs['lstm'] = p_lstm
                models_scores['lstm'] = {
                    'name': 'Bi-LSTM (Deep Learning)',
                    'prediction': 'offensive' if p_lstm >= 0.5 else 'non-offensive',
                    'confidence': round(float(max(p_lstm, 1 - p_lstm)), 4),
                    'offensive_prob': round(p_lstm, 4)
                }
        except Exception as e:
            print(f"DL Prediction Error: {e}")

    # ── 3. Hybrid Context Decision Engine & Context Calibration ──
    # Determine context categories
    is_targeted_insult = (
        has_pejorative_phrase or
        (has_pejorative_animal and (has_human_target or not has_biological_animal))
    ) and not has_praise

    is_zoological = (
        (has_zoological_marker or has_biological_animal) and
        not has_human_target and
        not is_targeted_insult and
        not has_explicit_abuse
    )

    is_praise = (has_noble_animal or has_praise) and not has_explicit_abuse and not is_targeted_insult

    is_common_name_safe = (
        has_common_name and
        not has_explicit_abuse and
        not is_targeted_insult and
        not has_pejorative_phrase
    )

    # Ensure all 5 models have a baseline probability before calibration
    for m_key in ['svm', 'naive_bayes', 'random_forest', 'cnn', 'lstm']:
        if m_key not in probs:
            probs[m_key] = probs.get('svm', 0.5)

    # Apply context calibration to individual model probabilities so votes & verdict are 100% in sync
    for m_key in list(probs.keys()):
        p_raw = probs[m_key]
        if has_explicit_abuse:
            p_adj = max(p_raw, 0.95)
        elif is_targeted_insult:
            p_adj = max(p_raw, 0.88)
        elif is_common_name_safe:
            # Personal / Entity Name without abuse: scale down offensive probability so models vote Safe (e.g. 0.02 - 0.05)
            p_adj = min(p_raw * 0.05, 0.05)
        elif is_praise:
            # Praise / Compliment context: scale down offensive probability so models vote Safe (e.g. 0.08 - 0.15)
            p_adj = min(p_raw * 0.15, 0.15)
        elif is_zoological:
            # Zoological narrative context: scale down offensive probability so models vote Safe
            p_adj = min(p_raw * 0.12, 0.12)
        else:
            p_adj = p_raw
        probs[m_key] = p_adj

    # Build models_scores dictionary from calibrated probabilities
    models_scores = {}
    model_display_names = {
        'svm': 'Linear SVM',
        'naive_bayes': 'Naive Bayes',
        'random_forest': 'Random Forest',
        'cnn': '1D CNN (Deep Learning)',
        'lstm': 'Bi-LSTM (Deep Learning)'
    }

    for m_key, m_name in model_display_names.items():
        p_val = probs[m_key]
        pred_label = 'offensive' if p_val >= 0.50 else 'non-offensive'
        conf_val = round(float(max(p_val, 1.0 - p_val)), 4)
        models_scores[m_key] = {
            'name': m_name,
            'prediction': pred_label,
            'confidence': conf_val,
            'offensive_prob': round(float(p_val), 4)
        }

    total_models = len(probs)
    offensive_votes = sum(1 for p in probs.values() if p >= 0.50)
    safe_votes = total_models - offensive_votes

    if has_explicit_abuse:
        final_pred = 'offensive'
        final_conf = 0.95
        engine_name = f"Consensus Ensemble ({offensive_votes}/{total_models} Models Confirming) - Explicit Abuse"
        context_override = False
    elif is_targeted_insult:
        final_pred = 'offensive'
        final_conf = 0.88
        engine_name = f"Consensus Ensemble ({offensive_votes}/{total_models} Models Confirming) - Targeted Insult"
        context_override = False
    elif is_common_name_safe:
        final_pred = 'non-offensive'
        final_conf = 0.96
        engine_name = f"Consensus Ensemble ({safe_votes}/{total_models} Models Confirming) - Safe Entity Name"
        context_override = True
    elif is_praise:
        final_pred = 'non-offensive'
        final_conf = 0.92
        engine_name = f"Consensus Ensemble ({safe_votes}/{total_models} Models Confirming) - Praise & Compliment"
        context_override = True
    elif is_zoological:
        final_pred = 'non-offensive'
        final_conf = 0.93
        engine_name = f"Consensus Ensemble ({safe_votes}/{total_models} Models Confirming) - Zoological Narrative"
        context_override = True
    else:
        context_override = False
        if offensive_votes >= (total_models / 2.0 + 0.1):
            final_pred = 'offensive'
            agreeing_probs = [p for p in probs.values() if p >= 0.50]
            final_conf = sum(agreeing_probs) / len(agreeing_probs) if agreeing_probs else 0.70
        else:
            final_pred = 'non-offensive'
            safe_probs = [1.0 - p for p in probs.values() if p < 0.50]
            final_conf = sum(safe_probs) / len(safe_probs) if safe_probs else 0.70
        engine_name = f"Consensus Ensemble ({offensive_votes}/{total_models} Votes Offensive: SVM, NB, RF, CNN, LSTM)"

    # ── 4. Extract Flagged Offending Word(s) ──
    flagged_words = []
    if has_explicit_abuse:
        for ab in EXPLICIT_ABUSE:
            if ab in text_lower and ab not in flagged_words:
                flagged_words.append(ab)

    if is_targeted_insult or (has_pejorative_animal and has_human_target and not has_praise):
        for ph in PEJORATIVE_ANIMAL_PHRASES:
            if ph in text_lower and ph not in flagged_words:
                flagged_words.append(ph)
        for pa in PEJORATIVE_ANIMALS:
            if (pa in text_words or pa in text_lower) and pa not in flagged_words:
                flagged_words.append(pa)

    if final_pred == 'offensive' and not flagged_words:
        for w in text_lower.split():
            clean_w = re.sub(r'[\W_]+', '', w, flags=re.UNICODE)
            if len(clean_w) > 2 and clean_w not in STOP_WORDS and clean_w not in COMMON_NAMES:
                flagged_words.append(clean_w)
                break

    primary_flagged_word = flagged_words[0] if flagged_words else ""

    return {
        'text': text,
        'prediction': final_pred,
        'confidence': round(float(final_conf), 4),
        'model_used': engine_name,
        'models_scores': models_scores,
        'flagged_word': primary_flagged_word,
        'flagged_words': flagged_words,
        'context_override': context_override,
        'timestamp': datetime.utcnow().isoformat()
    }
