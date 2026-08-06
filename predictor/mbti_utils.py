import numpy as np
import json
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent.parent / 'ml_models' / 'results.json'

def get_reliability_tier(dim):
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    r = results[dim]
    margin = r['test_accuracy'] - r['test_baseline']
    if not r['beats_baseline']:
        return 'unreliable'
    elif margin < 0.02:
        return 'weak'
    else:
        return 'reliable'

    
FEATURE_ORDER = [
    'avg_word_length',
    'total_word_count',
    'question_mark_freq',
    'exclamation_freq',
    'i_we_ratio',
    'unique_word_ratio',
    'avg_sentence_length',
    'ellipsis_freq',
    'caps_word_ratio',
]

QUESTIONS = [
    {'id': 'q2', 'feature': 'total_word_count', 'text': "When someone asks how your day was, how much do you typically say?",
     'options': [('One or two words', -0.57), ('A short sentence', 0.17), ('A few sentences', 0.74), ('A detailed story', 1.39)]},
    {'id': 'q3', 'feature': 'question_mark_freq', 'text': "In conversations, how often do you ask follow-up questions?",
     'options': [('Rarely', -0.06), ('Sometimes', -0.03), ('Often', 0.01), ('Constantly', 0.04)]},
    {'id': 'q6', 'feature': 'exclamation_freq', 'text': "When texting, how often do you use exclamation marks?",
     'options': [('Never', -0.54), ('Rarely', -0.30), ('Often', 0.16), ('Almost every message', 0.51)]},
    {'id': 'q8', 'feature': 'i_we_ratio', 'text': "In conversation, do you find yourself saying 'I think' more, or 'we should' more?",
     'options': [("Almost always 'we'", -0.39), ("More 'we' than 'I'", 0.21), ("More 'I' than 'we'", 0.66), ("Almost always 'I'", 1.18)]},
    {'id': 'q10', 'feature': 'avg_sentence_length', 'text': "When you write a message, do you prefer short, punchy sentences or long, flowing ones?",
     'options': [('Very short', -0.54), ('Fairly short', -0.08), ('Fairly long', 0.42), ('Very long and detailed', 0.90)]},
    {'id': 'q12', 'feature': 'ellipsis_freq', 'text': "Do you often trail off mid-thought when writing, using '...'?",
     'options': [('Never', -0.66), ('Rarely', -0.06), ('Sometimes', 0.54), ('Often', 1.14)]},
    {'id': 'q14', 'feature': 'caps_word_ratio', 'text': "How often do you use ALL CAPS for emphasis when texting?",
     'options': [('Never', -0.65), ('Rarely', -0.22), ('Sometimes', 0.37), ('Often', 0.88)]},
]

DEFAULT_ZERO_FEATURES = ['avg_word_length', 'unique_word_ratio']


def build_feature_vector(answers):
    """answers: dict like {'q2': 0.5, 'q3': -1.5, ...} -> numpy array shape (9, 1)"""
    feature_scores = {name: 0.0 for name in DEFAULT_ZERO_FEATURES}
    for question in QUESTIONS:
        feature_scores[question['feature']] = answers[question['id']]

    vector = [feature_scores[name] for name in FEATURE_ORDER]
    return np.array(vector).reshape(-1, 1)


def forward_propagation(X, params):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']

    Z1 = W1 @ X + b1
    A1 = np.maximum(0, Z1)
    Z2 = W2 @ A1 + b2
    A2 = np.maximum(0, Z2)
    Z3 = W3 @ A2 + b3
    A3 = 1 / (1 + np.exp(-Z3))
    return A3


def predict_mbti(answers, mbti_models):
    X = build_feature_vector(answers)
    letter_pairs = {'IE': ('I', 'E'), 'NS': ('N', 'S'), 'TF': ('T', 'F'), 'JP': ('J', 'P')}
    result_type = ''
    confidences = {}

    for dim, (letter_1, letter_0) in letter_pairs.items():
        prob = forward_propagation(X, mbti_models[dim])[0][0]
        tier = get_reliability_tier(dim)
        if prob >= 0.5:
            result_type += letter_1
        else:
            result_type += letter_0
        conf = prob if prob >= 0.5 else (1 - prob)
        confidences[dim] = {'value': round(conf * 100, 1), 'tier': tier}

    return result_type, confidences