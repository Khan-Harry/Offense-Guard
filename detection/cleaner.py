"""
Text Cleaner for Context and Model Preprocessing
Handles emojis, URL stripping, unicode normalization and spacing.
"""

import re


def clean_text_for_context(text):
    """Clean text while strictly preserving words and semantic order."""
    if not text:
        return ""
    # Strip URL links
    t = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Remove emoji and uncommon non-word characters, preserving Urdu script range U+0600-U+06FF
    t = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', t)
    # Collapse multiple whitespaces
    t = re.sub(r'\s+', ' ', t).strip()
    return t
