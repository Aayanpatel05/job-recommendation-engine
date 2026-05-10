import spacy
import re

nlp = spacy.load("en_core_web_sm")


def generate_queries_from_resume(resume_text, max_queries=10):

    doc = nlp(resume_text)

    phrases = []

    for chunk in doc.noun_chunks:

        phrase = chunk.text.strip().lower()

        # Remove phrases with weird symbols
        if re.search(r"[^a-zA-Z0-9\s]", phrase):
            continue

        # Remove phrases mostly made of numbers
        digit_count = sum(c.isdigit() for c in phrase)

        if len(phrase) > 0 and (digit_count / len(phrase)) > 0.2:
            continue

        # Remove very short phrases
        if len(phrase) < 4:
            continue

        # Remove duplicates
        if phrase in phrases:
            continue

        phrases.append(phrase)

    # fallback
    if not phrases:
        phrases = [
            "software development",
            "data science",
            "machine learning"
        ]

    return phrases[:max_queries]