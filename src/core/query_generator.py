import re
import spacy

nlp = spacy.load("en_core_web_sm")


DEFAULT_QUERIES = [
    "software development",
    "data science",
    "machine learning"
]


def is_valid_phrase(phrase: str) -> bool:

    if len(phrase) < 4:
        return False

    if re.search(r"[^a-zA-Z0-9\s]", phrase):
        return False

    digit_count = sum(char.isdigit() for char in phrase)

    if len(phrase) > 0 and (digit_count / len(phrase)) > 0.2:
        return False

    return True


def generate_queries_from_resume(
    resume_text,
    max_queries=10
):
    doc = nlp(resume_text)

    phrases = []

    for chunk in doc.noun_chunks:

        phrase = chunk.text.strip().lower()

        if not is_valid_phrase(phrase):
            continue

        if phrase in phrases:
            continue

        phrases.append(phrase)

    if not phrases:
        return DEFAULT_QUERIES[:max_queries]

    return phrases[:max_queries]