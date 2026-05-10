import spacy

nlp = spacy.load("en_core_web_sm")


def generate_queries_from_resume(resume_text, max_queries=10):

    doc = nlp(resume_text)

    phrases = []

    for chunk in doc.noun_chunks:

        phrase = chunk.text.strip().lower()

        # Clean up
        if len(phrase) < 4:
            continue

        if phrase in phrases:
            continue

        phrases.append(phrase)

    if not phrases:
        phrases = ["software", "engineer", "developer"]

    return phrases[:max_queries]