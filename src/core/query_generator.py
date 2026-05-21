import re
import spacy

nlp = spacy.load("en_core_web_sm")


DEFAULT_QUERIES = [
    "software development",
    "data science",
    "machine learning"
]

# Words/phrases that are too generic for job searching
STOP_WORDS = {
    "us citizen",
    "gpa",
    "expected graduation",
    "relevant courses",
    "coursework",
    "skills",
    "experience",
    "projects",
    "education",
    "work experience",
    "professional experience",
    "summary",
    "contact",
    "references",
    "phone",
    "email",
    "linkedin"
}


def is_valid_phrase(phrase: str) -> bool:

    # Minimum length
    if len(phrase) < 4:
        return False

    # Remove weird symbols
    if re.search(r"[^a-zA-Z0-9\s]", phrase):
        return False

    # Too many digits
    digit_count = sum(char.isdigit() for char in phrase)

    if len(phrase) > 0 and (digit_count / len(phrase)) > 0.2:
        return False

    # Generic stop phrases
    if phrase in STOP_WORDS:
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

        # Remove extra spaces
        phrase = re.sub(r"\s+", " ", phrase)

        if not is_valid_phrase(phrase):
            continue

        # Skip duplicates
        if phrase in phrases:
            continue

        phrases.append(phrase)

    # Fallback queries
    if not phrases:
        return DEFAULT_QUERIES[:max_queries]

    return phrases[:max_queries]