import re
import os
from pdfminer.high_level import extract_text as extract_pdf_text
import docx
import pandas as pd

def extract_text_from_resume(file_path: str) -> str:
    """
    Extract text from resume (PDF, DOCX, TXT) without external dependencies.
    """
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            text = extract_pdf_text(file_path)

        elif ext == ".docx":
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return text

    except Exception as e:
        raise Exception(f"Error extracting text: {str(e)}")


def clean_resume_text(text: str) -> str:
    """
    Clean text
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

CUE_PATTERN = re.compile(
    r"""
    (?:
        experience\s+(?:with|in) |
        proficiency\s+in |
        knowledge\s+of |
        familiar(?:ity)?\s+with |
        skilled\s+in |
        expertise\s+in |
        working\s+knowledge\s+of |
        hands[-\s]?on\s+experience\s+(?:with|in)
    )
    \s+                                   # whitespace after cue
    (?P<chunk>                             # capture the chunk
        .*?                                # non-greedy
    )
    (?=                                    # stop when we hit a delimiter
        [\.\;\n\r] |                       # period/semicolon/newline
        \u2022 |                           # bullet •
        \s-\s |                            # " - " often used in listings
        $                                   # or end of string
    )
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL
)

def extract_skill_chunks_from_description(text: str, max_chunks: int = 10) -> list[str]:
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return []

    chunks = []
    for m in CUE_PATTERN.finditer(text):
        chunk = m.group("chunk").strip()
        chunk = re.sub(r"\s+", " ", chunk)
        chunk = chunk.strip(" :,-–—•*")
        if len(chunk) >= 2:
            chunks.append(chunk)
        if len(chunks) >= max_chunks:
            break

    # dedupe preserving order
    seen = set()
    out = []
    for c in chunks:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out

def add_description_chunks_to_skills_desc(df: pd.DataFrame,
                                         desc_col="description",
                                         skills_col="skills_desc") -> pd.DataFrame:
    # make sure skills_desc exists
    if skills_col not in df.columns:
        df[skills_col] = ""

    def _append(row):
        desc = row.get(desc_col, "")
        existing = row.get(skills_col, "")
        existing = "" if pd.isna(existing) else str(existing)

        chunks = extract_skill_chunks_from_description(desc)
        if not chunks:
            return existing

        chunk_text = "; ".join(chunks)
        return (existing + ("; " if existing.strip() else "") + chunk_text).strip()

    df[skills_col] = df.apply(_append, axis=1)
    return df

EXPERIENCE_LEVEL_MAP = {
    # Executive
    "chief executive officer": "Executive",
    "chief technology officer": "Executive",
    "chief financial officer": "Executive",
    "chief operating officer": "Executive",
    "ceo": "Executive",
    "cto": "Executive",
    "cfo": "Executive",
    "coo": "Executive",
    "vice president": "Executive",
    "vp": "Executive",
    "general manager": "Executive",

    # Director
    "senior director": "Director",
    "director": "Director",
    "director of": "Director",
    "head of": "Director",

    # Senior / Mid level (merged cleanly)
    "principal": "Senior",
    "staff": "Senior",
    "lead": "Senior",
    "senior": "Senior",
    "sr": "Senior",
    "sr.": "Senior",
    "manager": "Senior",

    # Entry / Junior
    "associate": "Entry",
    "assistant": "Entry",
    "junior": "Entry",
    "jr": "Entry",

    # Internship
    "intern": "Intern",
    "internship": "Intern",
    "trainee": "Intern",
    "apprentice": "Intern",
}


def extract_experience_level(title, description, existing_value=None):
    """
    Clean experience classifier with normalized levels.
    """

    if pd.notna(existing_value) and str(existing_value).strip():
        return existing_value

    title = str(title).lower()
    description = str(description).lower()

    # 1. Title match (highest priority)
    for keyword, level in EXPERIENCE_LEVEL_MAP.items():
        if keyword in title:
            return level

    # 2. Description match
    for keyword, level in EXPERIENCE_LEVEL_MAP.items():
        if keyword in description:
            return level

    # 3. Years-based inference
    year_match = re.search(r'(\d+)\+?\s*years?', description)

    if year_match:
        years = int(year_match.group(1))

        if years < 1:
            return "Intern"
        elif years < 3:
            return "Entry"
        elif years < 7:
            return "Senior"
        elif years < 12:
            return "Director"
        else:
            return "Executive"

    return "Unknown"