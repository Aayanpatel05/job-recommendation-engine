import re
import os
from pdfminer.high_level import extract_text as extract_pdf_text
import docx

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