import textract
import re

def extract_text_from_resume(file_path: str) -> str:
    """
    Extract text from a resume file (PDF, DOCX, TXT, etc.)
    """
    text = textract.process(file_path).decode('utf-8')
    return text

def clean_resume_text(text: str) -> str:
    """
    Optional: Clean up text, remove extra spaces, normalize.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
    text = text.strip()
    return text