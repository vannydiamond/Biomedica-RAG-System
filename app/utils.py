 # Helpers: text extraction, chunking

import os
from typing import List 
from pypdf import PdfReader
import docx

def extract_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return " ".join([page.extract_text() for page in reader.pages])
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return " ".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format")