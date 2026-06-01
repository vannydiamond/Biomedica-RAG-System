# API routes (upload, ask)

from fastapi import APIRouter, UploadFile, File
from .services import process_document, answer_question 
import shutil
from .utils import extract_text
from .services import QASystem

router = APIRouter()
qa_system = QASystem()

@router.post("/upload")
async def upload_doc(file: UploadFile):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    text = extract_text(file_path)
    qa_system.build_index(text)
    return {"status": "document uploaded"}

@router.get("/ask")
async def ask_question(q: str):
    answer = qa_system.answer(q)
    return {"answer": answer}