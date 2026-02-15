from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.rag_service import SimpleRAG

app = FastAPI()
rag = SimpleRAG()

class Question(BaseModel):
    question: str

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    rag.ingest_document(content)
    return {"message": "Document indexed successfully."}

@app.post("/ask")
async def ask(q: Question):
    answer = rag.ask(q.question)
    return {"answer": answer["result"]}
