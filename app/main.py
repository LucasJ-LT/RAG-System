from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .ingest import ingest_documents
from .rag import generate_rag_answer

app = FastAPI(title="RAG FastAPI PoC", version="0.1.0")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ingest")
def ingest():
    try:
        count = ingest_documents()
        return {"indexed_documents": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    answer, sources = generate_rag_answer(question)
    return {"answer": answer, "sources": sources}
