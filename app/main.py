from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import settings
from .generator import generate
from .ingest import load_chunks
from .retriever import Retriever

app = FastAPI(title="DocuBot", version="0.1.0")
retriever = Retriever(load_chunks(settings.docs_dir, settings.chunk_size, settings.chunk_overlap))


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    source: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(retriever.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    hits = retriever.search(req.question, settings.top_k)
    answer = generate(req.question, [c for c, _ in hits], settings)
    return AskResponse(
        answer=answer,
        sources=[Source(source=c.source, score=round(s, 3), snippet=c.text[:160]) for c, s in hits],
    )
