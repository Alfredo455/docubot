import pytest
from fastapi.testclient import TestClient

from app.ingest import Chunk, chunk_words
from app.main import app
from app.retriever import Retriever


def test_chunk_overlap():
    text = " ".join(str(i) for i in range(10))
    chunks = chunk_words(text, size=4, overlap=1)
    assert chunks[0] == "0 1 2 3"
    assert chunks[1].startswith("3 ")
    assert chunks[-1].endswith("9")


def test_chunk_invalid_params():
    with pytest.raises(ValueError):
        chunk_words("a b c", size=2, overlap=2)


def test_retriever_ranks_relevant_first():
    r = Retriever([Chunk("a.md", "el equipaje facturado pesa 23 kg"), Chunk("b.md", "las mascotas pagan limpieza")])
    top, _ = r.search("limite de equipaje", k=1)[0]
    assert top.source == "a.md"


def test_ask_endpoint_returns_sources():
    client = TestClient(app)
    res = client.post("/ask", json={"question": "¿Cuánto pesa la maleta de mano?"})
    assert res.status_code == 200
    body = res.json()
    assert body["sources"] and body["sources"][0]["source"] == "vuelos.md"


def test_ask_validates_input():
    assert TestClient(app).post("/ask", json={"question": "a"}).status_code == 422
