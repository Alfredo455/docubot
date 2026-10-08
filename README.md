# DocuBot

Asistente de preguntas y respuestas sobre documentación en Markdown, con arquitectura RAG
(Retrieval-Augmented Generation).

**Stack:** Python 3.12 · FastAPI · Pydantic v2 · BM25 propio · Claude (opcional) · Docker · pytest

## Cómo funciona

1. `ingest.py` lee los `.md` y los parte en fragmentos con solapamiento.
2. `retriever.py` indexa los fragmentos con **BM25** (IDF suavizado, estable en corpus pequeños).
3. `generator.py` arma un prompt con los fragmentos recuperados y le pide a Claude que responda
   **solo con ese contexto**. Sin API key devuelve el fragmento más relevante (modo offline).
4. `POST /ask` responde con la respuesta y las fuentes citadas.

## Uso

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
curl -X POST localhost:8000/ask -H "content-type: application/json" \
     -d '{"question": "¿Cuánto pesa la maleta de mano?"}'
pytest
```

Variables (prefijo `DOCUBOT_`): `ANTHROPIC_API_KEY`, `DOCS_DIR`, `TOP_K`, `CHUNK_SIZE`.

## Decisiones

- **BM25 propio en vez de `rank-bm25`:** esa librería devuelve puntaje 0 con corpus de 1-2 documentos
  (IDF negativo). Con IDF suavizado el ranking funciona también en corpus pequeños.
- **Fallback offline:** el servicio y los tests corren sin red ni credenciales.
- **Siguiente paso:** embeddings + base vectorial para búsqueda semántica y reranking.
