"""Answer generation. Uses Claude when an API key is set, otherwise returns the top passage."""
from anthropic import Anthropic

from .config import Settings
from .ingest import Chunk

SYSTEM = (
    "Answer the question using ONLY the provided context. "
    "If the context does not contain the answer, say you don't know. Be concise."
)


def build_prompt(question: str, passages: list[Chunk]) -> str:
    context = "\n\n".join(f"[{i + 1}] ({c.source}) {c.text}" for i, c in enumerate(passages))
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate(question: str, passages: list[Chunk], cfg: Settings) -> str:
    if not passages:
        return "No encontré información relevante en la documentación."
    if not cfg.anthropic_api_key:
        return passages[0].text  # offline fallback: extractive answer
    client = Anthropic(api_key=cfg.anthropic_api_key)
    msg = client.messages.create(
        model=cfg.model,
        max_tokens=400,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(question, passages)}],
    )
    return msg.content[0].text
