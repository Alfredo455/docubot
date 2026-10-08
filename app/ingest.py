"""Load markdown files and split them into overlapping word chunks."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    if size <= overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def load_chunks(docs_dir: str, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).glob("**/*.md")):
        text = path.read_text(encoding="utf-8")
        for piece in chunk_words(text, size, overlap):
            chunks.append(Chunk(source=path.name, text=piece))
    return chunks
