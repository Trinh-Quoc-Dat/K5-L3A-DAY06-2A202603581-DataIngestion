
"""Fixed-size and paragraph-aware chunking."""
import re

from .models import Chunk, DocumentPage


def _make_chunk(
    page: DocumentPage,
    index: int,
    text: str,
    strategy: str,
) -> Chunk:
    """Create a traceable chunk with deterministic ID."""
    return Chunk(
        chunk_id=f"{page.document_id}-p{page.page}-{strategy}-{index:04d}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text.strip(),
        metadata=dict(page.metadata),
    )


def fixed_size(
    page: DocumentPage,
    size: int,
    overlap: int,
) -> list[Chunk]:
    """Split a page into fixed character windows with overlap."""
    if size <= 0:
        raise ValueError("size must be greater than zero.")

    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than size.")

    text = page.text.strip()

    if not text:
        return []

    chunks = []
    step = size - overlap
    start = 0
    index = 0

    while start < len(text):
        end = min(start + size, len(text))
        content = text[start:end].strip()

        if content:
            chunks.append(
                _make_chunk(page, index, content, "fixed")
            )
            index += 1

        if end >= len(text):
            break

        start += step

    return chunks


def _split_long_paragraph(paragraph: str, max_size: int) -> list[str]:
    """Split an oversized paragraph, preferring sentence boundaries."""
    if len(paragraph) <= max_size:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        # A single sentence may itself exceed max_size.
        if len(sentence) > max_size:
            if current:
                pieces.append(current)
                current = ""

            words = sentence.split()
            word_part = ""

            for word in words:
                candidate = f"{word_part} {word}".strip()

                if len(candidate) > max_size and word_part:
                    pieces.append(word_part)
                    word_part = word
                else:
                    word_part = candidate

                # Preserve very long tokens without truncation.
                if len(word_part) > max_size:
                    pieces.append(word_part)
                    word_part = ""

            if word_part:
                current = word_part

            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) > max_size and current:
            pieces.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces


def by_paragraph(
    page: DocumentPage,
    max_size: int,
) -> list[Chunk]:
    """Group paragraphs up to max_size, splitting oversized paragraphs."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than zero.")

    text = page.text.strip()

    if not text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks = []
    current_parts = []
    current_length = 0
    index = 0

    def flush() -> None:
        nonlocal index, current_parts, current_length

        if not current_parts:
            return

        content = "\n\n".join(current_parts).strip()

        chunks.append(
            _make_chunk(page, index, content, "paragraph")
        )

        index += 1
        current_parts = []
        current_length = 0

    for paragraph in paragraphs:
        pieces = _split_long_paragraph(paragraph, max_size)

        for piece in pieces:
            added_length = len(piece) + (
                2 if current_parts else 0
            )

            if current_parts and current_length + added_length > max_size:
                flush()

            current_parts.append(piece)
            current_length += len(piece) + (
                2 if len(current_parts) > 1 else 0
            )

    flush()
    return chunks
