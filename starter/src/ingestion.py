
"""Read and normalize PDF source data."""
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraphs."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove non-printing control characters, except newline and tab.
    text = "".join(
        char for char in text
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )

    # Normalize horizontal whitespace without joining paragraphs.
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
    ]

    # Preserve paragraph boundaries, remove excessive blank lines.
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
) -> list[DocumentPage]:
    """Extract every physical PDF page and preserve its page number."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF.")

    if not document_id or not source:
        raise ValueError("document_id and source are required.")

    reader = PdfReader(str(path))

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")

    pages: list[DocumentPage] = []

    for page_number, pdf_page in enumerate(reader.pages, start=1):
        raw_text = pdf_page.extract_text() or ""
        text = normalize_text(raw_text)

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={
                    "page_has_text": bool(text),
                    "extraction_method": "pypdf",
                },
            )
        )

    return pages

# Verified physical PDF page boundaries.
CHAPTER_STARTS = {
    1: 19,
    2: 67,
    3: 113,
    4: 155,
    5: 223,
    6: 275,
    7: 341,
    8: 387,
    9: 421,
    10: 471,
}

SECTION_STARTS = [
    (19, "1.1"), (22, "1.2"), (26, "1.3"),
    (38, "1.4"), (40, "1.5"),
    (68, "2.1"), (71, "2.2"), (76, "2.3"),
    (86, "2.4"), (98, "2.5"),
    (114, "3.1"), (120, "3.2"), (125, "3.3"),
    (129, "3.4"), (137, "3.5"),
    (156, "4.1"), (175, "4.2"),
    (197, "4.3"), (212, "4.4"),
    (223, "5.1"), (232, "5.2"),
    (237, "5.3"), (263, "5.4"),
    (276, "6.1"), (284, "6.2"),
    (303, "6.3"), (316, "6.4"), (326, "6.5"),
    (342, "7.1"), (351, "7.2"), (363, "7.3"),
    (367, "7.4"), (369, "7.5"),
    (388, "8.1"), (398, "8.2"), (405, "8.3"),
    (422, "9.1"), (436, "9.2"), (441, "9.3"),
    (449, "9.4"), (455, "9.5"),
    (472, "10.1"), (479, "10.2"), (494, "10.3"),
]

APPENDIX_RANGES = [
    (505, 522, "Appendix A"),
    (523, 538, "Appendix B"),
    (539, 544, "Appendix C"),
    (545, 558, "Appendix D"),
]


def assign_metadata(pages: list[DocumentPage]) -> list[DocumentPage]:
    """Assign metadata using verified physical PDF page boundaries."""

    for page in pages:
        number = page.page

        metadata = {
            "language": "en",
            "page_has_text": bool(page.text.strip()),
            "extraction_method": "pypdf",
        }

        # Front matter
        if 1 <= number <= 18:
            metadata["category"] = "front_matter"

        # Main chapters
        elif 19 <= number <= 503:
            chapter = max(
                chapter_number
                for chapter_number, start_page in CHAPTER_STARTS.items()
                if start_page <= number
            )

            metadata["category"] = "chapter"
            metadata["chapter"] = str(chapter)

            current_sections = [
                section
                for start, section in SECTION_STARTS
                if start <= number and section.split(".", 1)[0] == str(chapter)
            ]

            if current_sections:
                metadata["section"] = current_sections[-1]

        # References
        elif number == 504:
            metadata["category"] = "references"

        # Appendices
        elif 505 <= number <= 558:
            metadata["category"] = "appendix"

            for start, end, appendix_name in APPENDIX_RANGES:
                if start <= number <= end:
                    metadata["section"] = appendix_name
                    break

        # Answer key
        elif 559 <= number <= 562:
            metadata["category"] = "answer_key"

        # Index
        elif 563 <= number <= 569:
            metadata["category"] = "index"

        else:
            metadata["category"] = "unclassified"

        page.metadata.update(metadata)

    return pages
