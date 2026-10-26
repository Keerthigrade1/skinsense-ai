"""PDF text extraction and retrieval for document Q&A.

Uploaded files are untrusted: we check size, page count and file signature,
only ever read text (no scripts, links or embedded files are opened), and the
extracted text is treated purely as data for an extractive QA model, which
cannot follow instructions inside the document.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from . import nlp

MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 50
MAX_CHARS = 200_000
CHUNK_WORDS = 110
CHUNK_OVERLAP = 30
TOP_K_CHUNKS = 3
# At or above this QA confidence the span is shown as the answer. Below it we
# only show the most relevant passage and let the user judge.
CONFIDENT_SCORE = 0.10
MIN_FALLBACK_KEYWORD_HITS = 2
# Questions about medical suitability get a "can't advise" notice whatever the document says.
MEDICAL_TERMS = re.compile(
    r"\b(pregnan\w*|breastfeed\w*|eczema|rosacea|psoriasis|dermatitis|acne|allerg\w*|infection|cure|treat\w*|"
    r"medical|medication|prescription|doctor|safe for|safe to|condition)\b", re.I)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class PdfError(ValueError):
    pass


@dataclass
class Extracted:
    text: str
    pages: int
    warnings: list[str] = field(default_factory=list)


def extract_text(data: bytes) -> Extracted:
    if len(data) > MAX_BYTES:
        raise PdfError(f"File is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    if not data.lstrip()[:5].startswith(b"%PDF-"):
        raise PdfError("This doesn't look like a PDF file (missing PDF header).")
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PdfError("This PDF is password-protected.")
        n_pages = len(reader.pages)
    except PdfError:
        raise
    except Exception as exc:
        raise PdfError(f"Couldn't read this PDF ({type(exc).__name__}).") from exc

    warnings = []
    if n_pages > MAX_PAGES:
        warnings.append(f"Only the first {MAX_PAGES} of {n_pages} pages were read.")
    parts = []
    for i, page in enumerate(reader.pages[:MAX_PAGES]):
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            warnings.append(f"Page {i + 1} couldn't be read and was skipped.")
    text = _CONTROL_CHARS.sub(" ", "\n".join(parts))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        warnings.append("The document is long, so only the first part was used.")
    if not text:
        warnings.append("No selectable text found. Scanned/image-only PDFs aren't supported.")
    return Extracted(text=text, pages=n_pages, warnings=warnings)


def chunk(text: str) -> list[str]:
    words = text.split()
    step = CHUNK_WORDS - CHUNK_OVERLAP
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, max(len(words) - CHUNK_OVERLAP, 1), step)]


def top_chunks(question: str, chunks: list[str], k: int = TOP_K_CHUNKS) -> list[str]:
    q = nlp._content_words(question)
    scored = sorted(chunks, key=lambda c: -len(q & nlp._content_words(c)))
    return scored[:k]


def ask(question: str, text: str) -> dict:
    """Answer a question from document text.

    `status` is one of:
      "answer"     - confident span copied from the document
      "low"        - weak span; show the passage and ask the user to verify
      "not_found"  - the document doesn't mention the question's topic, or nothing matched
    """
    chunks = chunk(text)
    candidates = top_chunks(question, chunks)
    q_words = nlp._content_words(question)
    overlap = max((len(q_words & nlp._content_words(c)) for c in candidates), default=0)
    medical = bool(MEDICAL_TERMS.search(question))

    if overlap == 0:
        # Guard: an extractive model always returns *some* span, even for unrelated
        # questions, so we don't call it when no passage shares a keyword.
        return {"status": "not_found", "answer": "", "score": None, "context": "", "start": -1, "end": -1,
                "source": "guard", "keyword_overlap": 0, "medical": medical,
                "chunks_searched": len(candidates), "chunks_total": len(chunks)}

    result = nlp.answer_question(question, candidates)
    if result["source"] == "model":
        if not result["answer"].strip():
            status = "not_found"
        else:
            status = "answer" if result["score"] >= CONFIDENT_SCORE else "low"
    else:
        needed = min(MIN_FALLBACK_KEYWORD_HITS, max(result.get("question_words", 0), 1))
        status = "answer" if result.get("keyword_hits", 0) >= needed else "not_found"
    result.update(status=status, keyword_overlap=overlap, medical=medical,
                  chunks_searched=len(candidates), chunks_total=len(chunks))
    return result
