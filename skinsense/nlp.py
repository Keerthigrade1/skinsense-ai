"""Hugging Face pipelines with honest fallbacks.

Every function returns a dict with `source` set to "model" or "fallback".
Fallbacks are simple rule-based methods; the UI must label them as such and
never present their output as model predictions.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str
    task: str
    model_id: str
    title: str
    plain_task: str


MODELS = {
    "sentiment": ModelSpec(
        "sentiment", "text-classification",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "Review sentiment",
        "Text classification: reads a review and picks POSITIVE, NEUTRAL or NEGATIVE, with a confidence score.",
    ),
    "summarization": ModelSpec(
        "summarization", "summarization",
        "Falconsai/text_summarization",
        "Review summarization",
        "Summarization: writes a shorter version of a longer text, keeping its main points.",
    ),
    "qa": ModelSpec(
        "qa", "question-answering",
        "deepset/tinyroberta-squad2",
        "Document Q&A",
        "Extractive question answering: finds the span of text in a passage that best answers a question. "
        "It can only copy words from the passage, never write new facts.",
    ),
    "zero_shot": ModelSpec(
        "zero_shot", "zero-shot-classification",
        "cross-encoder/nli-distilroberta-base",
        "Zero-shot categorization",
        "Zero-shot classification: scores how well a text matches labels the model was never trained on, "
        "by testing sentences like 'This product description is about hydration.'",
    ),
}

FORCE_FALLBACK = os.environ.get("SKINSENSE_FORCE_FALLBACK", "").lower() in {"1", "true", "yes"}


@dataclass
class LoadedModel:
    spec: ModelSpec
    pipe: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.pipe is not None


@lru_cache(maxsize=None)
def load(key: str) -> LoadedModel:
    spec = MODELS[key]
    if FORCE_FALLBACK:
        return LoadedModel(spec, error="Fallback mode forced via SKINSENSE_FORCE_FALLBACK.")
    try:
        from transformers import pipeline

        return LoadedModel(spec, pipe=pipeline(spec.task, model=spec.model_id, device=-1))
    except Exception as exc:  # network, disk, missing dependency, etc.
        return LoadedModel(spec, error=f"{type(exc).__name__}: {str(exc)[:300]}")


def tokenize_preview(key: str, text: str, limit: int = 40) -> dict | None:
    """Show how the model's tokenizer splits text. None when the model isn't loaded."""
    m = load(key)
    if not m.ok:
        return None
    tok = m.pipe.tokenizer
    tokens = tok.tokenize(text)
    return {"tokens": tokens[:limit], "count": len(tokens), "max_length": getattr(tok, "model_max_length", None)}


# ---------------------------------------------------------------- sentiment
_POS = {"love", "great", "perfect", "soft", "gentle", "comfortable", "lovely", "worth", "good", "nice",
        "smooth", "smoother", "hydrating", "light", "lightweight", "refreshing", "finally", "best", "bouncy"}
_NEG = {"broke", "drying", "rough", "leaked", "sticky", "heavy", "irritated", "annoying", "pilled", "tiny",
        "strong", "shiny", "cast", "not", "too", "stripped", "itchy", "sting", "stopped", "hard"}


def classify_sentiment(texts: list[str]) -> list[dict]:
    m = load("sentiment")
    if m.ok:
        try:
            out = m.pipe(texts, truncation=True, max_length=512)
            return [{"label": o["label"].upper(), "score": float(o["score"]), "source": "model"} for o in out]
        except Exception as exc:
            err = str(exc)
    else:
        err = m.error
    results = []
    for t in texts:
        words = re.findall(r"[a-z']+", t.lower())
        pos, neg = sum(w in _POS for w in words), sum(w in _NEG for w in words)
        results.append({"label": "POSITIVE" if pos >= neg else "NEGATIVE", "score": None,
                        "source": "fallback", "error": err})
    return results


# ------------------------------------------------------------ summarization
def summarize(texts: list[str]) -> dict:
    joined = " ".join(t.strip() for t in texts if t.strip())
    if not joined:
        return {"summary": "", "source": "none"}
    m = load("summarization")
    if m.ok:
        try:
            n_words = len(joined.split())
            max_new = max(30, min(100, n_words))
            out = m.pipe(joined, max_new_tokens=max_new, min_new_tokens=12, do_sample=False, truncation=True)
            summary = re.sub(r"\s+([.,!?])", r"\1", out[0]["summary_text"]).strip()
            # Drop a trailing sentence fragment if generation stopped mid-sentence.
            if summary and summary[-1] not in ".!?" and re.search(r"[.!?]", summary):
                summary = summary[: max(summary.rfind(c) for c in ".!?") + 1]
            return {"summary": summary, "source": "model", "input_words": n_words}
        except Exception as exc:
            err = str(exc)
    else:
        err = m.error
    return {"summary": _extractive_summary(texts), "source": "fallback", "error": err}


def _extractive_summary(texts: list[str], n: int = 2) -> str:
    """Pick the sentences that share the most common words with the rest (rule-based)."""
    sentences = [s.strip() for t in texts for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) > 3]
    freq = Counter(w for s in sentences for w in re.findall(r"[a-z]{4,}", s.lower()))
    ranked = sorted(sentences, key=lambda s: -sum(freq[w] for w in set(re.findall(r"[a-z]{4,}", s.lower()))))
    return " ".join(ranked[:n])


# ----------------------------------------------------------------------- QA
def answer_question(question: str, contexts: list[str]) -> dict:
    """Run extractive QA over candidate passages and keep the best-scoring span."""
    m = load("qa")
    if m.ok:
        try:
            best = None
            for ctx in contexts:
                out = m.pipe(question=question, context=ctx, truncation=True)
                if best is None or out["score"] > best["score"]:
                    best = {**out, "context": ctx}
            return {"answer": best["answer"], "score": float(best["score"]), "start": best["start"],
                    "end": best["end"], "context": best["context"], "source": "model"}
        except Exception as exc:
            err = str(exc)
    else:
        err = m.error
    # Fallback: return the passage sentence with the most question-word overlap.
    q = _content_words(question)
    best_sent, best_ctx, best_hits = "", "", 0
    for ctx in contexts:
        for s in re.split(r"(?<=[.!?])\s+|\n+", ctx):
            hits = len(q & _content_words(s))
            if hits > best_hits:
                best_sent, best_ctx, best_hits = s, ctx, hits
    start = best_ctx.find(best_sent) if best_sent else -1
    return {"answer": best_sent, "score": None, "start": start, "end": start + len(best_sent),
            "context": best_ctx, "source": "fallback", "keyword_hits": best_hits,
            "question_words": len(q), "error": err}


_STOP = set("""a an the is are was were be of to in on for and or with what which who how does do did
can could should would it its this that these those my your i me you about from at by as any there""".split())


def _stem(word: str) -> str:
    for suffix in ("ing", "age", "ed", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _content_words(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


# ---------------------------------------------------------------- zero-shot
ZERO_SHOT_LABELS = ["hydration", "oil control", "fragrance-free", "brightening", "soothing", "sun protection",
                    "exfoliation"]
HYPOTHESIS = "This product description is about {}."
_KEYWORDS = {
    "hydration": ["hydrat", "moistur", "hyaluron", "glycerin", "plump", "dry"],
    "oil control": ["oil", "shine", "matte", "zinc", "clay", "sebum"],
    "fragrance-free": ["fragrance-free", "no added fragrance"],
    "brightening": ["bright", "vitamin c", "ascorb", "glow", "dull"],
    "soothing": ["calm", "sooth", "centella", "comfort", "oat", "panthenol"],
    "sun protection": ["spf", "sunscreen", "zinc oxide", "titanium dioxide"],
    "exfoliation": ["acid", "bha", "aha", "texture", "exfoli"],
}


def zero_shot(text: str, labels: list[str] | None = None) -> dict:
    labels = labels or ZERO_SHOT_LABELS
    m = load("zero_shot")
    if m.ok:
        try:
            out = m.pipe(text, candidate_labels=labels, hypothesis_template=HYPOTHESIS, multi_label=True)
            return {"scores": dict(zip(out["labels"], map(float, out["scores"]))), "source": "model"}
        except Exception as exc:
            err = str(exc)
    else:
        err = m.error
    low = text.lower()
    matched = {lab: [k for k in _KEYWORDS.get(lab, []) if k in low] for lab in labels}
    return {"matches": matched, "source": "fallback", "error": err}
