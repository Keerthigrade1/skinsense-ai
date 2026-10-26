"""End-to-end check of every feature without the UI.

Run: python scripts/smoke_test.py
Set SKINSENSE_FORCE_FALLBACK=1 to test the no-model fallback path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skinsense import data, nlp, pdf_qa  # noqa: E402
from skinsense.recommender import Preferences, recommend  # noqa: E402

products = data.load_products()
rec = recommend(products, Preferences("dry", ["dryness"], 30, [], True, False))
print("Funnel:", rec.funnel)
print("Top picks:", rec.results[["name", "score"]].values.tolist())
assert not rec.results.empty and rec.results["fragrance_free"].all()
assert recommend(products, Preferences("oily", ["dryness"], 5)).results.empty

reviews = data.load_reviews()
texts = reviews[reviews.product_id == "P004"].review_text.tolist()
sents = nlp.classify_sentiment(texts)
print("\nSentiment:", [(s["label"], s["score"] and round(s["score"], 3), s["source"]) for s in sents])
pos = [t for t, s in zip(texts, sents) if s["label"] == "POSITIVE"]
neg = [t for t, s in zip(texts, sents) if s["label"] != "POSITIVE"]
print("Positive summary:", nlp.summarize(pos))
print("Negative summary:", nlp.summarize(neg))

desc = products.set_index("product_id").loc["P003", "description"]
print("\nZero-shot (P003):", nlp.zero_shot(desc))

ext = pdf_qa.extract_text(data.SAMPLE_PDF.read_bytes())
print(f"\nPDF: {ext.pages} page(s), {len(ext.text.split())} words, warnings={ext.warnings}")
expected = {"What are the ingredients?": ("answer", "low"), "How should it be stored?": ("answer",),
            "How do I use it?": ("answer",), "What is the price?": ("not_found",),
            "Who is the CEO of the company?": ("not_found",), "Is it safe during pregnancy?": ("not_found",)}
for q, ok in expected.items():
    r = pdf_qa.ask(q, ext.text)
    print(f"Q: {q}\n   {r['status']} score={r['score'] and round(r['score'], 3)} medical={r['medical']} "
          f"answer={r['answer']!r}")
    if not nlp.FORCE_FALLBACK:
        assert r["status"] in ok, (q, r["status"])

try:
    pdf_qa.extract_text(b"not a pdf")
except pdf_qa.PdfError as e:
    print("\nRejected non-PDF:", e)

print("\nTokens:", nlp.tokenize_preview("sentiment", "Niacinamide serum"))
print("\nModel sources:", {k: ("model" if nlp.load(k).ok else "fallback") for k in nlp.MODELS})
