# SkinSense AI 🌿

**Live demo:** _coming soon_

An AI-assisted skincare **product discovery** MVP built with Streamlit and Hugging Face Transformers.
Users describe their skin type, goals and budget, get transparent recommendations from a sample catalog,
explore review sentiment and summaries, and ask questions about product information PDFs, with every AI
step explained in plain language.

> **Not medical advice.** SkinSense AI does not diagnose skin conditions or claim any product treats one.
> All brands, products, prices and reviews are **fictional sample data**.

## Features

| Page | What it does | How |
|---|---|---|
| **Discover** | Personalized matches by skin type, goals, budget, product type, fragrance-free and vegan | Rule-based filters + points (no AI), with a visible filter funnel and per-product reasons |
| | AI category tags (hydration, oil control, fragrance-free…) | Zero-shot classification on product descriptions; catalog data overrides the model when they disagree |
| **Review Insights** | Sentiment label + confidence per review; "what people like" vs "complaints & mixed" summaries | Text classification → summarization of each group |
| **Ingredient Q&A** | Ask questions about an uploaded PDF, the bundled sample sheet, or a catalog entry | PyPDF extraction → keyword retrieval → extractive QA, with supporting text highlighted and an honest "not found" |
| **How It Works** | Models used, live tokenization demo, how to read scores, limitations and safety principles | |

Every AI result has an **AI transparency** panel showing the task, model, tokenization, raw output and limitations.

## Models (small, CPU-friendly; about 1.4 GB total, downloaded on first use)

| Task | Model |
|---|---|
| Sentiment (positive / neutral / negative) | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Summarization | `Falconsai/text_summarization` (T5-small) |
| Extractive question answering | `deepset/tinyroberta-squad2` |
| Zero-shot classification | `cross-encoder/nli-distilroberta-base` |

## Setup and run

Requires Python 3.12.

```bash
cd skinsense-ai
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt   # includes the small CPU-only PyTorch build
streamlit run app.py
```

Open http://localhost:8501. The first time you use each AI feature, its model downloads (you'll see a spinner).
After that, everything runs offline.

### Fallback mode

If a model can't download or load (no internet, low disk space, missing dependency), that feature switches to a
simple rule-based method and shows a **yellow "Fallback mode: the AI model did not run" banner** with the reason.
Fallback results never show confidence scores and are never presented as model output.
To demo it on purpose:

```bash
# Windows PowerShell
$env:SKINSENSE_FORCE_FALLBACK="1"; streamlit run app.py
# macOS/Linux
SKINSENSE_FORCE_FALLBACK=1 streamlit run app.py
```

### Tests

```bash
python scripts/smoke_test.py   # every feature end-to-end, no UI
python scripts/ui_test.py      # renders every page headlessly and clicks the main buttons
```

Both also pass with `SKINSENSE_FORCE_FALLBACK=1`.

## Project structure

```
app.py                  # entry point + top navigation
views/                  # one file per page
skinsense/
  data.py               # loads the sample catalog and reviews
  recommender.py        # transparent filtering + scoring rules
  nlp.py                # Hugging Face pipelines + honest fallbacks
  pdf_qa.py             # safe PDF extraction, chunking, retrieval, answer gating
  ui.py                 # theme, cards, banners, transparency panel
data/                   # products.csv, reviews.csv (fictional)
sample_docs/            # sample product sheet PDF for the Q&A demo
scripts/                # sample PDF generator, smoke + UI tests
```

## Safety and accuracy decisions

- **Ranking uses no AI.** Recommendations come from catalog fields and simple points, so every result can be explained.
- **Scores never imply suitability.** Sentiment and zero-shot scores describe text, not how a product will work on your skin.
- **Grounded facts only.** Prices, ingredients and claims come only from catalog data or the provided document.
- **"Not found" over guessing.** Extractive QA models always return *some* span. We skip the model when no passage
  shares a keyword with the question, flag answers below 0.10 confidence as low-confidence (with the passage shown), and
  add a "can't advise" notice to medical-suitability questions.
- **Uploaded PDFs are untrusted.** Size (10 MB), page (50) and character limits, PDF signature check, password-protected
  files rejected, text-only extraction, HTML-escaped display, and the text is only searched, never followed as instructions.
- **Honest fallbacks** are clearly labelled, as described above.

## Known limitations

- Small models trade accuracy for speed: summaries can be extractive or miss points, and the sentiment model was
  trained on tweets, not skincare reviews.
- Zero-shot tags on short marketing copy can be noisy (for example, a citrus cleanser scoring high on "soothing").
- Q&A handles list-style answers (like full ingredient lists) poorly. These usually show as low-confidence with the full passage.
- Scanned (image-only) PDFs aren't supported (no OCR).
- English only.
