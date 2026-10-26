from html import escape

import pandas as pd
import streamlit as st

from skinsense import nlp, ui

ui.hero("How It Works",
        "A plain-language look at what the AI does in SkinSense, which models we use, and where they fall short.")

st.subheader("The big picture")
st.markdown("""
| Feature | What does the work | Why |
|---|---|---|
| **Discover** | Simple filters + points (no AI) | Recommendations should be predictable and explainable |
| **AI category tags** | Zero-shot classification | Tags products with categories it was never trained on |
| **Review Insights** | Sentiment classification → summarization | Turns many reviews into quick themes |
| **Ingredient Q&A** | Keyword search → extractive question answering | Answers only with text found in the document |
""")

st.subheader("The models")
rows = [{"Feature": s.title, "Hugging Face task": s.task, "Model": s.model_id, "In plain words": s.plain_task}
        for s in nlp.MODELS.values()]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("We picked small, distilled models so the app runs on a normal laptop CPU and on free hosting. "
           "The trade-off is lower accuracy than large models.")

if st.button("Check model status (loads all models)"):
    with st.spinner("Loading models… the first time this downloads about 1.1 GB."):
        status = []
        for key, spec in nlp.MODELS.items():
            m = nlp.load(key)
            status.append({"Model": spec.model_id, "Status": "✅ loaded" if m.ok else "⚠️ fallback",
                           "Detail": "" if m.ok else m.error})
    st.dataframe(pd.DataFrame(status), hide_index=True, width="stretch")

st.subheader("Tokenization, live")
st.markdown("Models don't read words; they read **tokens**, small pieces of words mapped to numbers. "
            "Rare words like ingredient names get split into several pieces. Type something below to see how.")
sample = st.text_input("Text to tokenize", "Niacinamide serum feels lightweight and non-greasy.")
if sample.strip():
    with st.spinner("Loading tokenizer…"):
        tp = nlp.tokenize_preview("sentiment", sample, limit=80)
    if tp:
        st.markdown("".join(f'<span class="token">{escape(t)}</span>' for t in tp["tokens"]),
                    unsafe_allow_html=True)
        st.caption(f"{len(sample.split())} words → {tp['count']} tokens (DistilBERT tokenizer). "
                   "“##” means the piece continues the previous word.")
    else:
        st.markdown('<div class="fallback">The tokenizer couldn\'t be loaded (model unavailable), so a live '
                    'tokenization example can\'t be shown. Here\'s a plain whitespace split instead, which is '
                    '<b>not</b> what the model uses:</div>', unsafe_allow_html=True)
        st.markdown("".join(f'<span class="token">{escape(t)}</span>' for t in sample.split()),
                    unsafe_allow_html=True)

st.subheader("Reading model scores")
st.markdown("""
- **Sentiment confidence (0–1):** how sure the model is about POSITIVE vs NEGATIVE. Sure ≠ correct.
- **Zero-shot score (0–1):** how strongly the product *text* supports a label. It isn't a measure of results.
- **Q&A confidence (0–1):** how likely the copied span answers the question. Below 0.10 we flag it as
  low-confidence and show the passage. If the document never mentions your question's key words, we say "not found"
  without asking the model at all.
""")

st.subheader("Limitations & safety principles")
st.markdown("""
- **No diagnosis, no medical claims.** Skin type and goals are self-described. Nothing here says a product treats a condition.
- **Scores never decide suitability.** Sentiment and zero-shot scores describe text, not your skin.
- **Grounded facts only.** Prices, ingredients and claims come from catalog data or the document you provide.
  If the information isn't there, we say so.
- **Uploaded PDFs are untrusted.** We check size, type and page count, extract plain text only, and never follow
  instructions found inside a document.
- **Honest fallbacks.** If a model can't load (no internet, low disk space), the app switches to simple rules and
  shows a yellow banner. It never presents those results as AI predictions.
- **Sample data.** All brands, products, prices and reviews are fictional, for demo purposes.
- **Bias.** These models were trained mostly on general English web, news and movie-review text. They may misread
  skincare slang, non-English reviews, or experiences of underrepresented skin tones.
""")
