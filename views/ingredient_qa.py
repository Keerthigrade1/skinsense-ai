from html import escape

import streamlit as st

from skinsense import data, nlp, pdf_qa, ui

ui.hero("Ingredient Q&A",
        "Ask questions about a product information sheet. Answers are copied straight from the document, "
        "with the supporting passage shown, and we tell you when the answer isn't there.")

source = st.radio("Document source",
                  ["Sample product sheet (PDF)", "Upload a PDF", "Catalog product data"], horizontal=True)

text, label = "", ""
if source == "Sample product sheet (PDF)":
    try:
        ext = pdf_qa.extract_text(data.SAMPLE_PDF.read_bytes())
        text, label = ext.text, f"{data.SAMPLE_PDF.name} · {ext.pages} page(s)"
    except (OSError, pdf_qa.PdfError) as exc:
        st.error(f"Couldn't load the sample PDF: {exc}. Run `python scripts/make_sample_pdf.py`.")
elif source == "Upload a PDF":
    up = st.file_uploader("Upload a product information PDF (max 10 MB, text-based PDFs only)", type=["pdf"])
    st.caption("Uploaded files are treated as untrusted: we only extract plain text, nothing in the file is "
               "executed, and the file isn't stored after you leave.")
    if up is not None:
        try:
            with st.spinner("Extracting text…"):
                ext = pdf_qa.extract_text(up.getvalue())
            for w in ext.warnings:
                st.warning(w)
            text, label = ext.text, f"{up.name} · {ext.pages} page(s)"
        except pdf_qa.PdfError as exc:
            st.error(f"We couldn't use this file: {exc}")
else:
    products = data.load_products()
    name = st.selectbox("Product", products["name"])
    row = products[products["name"] == name].iloc[0]
    text, label = data.product_fact_sheet(row), f"Catalog entry for {name}"

if not text:
    ui.empty_state("📄", "No document loaded", "Pick the sample sheet, upload a PDF, or choose a catalog product.")
    ui.footer()
    st.stop()

with st.expander(f"Document loaded: {label} · {len(text.split())} words. Click to preview the extracted text"):
    st.text(text[:4000] + ("…" if len(text) > 4000 else ""))

SUGGESTED = ["What are the ingredients?", "How do I use it?", "Does it contain fragrance?",
             "How should it be stored?", "What is the price?"]
st.markdown("**Try a question:**")
chip_cols = st.columns(len(SUGGESTED))
for col, q in zip(chip_cols, SUGGESTED):
    if col.button(q, width="stretch"):
        st.session_state["qa_q"] = q
question = st.text_input("Your question", key="qa_q", placeholder="e.g. What size is the bottle?")

if question.strip():
    with st.spinner("Searching the document (first run downloads the Q&A model, ~260 MB)…"):
        res = pdf_qa.ask(question, text)

    if res["source"] == "fallback":
        ui.fallback_banner("qa", res.get("error"))
    if res["medical"]:
        st.warning("This sounds like a question about medical suitability. SkinSense can't advise on that, "
                   "whatever a document says. Please ask a dermatologist, doctor or pharmacist.", icon="🩺")

    status = res["status"]
    if status == "answer":
        st.markdown(f'<div class="answer">{escape(res["answer"])}</div>', unsafe_allow_html=True)
        if res["score"] is not None:
            st.caption(f"Model confidence: {res['score']:.2f} · answer copied word-for-word from the document")
        else:
            st.caption(f"Rule-based match: this sentence shares {res['keyword_hits']} keywords with your question")
    elif status == "low":
        st.warning(f"**Low-confidence answer:** “{res['answer']}” (confidence {res['score']:.2f}). The model isn't "
                   "sure, so please read the highlighted passage below and judge for yourself.", icon="🤔")
    else:
        why = ("None of the document's text mentions the key words in your question"
               if res["source"] == "guard" else "The model found no answer span in the most relevant passages")
        st.info(f"**Not found in this document.** {why}, so we're not guessing. Try rephrasing, or check "
                "the full product label.", icon="🔎")

    if status in ("answer", "low") and res["context"]:
        ctx, s, e = res["context"], res["start"], res["end"]
        hl = (escape(ctx[:s]) + "<mark>" + escape(ctx[s:e]) + "</mark>" + escape(ctx[e:])) if s >= 0 else escape(ctx)
        st.markdown("**Supporting source text** (from the document; the highlighted part is the answer)")
        st.markdown(f'<div class="source">{hl}</div>', unsafe_allow_html=True)

    ui.transparency_panel(
        "qa", question + " " + (res.get("context") or ""),
        {k: res.get(k) for k in ["status", "answer", "score", "start", "end", "source", "keyword_overlap",
                                  "chunks_searched", "chunks_total"]},
        "model" if res["source"] == "model" else res["source"],
        [f"The document is split into passages of about {pdf_qa.CHUNK_WORDS} words. The {pdf_qa.TOP_K_CHUNKS} "
         "passages sharing the most words with your question are sent to the model.",
         "Extractive: the model only copies a span from the passage. It cannot combine facts or answer yes/no "
         "questions directly.",
         "If no passage shares a key word with your question, the model isn't run and we report “not found”. "
         "Extractive models always return *some* text, even for unrelated questions.",
         f"Answers below {pdf_qa.CONFIDENT_SCORE} confidence are marked low-confidence, with the passage shown so "
         "you can check.",
         "Ingredient info is shown only if it's in the document. We never add ingredients from elsewhere.",
         "The document is untrusted input. Its text is only searched, never followed as instructions."],
    )
