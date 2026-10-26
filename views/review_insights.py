from html import escape

import streamlit as st

from skinsense import data, nlp, ui

ui.hero("Review Insights",
        "See what reviewers love and complain about. An AI model labels each review's sentiment, then "
        "summarizes the positive and negative reviews separately.")

products = data.load_products()
reviews = data.load_reviews()
reviewed = products[products["product_id"].isin(reviews["product_id"])]

options = dict(zip(reviewed["name"] + " · " + reviewed["brand"], reviewed["product_id"]))
choice = st.selectbox("Choose a product with sample reviews", list(options))
pid = options[choice]
prod = reviewed.set_index("product_id").loc[pid]
prod_reviews = reviews[reviews["product_id"] == pid].reset_index(drop=True)

st.markdown(ui.badges(prod), unsafe_allow_html=True)
st.caption(f"{len(prod_reviews)} sample reviews · sample price ${prod['price_usd']:.2f}")

key = f"insights_{pid}"
if st.button("✨ Analyze reviews", type="primary"):
    texts = prod_reviews["review_text"].tolist()
    with st.spinner("Classifying sentiment (first run downloads ~270 MB)…"):
        sentiments = nlp.classify_sentiment(texts)
    pos = [t for t, s in zip(texts, sentiments) if s["label"] == "POSITIVE"]
    # Neutral reviews are usually mixed ("nice but..."), so their complaints go in the complaints group.
    neg = [t for t, s in zip(texts, sentiments) if s["label"] != "POSITIVE"]
    with st.spinner("Summarizing themes (first run downloads ~240 MB)…"):
        st.session_state[key] = {
            "sentiments": sentiments,
            "pos": nlp.summarize(pos) if pos else None,
            "neg": nlp.summarize(neg) if neg else None,
            "n_pos": len(pos), "n_neg": len(neg),
        }

res = st.session_state.get(key)
if not res:
    ui.empty_state("💬", "No analysis yet", "Click “Analyze reviews” to run sentiment and summarization.")
else:
    sents = res["sentiments"]
    if sents[0]["source"] == "fallback":
        ui.fallback_banner("sentiment", sents[0].get("error"))
    m1, m2, m3 = st.columns(3)
    m1.metric("Reviews analyzed", len(sents))
    m2.metric("Labelled positive", f"{res['n_pos']} of {len(sents)}")
    m3.metric("Avg sample star rating", f"{prod_reviews['rating'].mean():.1f} / 5")

    st.subheader("Key themes")
    c1, c2 = st.columns(2)
    for col, title, summ, n in [(c1, "👍 What people like", res["pos"], res["n_pos"]),
                                (c2, "👎 Complaints & mixed feedback", res["neg"], res["n_neg"])]:
        with col, st.container(border=True):
            st.markdown(f"**{title}** · from {n} review{'s' if n != 1 else ''}")
            if not summ:
                st.caption("No reviews in this group.")
                continue
            if summ["source"] == "fallback":
                ui.fallback_banner("summarization", summ.get("error"))
                st.markdown(f"_Most representative sentences (rule-based):_ {summ['summary']}")
            else:
                st.markdown(summ["summary"])
    st.caption("Summaries are AI-generated from the reviews and can miss details or merge separate points. "
               "Check the individual reviews below.")

    st.subheader("Individual reviews")
    for (_, r), s in zip(prod_reviews.iterrows(), sents):
        cls = {"POSITIVE": "sage", "NEGATIVE": "rose"}.get(s["label"], "amber")
        score = f" · confidence {s['score']:.2f}" if s["score"] is not None else " · rule-based, no score"
        st.markdown(
            f'<div class="review"><span class="badge {cls}">{s["label"]}{score}</span> '
            f'<span class="badge">{"★" * int(r["rating"])} {r["rating"]}/5 (reviewer)</span><br>'
            f'{escape(r["review_text"])}<br><small style="color:#7A6B65">— {escape(r["reviewer"])}</small></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="note">Sentiment tells you how a reviewer <i>felt</i>, not whether a product is '
                'right or safe for your skin. A happy review from someone with different skin may not apply to you.'
                '</div>', unsafe_allow_html=True)

    first_text = prod_reviews.loc[0, "review_text"]
    ui.transparency_panel(
        "sentiment", first_text, {"review": first_text, **sents[0]}, sents[0]["source"],
        ["Three labels (POSITIVE/NEUTRAL/NEGATIVE). Mixed reviews often land on NEUTRAL, or get forced into one side.",
         "The model was trained on tweets, not skincare reviews, so phrases like “broke me out” may be misread.",
         "A high confidence score means the model is sure, not that it is correct.",
         "Sentiment is never used to judge whether a product suits your skin."],
    )
    summ_raw = res["pos"] or res["neg"]
    if summ_raw:
        ui.transparency_panel(
            "summarization", " ".join(prod_reviews["review_text"]), summ_raw, summ_raw["source"],
            ["Small model: summaries can be repetitive or leave out points.",
             "Generative summaries can occasionally state things not in the reviews. Verify against the reviews.",
             "Reviews are grouped by the sentiment model first, so a mislabelled review ends up in the wrong summary."],
        )

st.divider()
st.subheader("Try it with your own review")
own = st.text_area("Paste or write a product review", placeholder="e.g. Feels lovely but the pump stopped working…")
if st.button("Classify sentiment"):
    if not own.strip():
        st.error("Please enter some review text first.")
    else:
        with st.spinner("Classifying…"):
            s = nlp.classify_sentiment([own])[0]
        if s["source"] == "fallback":
            ui.fallback_banner("sentiment", s.get("error"))
            st.markdown(f"**Rule-based label:** {s['label']} (no confidence score)")
        else:
            st.markdown(f"**Predicted label:** {s['label']} · **confidence:** {s['score']:.2f}")
            st.progress(s["score"])
