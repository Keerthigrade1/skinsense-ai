import streamlit as st

from skinsense import data, nlp, ui
from skinsense.recommender import Preferences, recommend

ui.hero("Find products that fit your routine",
        "Tell us about your skin and budget. We match products from our sample catalog using simple, "
        "visible rules, so you can see exactly why each one appears.")

products = data.load_products()

with st.container(border=True):
    c1, c2, c3 = st.columns([1, 1.4, 1])
    skin_type = c1.selectbox("Skin type", data.SKIN_TYPES, index=3,
                             help="Self-described. We don't assess or diagnose skin.")
    concerns = c2.multiselect("Skincare goals", data.CONCERNS, default=["dryness"],
                              help="Cosmetic goals only. For medical concerns, please see a dermatologist.")
    budget = c3.slider("Max budget per product ($)", 5, 50, 30)
    c4, c5, c6 = st.columns([1.4, 0.8, 0.8])
    cats = c4.multiselect("Product types (optional)", data.categories())
    ff = c5.toggle("Fragrance-free only")
    vegan = c6.toggle("Vegan only")

prefs = Preferences(skin_type, concerns, float(budget), cats, ff, vegan)
rec = recommend(products, prefs)

st.markdown(
    '<div class="funnel">How we filtered: '
    + " → ".join(f"{step} <b>({n})</b>" for step, n in rec.funnel)
    + "</div>",
    unsafe_allow_html=True,
)
st.write("")

if rec.results.empty:
    blocker = next((step for step, n in rec.funnel if n == 0), "your filters")
    ui.empty_state("🌱", "No matches yet",
                   f"Nothing in the sample catalog passed the “{blocker}” step. "
                   "Try raising your budget, removing a product type, or choosing fewer goals.")
    ui.footer()
    st.stop()

st.subheader(f"Your top {len(rec.results)} matches")
cols = st.columns(3)
for i, row in rec.results.iterrows():
    with cols[i % 3]:
        ui.product_card(row, i + 1)
        st.write("")

with st.expander("How is the score calculated?"):
    st.markdown(
        "- **Hard filters first:** skin type listed by the catalog, price within budget, and your optional type, "
        "fragrance-free and vegan choices.\n"
        "- **Then points:** +3 for each of your goals the product is tagged for, plus its sample star rating "
        "(for example 4.5 stars = +4.5).\n"
        "- **Ties** go to the cheaper product.\n"
        "- **No AI is used to rank products.** That keeps recommendations predictable and explainable. "
        "Tags come from catalog data, not from the model."
    )

# ------------------------------------------------------- zero-shot tagging
st.divider()
st.subheader("AI category tags")
st.caption("Zero-shot classification reads each product description and scores how well it matches "
           "categories like hydration or oil control. These scores describe the text, not how a product "
           "will work on your skin.")

result_ids = tuple(rec.results["product_id"])
if st.button("✨ Generate AI tags for these matches", type="primary"):
    with st.spinner("Loading the zero-shot model (first run downloads ~330 MB) and scoring descriptions…"):
        st.session_state["zs"] = {
            "ids": result_ids,
            "out": {pid: nlp.zero_shot(products.set_index("product_id").loc[pid, "description"])
                    for pid in result_ids},
        }

zs = st.session_state.get("zs")
if not zs or zs["ids"] != result_ids:
    ui.empty_state("🏷️", "No AI tags yet", "Click the button above to categorize the products shown.")
else:
    first = next(iter(zs["out"].values()))
    if first["source"] == "fallback":
        ui.fallback_banner("zero_shot", first.get("error"))
    by_id = products.set_index("product_id")
    cols = st.columns(3)
    for i, pid in enumerate(zs["ids"]):
        row, out = by_id.loc[pid], zs["out"][pid]
        with cols[i % 3], st.container(border=True):
            st.markdown(f"**{row['name']}**")
            if out["source"] == "model":
                for label, score in list(out["scores"].items())[:3]:
                    st.progress(score, text=f"{label} · {score:.2f}")
                ff_score = out["scores"].get("fragrance-free", 0)
                if ff_score >= 0.5 and not row["fragrance_free"]:
                    st.warning("Model leaned “fragrance-free”, but catalog data says this contains added "
                               "fragrance. Catalog data wins.", icon="⚠️")
                elif ff_score < 0.5 and row["fragrance_free"]:
                    st.caption("Catalog data confirms: fragrance-free (the model's score for this was low).")
            else:
                hits = {k: v for k, v in out["matches"].items() if v}
                st.markdown("Keyword matches (rule-based): " + (", ".join(hits) if hits else "none"))
    sample_pid = zs["ids"][0]
    ui.transparency_panel(
        "zero_shot", by_id.loc[sample_pid, "description"], zs["out"][sample_pid], first["source"],
        ["Scores show how well the *description text* matches a label. They are not evidence of results, "
         "safety or suitability for your skin.",
         "Labels are scored independently (multi-label), so several can be high at once.",
         "Short marketing descriptions give the model little to go on, so treat low-confidence tags with caution.",
         "When the model and the catalog disagree (e.g. fragrance), the catalog data is shown as the source of truth."],
    )
