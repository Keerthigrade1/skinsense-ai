"""Shared Streamlit UI pieces: theme CSS, cards, banners, transparency panel."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from . import nlp

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');
:root {
  --cream:#FBF7F3; --sand:#F3EAE3; --blush:#E9C9BF; --rose:#B5786A; --sage:#7F9C84;
  --sage-soft:#E3ECE4; --ink:#3B302C; --muted:#7A6B65; --amber:#F6E7C8; --line:#EADDD4;
}
html, body, [class*="css"], .stMarkdown, p, li, label { font-family:'Inter',sans-serif; }
h1, h2, h3 { font-family:'Fraunces',serif !important; color:var(--ink); letter-spacing:-0.01em; }
.block-container { padding-top:2.2rem; max-width:1150px; }
.hero { background:linear-gradient(135deg,#F6E3DC 0%,#F3EAE3 55%,#E3ECE4 100%);
  border-radius:22px; padding:28px 30px; margin-bottom:18px; border:1px solid var(--line); }
.hero h1 { margin:0 0 6px 0; font-size:2.1rem; }
.hero p { margin:0; color:var(--muted); font-size:1.02rem; max-width:720px; }
.card { background:#fff; border:1px solid var(--line); border-radius:18px; padding:18px 18px 14px;
  height:100%; box-shadow:0 1px 2px rgba(59,48,44,.04); }
.card h4 { font-family:'Fraunces',serif; margin:2px 0 2px; font-size:1.12rem; color:var(--ink); }
.card .brand { color:var(--muted); font-size:.85rem; }
.card .price { font-weight:600; color:var(--rose); font-size:1.05rem; }
.card ul { padding-left:18px; margin:8px 0 4px; color:var(--ink); font-size:.86rem; }
.card .ing { font-size:.8rem; color:var(--muted); margin-top:6px; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.74rem; font-weight:500;
  margin:2px 4px 2px 0; background:var(--sand); color:var(--ink); }
.badge.sage { background:var(--sage-soft); color:#3F5E46; }
.badge.rose { background:#F6E3DC; color:#8A4F43; }
.badge.amber { background:var(--amber); color:#7A5A1E; }
.score-pill { float:right; background:var(--ink); color:#fff; border-radius:10px; padding:2px 9px;
  font-size:.78rem; font-weight:600; }
.note { background:var(--sage-soft); border-radius:14px; padding:12px 16px; color:#33493A; font-size:.9rem; }
.fallback { background:var(--amber); border-left:4px solid #D9A441; border-radius:12px; padding:12px 16px;
  color:#5E4512; font-size:.9rem; margin:6px 0 12px; }
.empty { text-align:center; padding:38px 20px; border:1.5px dashed var(--blush); border-radius:18px;
  color:var(--muted); background:#fff; }
.empty .big { font-size:2rem; }
.answer { background:#fff; border:1px solid var(--line); border-left:5px solid var(--sage); border-radius:14px;
  padding:14px 18px; font-size:1.05rem; }
.source { background:#fff; border:1px dashed var(--blush); border-radius:12px; padding:12px 16px;
  font-size:.88rem; color:var(--ink); line-height:1.55; }
.source mark { background:#F6E3DC; padding:1px 3px; border-radius:4px; }
.review { background:#fff; border:1px solid var(--line); border-radius:14px; padding:12px 16px; margin-bottom:8px; }
.token { display:inline-block; font-family:ui-monospace,monospace; font-size:.78rem; background:var(--sand);
  border-radius:6px; padding:1px 6px; margin:2px; }
.funnel { font-size:.86rem; color:var(--muted); }
.disclaimer { font-size:.8rem; color:var(--muted); border-top:1px solid var(--line); margin-top:28px; padding-top:12px; }
@media (max-width: 640px) { .hero { padding:20px; } .hero h1 { font-size:1.6rem; } }
</style>
"""

DISCLAIMER = (
    "SkinSense AI is a demo for product discovery only. It does not diagnose skin conditions or give medical "
    "advice, and no product here is claimed to treat any condition. All products, prices and reviews are "
    "fictional sample data. Patch test new products and consult a dermatologist for medical concerns."
)


def setup_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f'<div class="disclaimer">{escape(DISCLAIMER)}</div>', unsafe_allow_html=True)


def empty_state(icon: str, title: str, body: str) -> None:
    st.markdown(f'<div class="empty"><div class="big">{icon}</div><b>{escape(title)}</b><br>{escape(body)}</div>',
                unsafe_allow_html=True)


def fallback_banner(key: str, error: str | None) -> None:
    spec = nlp.MODELS[key]
    st.markdown(
        f'<div class="fallback"><b>Fallback mode: the AI model did not run.</b> '
        f'<code>{escape(spec.model_id)}</code> could not be loaded, so the result below comes from a simple '
        f'rule-based method and is <b>not</b> a model prediction. No confidence score is shown.<br>'
        f'<small>Reason: {escape(error or "unknown")}</small></div>',
        unsafe_allow_html=True,
    )


def badges(row: pd.Series) -> str:
    out = [f'<span class="badge">{escape(row["category"])}</span>']
    out.append('<span class="badge sage">Fragrance-free</span>' if row["fragrance_free"]
               else '<span class="badge amber">Contains fragrance</span>')
    if row["vegan"]:
        out.append('<span class="badge sage">Vegan</span>')
    out += [f'<span class="badge rose">{escape(c)}</span>' for c in row["concerns"]]
    return "".join(out)


def product_card(row: pd.Series, rank: int) -> None:
    reasons = "".join(f"<li>{escape(r)}</li>" for r in row["reasons"])
    st.markdown(
        f"""<div class="card">
        <span class="score-pill">#{rank} · {row['score']:.1f} pts</span>
        <div class="brand">{escape(row['brand'])}</div>
        <h4>{escape(row['name'])}</h4>
        <div class="price">${row['price_usd']:.2f} <span class="brand">· {row['size_ml']} ml · sample price</span></div>
        <div style="margin-top:6px">{badges(row)}</div>
        <ul>{reasons}</ul>
        <div class="ing"><b>Key ingredients (from catalog):</b> {escape(', '.join(row['key_ingredients']))}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def transparency_panel(key: str, sample_text: str, raw_output, source: str, limitations: list[str]) -> None:
    """Explain tokenization, task, output and limitations for one AI step."""
    spec = nlp.MODELS[key]
    with st.expander("🔍 AI transparency: what happened here?"):
        st.markdown(f"**Task:** {spec.plain_task}")
        if source == "model":
            st.markdown(f"**Model:** `{spec.model_id}` (Hugging Face, runs locally on CPU)")
            tp = nlp.tokenize_preview(key, sample_text)
            if tp:
                st.markdown(
                    f"**Tokenization:** before the model reads text, a tokenizer splits it into "
                    f"sub-word pieces called tokens. This input became **{tp['count']} tokens**"
                    + (f" (the model reads at most {tp['max_length']})." if tp["max_length"] and tp["max_length"] < 100000 else ".")
                    + " First few tokens:"
                )
                st.markdown("".join(f'<span class="token">{escape(t)}</span>' for t in tp["tokens"]),
                            unsafe_allow_html=True)
                st.caption("Symbols like “##” or “Ġ” are tokenizer markers for word pieces and spaces.")
        else:
            st.markdown("**Model:** none. A rule-based fallback produced this output, so there's no "
                        "tokenization or confidence score to show.")
        st.markdown("**Raw output:**")
        st.json(raw_output, expanded=False)
        st.markdown("**Limitations:**\n" + "\n".join(f"- {l}" for l in limitations))
