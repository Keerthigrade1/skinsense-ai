"""Render every page headlessly, click the main buttons, and fail on any exception.

Run: python scripts/ui_test.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

TIMEOUT = 300


def run(page: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "views" / f"{page}.py"), default_timeout=TIMEOUT).run()
    assert not at.exception, (page, at.exception)
    return at


def click(at: AppTest, label_start: str) -> AppTest:
    btn = next(b for b in at.button if b.label.startswith(label_start))
    btn.click().run()
    assert not at.exception, (label_start, at.exception)
    return at


def text_of(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown) + " ".join(str(x.value) for x in at.caption)


# Discover: default results, AI tags, then an empty state.
at = run("discover")
assert "Your top" in " ".join(s.value for s in at.subheader)
click(at, "✨ Generate AI tags")
assert len(at.get("progress")) > 0 or "Fallback mode" in text_of(at)
at.slider[0].set_value(5).run()
assert not at.exception and "No matches yet" in text_of(at)
print("discover ok")

# Review Insights: analyze, plus custom review (empty, then filled).
at = run("review_insights")
click(at, "✨ Analyze reviews")
assert "Key themes" in " ".join(s.value for s in at.subheader)
click(at, "Classify sentiment")
assert at.error, "expected an error for empty review text"
at.text_area[0].input("Feels lovely but the pump broke after a week.")
click(at, "Classify sentiment")
print("review_insights ok:", [m.value for m in at.markdown if "label" in m.value.lower()][-1])

# Ingredient Q&A: sample PDF answer, not-found, medical, catalog source.
at = run("ingredient_qa")
for q, expect in [("How should it be stored?", "Supporting source text"),
                  ("What is the price?", None),
                  ("Is it safe during pregnancy?", None)]:
    at.text_input(key="qa_q").input(q).run()
    assert not at.exception, (q, at.exception)
    if expect:
        assert expect in text_of(at), q
    else:
        assert any("Not found" in i.value for i in at.info), q
assert any("medical suitability" in w.value for w in at.warning)
at.radio[0].set_value("Catalog product data").run()
at.text_input(key="qa_q").input("Does it contain fragrance?").run()
assert not at.exception
print("ingredient_qa ok")

# How It Works: model status + tokenization.
at = run("how_it_works")
click(at, "Check model status")
print("how_it_works ok:", at.dataframe[-1].value["Status"].tolist())
print("\nALL UI CHECKS PASSED")
