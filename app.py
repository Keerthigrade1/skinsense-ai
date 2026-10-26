"""SkinSense AI entry point. Run: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="SkinSense AI", page_icon="🌿", layout="wide")

from skinsense import ui  # noqa: E402  (set_page_config must run first)

ui.setup_page()

pages = [
    st.Page("views/discover.py", title="Discover", icon="🌿", default=True),
    st.Page("views/review_insights.py", title="Review Insights", icon="💬"),
    st.Page("views/ingredient_qa.py", title="Ingredient Q&A", icon="📄"),
    st.Page("views/how_it_works.py", title="How It Works", icon="🔍"),
]
st.navigation(pages, position="top").run()
ui.footer()
