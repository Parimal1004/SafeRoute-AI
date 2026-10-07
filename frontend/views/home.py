import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components import api, ui  # noqa: E402

st.markdown("""
<div class="sr-hero">
  <div class="sr-eyebrow">SDG 11 &middot; Sustainable Cities and Communities</div>
  <h1>SafeRoute AI</h1>
  <p>An AI-powered route recommendation system that combines machine learning and an AI agent to help you
  choose routes based on predicted safety risk rather than travel time alone.</p>
  <div class="sr-quote">The shortest route isn't always the safest route.</div>
</div>
""", unsafe_allow_html=True)

if st.button("Start planning", type="primary"):
    st.switch_page("views/planner.py")

st.markdown("### How it works")
ui.step_strip([("User input", "origin, time, profile"), ("Route data", "3 alternatives"),
               ("ML risk prediction", "Random Forest"), ("Explainability", "factor effects"),
               ("AI agent", "reads the numbers"), ("Recommendation", "personalised")])

st.markdown("&nbsp;")
c1, c2, c3 = st.columns(3)
c1.markdown(ui.info_card("Predicted risk, not just distance",
                         "Every route gets a 0-100 safety score from a model that looks at lighting, traffic, "
                         "accident history, pedestrian activity, weather, time of day and who is travelling."),
            unsafe_allow_html=True)
c2.markdown(ui.info_card("Explained, not a black box",
                         "See which factors lifted or lowered each score, and run What If? scenarios "
                         "(night, rain, more traffic, a different profile) to watch the recommendation change."),
            unsafe_allow_html=True)
c3.markdown(ui.info_card("An agent that uses real numbers",
                         "The AI agent re-runs the model for your plan and answers questions from those "
                         "predictions. It never claims a route is safe."), unsafe_allow_html=True)

st.markdown("### The model at a glance")
try:
    info = api.model_info()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model", "Random Forest")
    m2.metric("Training rows (synthetic)", f"{info['trained_rows']:,}")
    m3.metric("Hold-out R²", info["metrics_holdout"]["r2"])
    m4.metric("Hold-out MAE (points)", info["metrics_holdout"]["mae"])
except api.BackendError as exc:
    st.warning(f"Model details unavailable right now. {exc}")

ui.disclaimer()
