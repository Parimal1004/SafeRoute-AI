"""SafeRoute AI - Streamlit entry point.   Run:  streamlit run frontend/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from components import api, ui  # noqa: E402

st.set_page_config(page_title="SafeRoute AI", page_icon=":material/route:", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()

pages = [
    st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("views/planner.py", title="Safe Route Planner", icon=":material/alt_route:"),
    st.Page("views/what_if.py", title="What If?", icon=":material/science:"),
    st.Page("views/insights.py", title="Safety Insights", icon=":material/insights:"),
    st.Page("views/agent.py", title="AI Agent", icon=":material/smart_toy:"),
    st.Page("views/about.py", title="About", icon=":material/info:"),
]
nav = st.navigation(pages)


@st.cache_data(ttl=30, show_spinner=False)
def _status():
    try:
        return api.health(timeout=6)
    except api.BackendError:
        return None


with st.sidebar:
    h = _status()
    if h is None:
        st.markdown("**Backend:** not reachable  \n"
                    "<span class='sr-note'>If it is hosted on a free plan it may be waking up. "
                    "Try again in a minute.</span>", unsafe_allow_html=True)
    else:
        agent = (f"LLM ({h['llm']['model']})" if h["agent_mode"] == "llm" else "offline agent")
        st.markdown(f"**Backend:** online  \n<span class='sr-note'>Agent: {agent}</span>", unsafe_allow_html=True)
    st.markdown("<span class='sr-note'>Demo data is synthetic. Not a guarantee of real-world safety.</span>",
                unsafe_allow_html=True)

nav.run()
