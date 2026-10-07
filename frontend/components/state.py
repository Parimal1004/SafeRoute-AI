"""Session-state helpers shared by the pages."""
from __future__ import annotations

from typing import List, Optional

import streamlit as st

from components import api
from components.constants import DEMO_PLAN, FALLBACK_PLACES


def get_places() -> List[str]:
    try:
        return api.places()
    except api.BackendError:
        return FALLBACK_PLACES


def place_input(label: str, key: str, default: str, container=None) -> str:
    """Pick a known demo place or type any other place name."""
    container = container or st
    options = get_places() + ["Other (type a place)"]
    idx = options.index(default) if default in options else 0
    choice = container.selectbox(label, options, index=idx, key=f"{key}_choice")
    if choice == "Other (type a place)":
        return container.text_input(f"{label} (custom place)", value="", key=f"{key}_custom",
                                    placeholder="e.g. Kondapur or 17.44, 78.38").strip()
    return choice


def get_plan() -> Optional[dict]:
    return st.session_state.get("plan")


def set_plan(request: dict, result: dict) -> None:
    st.session_state["plan"] = {"request": request, "result": result}
    for k in ("chat", "whatif", "time_profile"):
        st.session_state.pop(k, None)


def run_plan(request: dict) -> dict:
    result = api.recommend(request)
    set_plan(request, result)
    return result


def require_plan() -> dict:
    """Return the current plan or show a prompt (with a demo shortcut) and stop the page."""
    plan = get_plan()
    if plan:
        return plan
    st.info("No route plan yet. Plan a route first, or load the demo (CBIT to Gachibowli, walking, student, 10 PM).")
    c1, c2 = st.columns(2)
    if c1.button("Load demo plan", type="primary"):
        try:
            with st.spinner("Scoring routes..."):
                run_plan(dict(DEMO_PLAN))
            st.rerun()
        except api.BackendError as exc:
            st.error(str(exc))
    c2.page_link("views/planner.py", label="Go to the Safe Route Planner")
    st.stop()
