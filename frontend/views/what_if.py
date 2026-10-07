import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import api, state, ui  # noqa: E402
from components.constants import MODES, RISK_BG, RISK_COLORS, TRAVELERS, WEATHER  # noqa: E402

SAME = "Same as plan"

st.title("What If?")
st.caption("Change the conditions and see the model re-score every route, then the agent explains what moved.")

plan = state.require_plan()
req, base = plan["request"], plan["result"]
ctx = base["context"]
st.markdown(f"**Current plan:** {ctx['origin']} → {ctx['destination']} · {ctx['mode']} · "
            f"{ctx['traveler']} · {ctx['time']} · {ctx['weather']}")


# ----- controls ----------------------------------------------------------- #
def _preset(values: dict) -> None:
    for k, v in values.items():
        st.session_state[k] = v


def _reset() -> None:
    _preset({"wi_use_time": False, "wi_weather": SAME, "wi_traffic": 0, "wi_mode": SAME, "wi_traveler": SAME})
    st.session_state.pop("whatif", None)


for key, default in (("wi_use_time", False), ("wi_time", dt.time(22, 0)), ("wi_weather", SAME), ("wi_traffic", 0),
                     ("wi_mode", SAME), ("wi_traveler", SAME)):
    st.session_state.setdefault(key, default)

st.markdown("**Quick scenarios**")
p = st.columns(6)
p[0].button("At 6 PM", on_click=_preset, args=({"wi_use_time": True, "wi_time": dt.time(18, 0)},))
p[1].button("At 10 PM", on_click=_preset, args=({"wi_use_time": True, "wi_time": dt.time(22, 0)},))
p[2].button("Heavy rain", on_click=_preset, args=({"wi_weather": "heavy_rain"},))
p[3].button("Traffic +30%", on_click=_preset, args=({"wi_traffic": 30},))
p[4].button("As a cyclist", on_click=_preset, args=({"wi_traveler": "cyclist", "wi_mode": "cycling"},))
p[5].button("Driving instead", on_click=_preset, args=({"wi_mode": "car"},))

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    use_time = c1.checkbox("Change travel time", key="wi_use_time")
    new_time = c1.time_input("New time", key="wi_time", step=900, disabled=not use_time)
    new_weather = c2.selectbox("Weather", [SAME] + list(WEATHER), key="wi_weather",
                               format_func=lambda k: k if k == SAME else WEATHER[k])
    traffic = c3.slider("Traffic change (%)", -50, 100, key="wi_traffic")
    c4, c5, c6 = st.columns(3)
    new_mode = c4.selectbox("Travel mode", [SAME] + list(MODES), key="wi_mode",
                            format_func=lambda k: k if k == SAME else MODES[k])
    new_traveler = c5.selectbox("Traveler profile", [SAME] + list(TRAVELERS), key="wi_traveler",
                                format_func=lambda k: k if k == SAME else TRAVELERS[k])
    run, reset = c6.columns(2)
    go = run.button("Run What If", type="primary")
    reset.button("Reset", on_click=_reset)

if go:
    scenario = {"traffic_change_pct": traffic}
    if use_time:
        scenario["travel_time"] = new_time.strftime("%H:%M")
    if new_weather != SAME:
        scenario["weather"] = new_weather
    if new_mode != SAME:
        scenario["travel_mode"] = new_mode
    if new_traveler != SAME:
        scenario["traveler_type"] = new_traveler
    try:
        with st.spinner("Re-scoring every route..."):
            st.session_state["whatif"] = api.what_if(req, scenario)
    except api.BackendError as exc:
        st.error(str(exc))

res = st.session_state.get("whatif")
if res is None:
    st.info("Pick a quick scenario or set your own conditions, then press **Run What If**.")
else:
    st.markdown("### Result")
    if not res["changes"]:
        st.warning("Nothing was changed, so the scores are identical. Adjust at least one condition.")
    st.markdown(" ".join(f'<span class="sr-chip">{ui.esc(c)}</span>' for c in res["changes"]),
                unsafe_allow_html=True)
    st.markdown("&nbsp;")
    routes = res["routes"]
    cols = st.columns(len(routes))
    for col, r in zip(cols, routes):
        col.metric(r["name"], f"{r['baseline_score']:.0f} → {r['scenario_score']:.0f}",
                   f"{r['delta']:+.1f} points", delta_color="normal")

    rb, ra = res["recommended_before"], res["recommended_after"]
    if res["recommendation_changed"]:
        st.markdown(f'<div class="sr-banner" style="background:{RISK_BG["Medium"]};color:{RISK_COLORS["Medium"]}">'
                    f'<b>The recommendation changes:</b> Route {rb} → Route {ra} under this scenario.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sr-banner" style="background:{RISK_BG["Low"]};color:{RISK_COLORS["Low"]}">'
                    f'The recommendation stays <b>Route {ra}</b> under this scenario.</div>',
                    unsafe_allow_html=True)

    ui.agent_card(res["agent"], "SafeRoute AI Agent: what changed and why")
    st.markdown("&nbsp;")

    left, right = st.columns([3, 2])
    with left:
        ui.plot(ui.before_after_bars(routes), "wi_bars")
    with right:
        df = pd.DataFrame([{"Route": r["name"],
                            "Score": f"{r['baseline_score']:.0f} \u2192 {r['scenario_score']:.0f}",
                            "Risk": f"{r['baseline_risk']} \u2192 {r['scenario_risk']}",
                            "Minutes": f"{r['baseline_duration']} \u2192 {r['scenario_duration']}"}
                           for r in routes])
        st.table(df.set_index("Route"))

    st.markdown("### What drove the change?")
    st.caption("Points each changed condition added or removed (exact Shapley attribution; the bars add up to "
               "the total change).")
    choice = st.radio("Route", [r["id"] for r in routes], index=[r["id"] for r in routes].index(rb),
                      format_func=lambda i: f"Route {i}", horizontal=True, key="wi_driver_route")
    sel = next(r for r in routes if r["id"] == choice)
    if sel["drivers"]:
        ui.plot(ui.driver_bars(sel["drivers"], f"{sel['name']}: {sel['baseline_score']:.0f} → "
                                               f"{sel['scenario_score']:.0f}"), "wi_drivers")
    else:
        st.write("No input changed for this route.")

st.markdown("### Safety across the day")
st.caption("Predicted score of each route for every hour (same date, weather and profile as your plan).")
if "time_profile" not in st.session_state:
    try:
        with st.spinner("Computing the 24-hour profile..."):
            st.session_state["time_profile"] = api.time_profile(req)
    except api.BackendError as exc:
        st.error(str(exc))
prof = st.session_state.get("time_profile")
if prof:
    ui.plot(ui.time_profile_chart(prof, ctx["hour"]), "wi_profile")
ui.disclaimer()
