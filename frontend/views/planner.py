import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from components import api, state, ui  # noqa: E402
from components.constants import MODES, TRAVELERS, WEATHER  # noqa: E402

st.title("Safe Route Planner")
st.caption("Compare route alternatives by predicted safety as well as travel time. "
           "Routes are a demo simulation; scores come from a model trained on synthetic data.")

with st.container(border=True):
    c1, c2 = st.columns(2)
    origin = state.place_input("From", "origin", "CBIT", c1)
    destination = state.place_input("To", "destination", "Gachibowli", c2)
    c3, c4, c5 = st.columns(3)
    mode = c3.selectbox("Travel mode", list(MODES), format_func=MODES.get, key="pl_mode")
    traveler = c4.selectbox("Traveler profile", list(TRAVELERS), format_func=TRAVELERS.get, key="pl_traveler")
    weather = c5.selectbox("Weather", list(WEATHER), format_func=WEATHER.get, key="pl_weather")
    c6, c7, _ = st.columns(3)
    travel_date = c6.date_input("Travel date", value=dt.date.today(), key="pl_date")
    travel_time = c7.time_input("Travel time", value=dt.time(22, 0), step=900, key="pl_time")
    go = st.button("Find the safest route", type="primary")

if go:
    if not origin or not destination:
        st.error("Enter both a start and a destination.")
    elif origin.strip().lower() == destination.strip().lower():
        st.error("Start and destination must be different.")
    else:
        request = {"origin": origin, "destination": destination, "travel_mode": mode, "traveler_type": traveler,
                   "travel_date": travel_date.isoformat(), "travel_time": travel_time.strftime("%H:%M"),
                   "weather": weather, "include_agent": True}
        try:
            with st.spinner("Generating routes, predicting risk and asking the agent..."):
                state.run_plan(request)
        except api.BackendError as exc:
            st.error(str(exc))

plan = state.get_plan()
if not plan:
    st.info("Choose a start, destination and traveler profile, then press **Find the safest route**.")
    ui.disclaimer()
    st.stop()

res, req = plan["result"], plan["request"]
ctx = res["context"]
routes = res["routes"]
rec = next(r for r in routes if r["recommended"])

st.markdown(f"### {ctx['origin']} → {ctx['destination']}")
st.caption(f"{ctx['mode']} · {ctx['traveler']} · {ctx['day']} {ctx['time']} · {ctx['weather']} "
           f"· about {ctx['straight_km']} km in a straight line")
if not (ctx["origin_known"] and ctx["destination_known"]):
    st.info("One of the places is not in the demo list, so it was placed at a stable simulated location. "
            "Distances and routes are illustrative.")
if ctx["mode_key"] == "walking" and rec["duration_min"] > 60:
    st.info(f"This is a long walk ({rec['duration_min']} min). Longer exposure lowers the predicted score. "
            "Try Public Transport or Car in the planner, or pick closer places.")

ui.agent_card(res["agent"])
st.markdown("&nbsp;")

cols = st.columns(len(routes))
for col, r in zip(cols, routes):
    col.markdown(ui.route_card(r, res["fastest_id"], res["safest_id"]), unsafe_allow_html=True)
st.caption(res["recommendation_basis"])

st.markdown("### Map and trade-off")
m1, m2 = st.columns([3, 2])
with m1:
    ui.plot(ui.route_map(res), "map_routes")
with m2:
    ui.plot(ui.tradeoff_scatter(routes), "scatter_tradeoff")
    ui.plot(ui.score_bars(routes), "bars_scores")

st.markdown("### Why these scores?")
st.caption("Each bar shows how many points a factor adds to (green) or removes from (red) the predicted score, "
           "compared with typical conditions. It is an approximation of the model's reasoning.")
tabs = st.tabs([r["name"] + (" ★" if r["recommended"] else "") for r in routes])
for tab, r in zip(tabs, routes):
    with tab:
        left, right = st.columns([3, 2])
        with left:
            ui.plot(ui.factor_bars(r["factors"], f"Why {r['name']} scored {r['safety_score']:.0f}/100"),
                    f"factors_{r['id']}")
        with right:
            pos = "".join(f"<li>{ui.esc(t)}</li>" for t in r["strengths"]) or "<li>None significant</li>"
            neg = "".join(f"<li>{ui.esc(t)}</li>" for t in r["concerns"]) or "<li>None significant</li>"
            st.markdown(f"**Positive factors**<ul>{pos}</ul>**Negative factors**<ul>{neg}</ul>",
                        unsafe_allow_html=True)
            f = r["features"]
            st.caption(f"Model inputs for this route: lighting {f['street_lighting']:.0f}, pedestrians "
                       f"{f['pedestrian_density']:.0f}, traffic {f['traffic_density']:.0f}, accidents "
                       f"{f['accident_frequency']:.1f}, road condition {f['road_condition']:.0f}, "
                       f"{f['road_type'].replace('_', ' ')} road (synthetic values).")

st.markdown("### Safety hotspots")
if res["hotspots"]:
    st.caption("Simulated junctions on these routes with the lowest predicted scores.")
    hcols = st.columns(len(res["hotspots"]))
    for i, (col, h) in enumerate(zip(hcols, res["hotspots"]), start=1):
        col.markdown(f'<div class="sr-hot"><b>{i}. {ui.esc(h["name"])}</b><br>'
                     f'{ui.pill(h["risk_level"])} &nbsp; {h["safety_score"]:.0f}/100<br>'
                     f'<span class="sr-note">Main factor: {ui.esc(h["main_factor"])}</span></div>',
                     unsafe_allow_html=True)
else:
    st.success("None of the simulated junctions on these routes fall in the elevated-risk range under the "
               "current conditions.")

n1, n2 = st.columns(2)
n1.page_link("views/what_if.py", label="Try What If? scenarios on this plan")
n2.page_link("views/agent.py", label="Ask the AI Agent about these results")
ui.disclaimer()
