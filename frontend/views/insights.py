import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import api, state, ui  # noqa: E402
from components.constants import FEATURE_LABELS, MODES, SCORE_NOTE, TRAVELERS, WEATHER  # noqa: E402

st.title("Safety Insights")
st.caption("How the model works, what drives the scores, and where the simulated risk concentrates.")

tab_model, tab_factors, tab_hot, tab_play = st.tabs(
    ["Model and feature importance", "Route risk factors", "Safety hotspots", "Model playground"])

# ---------------------------------------------------------------- model --- #
with tab_model:
    try:
        info = api.model_info()
    except api.BackendError as exc:
        st.error(str(exc))
        st.stop()
    m = info["metrics_holdout"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "Random Forest")
    c2.metric("Training rows (synthetic)", f"{info['trained_rows']:,}")
    c3.metric("Hold-out R²", m["r2"])
    c4.metric("Hold-out MAE", f"{m['mae']} pts")

    kind = st.radio("Importance type", ["Random Forest importance", "Permutation importance"], horizontal=True)
    imp = info["feature_importance"] if kind.startswith("Random") else info["permutation_importance"]
    ui.plot(ui.importance_bar(imp, 12, "Which inputs matter most to the model"), "imp_bar")
    st.caption("Random Forest importance is how much each feature reduces prediction error across the trees "
               "(one-hot columns are summed per feature). Permutation importance is how much the score "
               "error grows when a feature is shuffled, measured on held-out data. "
               "These importances describe the synthetic world the model was trained on.")

    st.markdown("**Model comparison (same hold-out data)**")
    comp = pd.DataFrame(info["model_comparison"]).T.rename(columns={"r2": "R²", "mae": "MAE", "rmse": "RMSE"})
    st.table(comp)
    th = info["thresholds"]
    st.markdown(f"**Risk bands:** {th['low_risk_min']}-100 Low risk · {th['high_risk_max']}-"
                f"{th['low_risk_min'] - 1} Medium risk · 0-{th['high_risk_max'] - 1} High risk. "
                f"{SCORE_NOTE}")

# -------------------------------------------------------------- factors --- #
with tab_factors:
    plan = state.get_plan()
    if not plan:
        st.info("Plan a route first to see the factors behind each route's score.")
        st.page_link("views/planner.py", label="Go to the Safe Route Planner")
    else:
        res = plan["result"]
        st.caption(f"{res['context']['origin']} → {res['context']['destination']} · "
                   f"{res['context']['traveler']} · {res['context']['time']}. Green adds to the score, red "
                   "removes from it, compared with typical conditions.")
        cols = st.columns(len(res["routes"]))
        for col, r in zip(cols, res["routes"]):
            with col:
                st.markdown(f"**{r['name']}**" + (" ★" if r["recommended"] else "")
                            + f" &nbsp; {r['safety_score']:.0f}/100 &nbsp; {ui.pill(r['risk_level'])}",
                            unsafe_allow_html=True)
                ui.plot(ui.factor_bars(r["factors"]), f"ins_factors_{r['id']}")

# -------------------------------------------------------------- hotspots -- #
with tab_hot:
    st.caption("Twelve synthetic demo junctions on an abstract grid (not real places). "
               "Change the conditions to see how the predicted risk shifts.")
    h1, h2, h3, h4 = st.columns(4)
    hour = h1.slider("Hour of day", 0, 23, 22, key="hs_hour")
    trav = h2.selectbox("Traveler", list(TRAVELERS), format_func=TRAVELERS.get, key="hs_trav")
    mode = h3.selectbox("Mode", list(MODES), format_func=MODES.get, key="hs_mode")
    wx = h4.selectbox("Weather", list(WEATHER), format_func=WEATHER.get, key="hs_wx")
    try:
        data = api.hotspots(float(hour), trav, mode, wx)
    except api.BackendError as exc:
        st.error(str(exc))
        data = None
    if data:
        j = data["junctions"]
        left, right = st.columns([3, 2])
        with left:
            ui.plot(ui.hotspot_map(j), "hs_map")
        with right:
            st.markdown("**Highest predicted risk**")
            for i, x in enumerate(j[:5], start=1):
                st.markdown(f'<div class="sr-hot"><b>{i}. {ui.esc(x["name"])}</b> &nbsp; '
                            f'{ui.pill(x["risk_level"])} {x["safety_score"]:.0f}/100<br>'
                            f'<span class="sr-note">Main factor: {ui.esc(x["main_factor"])}</span></div>',
                            unsafe_allow_html=True)
        st.caption(data["note"])

# ------------------------------------------------------------ playground -- #
with tab_play:
    st.caption("Set the conditions of an imaginary route yourself and ask the trained model for its prediction.")
    a, b, c = st.columns(3)
    with a:
        lighting = st.slider("Street lighting", 0, 100, 70)
        pedestrians = st.slider("Pedestrian density", 0, 100, 50)
        road_cond = st.slider("Road condition", 0, 100, 65)
        crime = st.slider("Crime index (synthetic)", 0, 100, 30)
    with b:
        accidents = st.slider("Accident frequency", 0.0, 10.0, 3.0, 0.5)
        traffic = st.slider("Traffic density", 0, 100, 45)
        inter = st.slider("Intersections per km", 0, 25, 8)
        hour = st.slider("Hour of day", 0, 23, 22, key="pg_hour")
    with c:
        weather = st.selectbox("Weather", list(WEATHER), format_func=WEATHER.get, key="pg_wx")
        traveler = st.selectbox("Traveler", list(TRAVELERS), format_func=TRAVELERS.get, key="pg_trav")
        mode = st.selectbox("Mode", list(MODES), format_func=MODES.get, key="pg_mode")
        road_type = st.selectbox("Road type", ["arterial", "commercial", "residential", "service_lane", "highway"],
                                 format_func=lambda x: x.replace("_", " "))
        distance = st.slider("Distance (km)", 0.5, 15.0, 2.0, 0.5)
    rain = {"clear": 0, "cloudy": 0, "light_rain": 3, "heavy_rain": 15, "fog": 0}[weather]
    vis_mult = {"clear": 1, "cloudy": 0.95, "light_rain": 0.7, "heavy_rain": 0.4, "fog": 0.2}[weather]
    dark = hour >= 20 or hour < 6
    visibility = round(max(0.2, (4 + 4 * lighting / 100 if dark else 10) * vis_mult), 1)
    speed = {"walking": 4.8, "cycling": 14, "public_transport": 20, "car": 30}[mode]
    if st.button("Predict safety score", type="primary"):
        features = {"traffic_density": traffic, "accident_frequency": accidents, "street_lighting": lighting,
                    "pedestrian_density": pedestrians, "road_condition": road_cond, "crime_rate": crime,
                    "rainfall": rain, "visibility": visibility, "intersection_density": inter,
                    "vehicle_density": traffic, "time_of_day": float(hour), "day_of_week": 0,
                    "distance": distance, "route_duration": round(distance / speed * 60, 1),
                    "weather_condition": weather, "road_type": road_type, "traveler_type": traveler,
                    "travel_mode": mode}
        try:
            out = api.predict_risk(features)
            g1, g2 = st.columns([2, 3])
            with g1:
                ui.plot(ui.gauge(out["safety_score"], out["risk_level"]), "pg_gauge")
                st.markdown(f"{ui.pill(out['risk_level'])}", unsafe_allow_html=True)
                st.caption(out["note"])
            with g2:
                ui.plot(ui.factor_bars(out["factors"], "What shaped this prediction"), "pg_factors")
        except api.BackendError as exc:
            st.error(str(exc))
    st.caption("Visibility and trip time are derived from your choices (darker after 8 PM or before 6 AM, "
               "reduced by rain or fog).")
ui.disclaimer()
