"""Styling, HTML cards and Plotly charts for the SafeRoute AI dashboard."""
from __future__ import annotations

import html
import inspect
import math
from typing import Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

from components.constants import ACCENT, DISCLAIMER, FEATURE_LABELS, RISK_BG, RISK_COLORS

FONT = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"
INK, MUTED, BORDER = "#0f172a", "#64748b", "#e2e8f0"


# --------------------------------------------------------------------------- #
# Compatibility helpers
# --------------------------------------------------------------------------- #
def wide(fn) -> dict:
    """Full-width kwargs that work on both old and new Streamlit versions."""
    params = inspect.signature(fn).parameters
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, key=key, config={"displayModeBar": False}, **wide(st.plotly_chart))


def esc(x) -> str:
    return html.escape(str(x))


# --------------------------------------------------------------------------- #
# CSS
# --------------------------------------------------------------------------- #
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: {FONT}; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; color: {INK}; }}
h1 {{ font-weight: 700; }} h2 {{ font-weight: 650; margin-top: 1.6rem; }} h3 {{ font-weight: 600; }}
.sr-hero {{ background: linear-gradient(135deg, #0f172a 0%, #134e4a 100%); border-radius: 18px;
  padding: 2.6rem 2.4rem; color: #f8fafc; margin-bottom: 1.4rem; }}
.sr-hero h1 {{ color: #fff; font-size: 2.6rem; margin: 0 0 .4rem 0; }}
.sr-hero p {{ color: #cbd5e1; font-size: 1.08rem; max-width: 46rem; margin: .3rem 0; }}
.sr-hero .sr-quote {{ color: #5eead4; font-weight: 600; font-size: 1.2rem; margin-top: 1rem; }}
.sr-eyebrow {{ text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; color: #5eead4; font-weight: 600; }}
.sr-card {{ background: #fff; border: 1px solid {BORDER}; border-radius: 14px; padding: 1.1rem 1.2rem;
  box-shadow: 0 1px 2px rgba(15,23,42,.04); min-height: 168px; }}
.sr-card h4 {{ margin: 0 0 .35rem 0; font-size: 1rem; color: {INK}; }}
.sr-card p {{ margin: 0; color: {MUTED}; font-size: .92rem; line-height: 1.5; }}
.sr-step {{ background: #fff; border: 1px solid {BORDER}; border-radius: 12px; padding: .8rem .9rem; text-align: center;
  font-size: .86rem; font-weight: 600; color: {INK}; }}
.sr-step small {{ display:block; color: {MUTED}; font-weight: 500; margin-top: .2rem; font-size: .75rem; }}
.sr-route {{ background: #fff; border: 1px solid {BORDER}; border-radius: 16px; padding: 1.1rem 1.2rem; box-shadow: 0 1px 2px rgba(15,23,42,.04); min-height: 330px; }}
.sr-route.rec {{ border: 2px solid {ACCENT}; box-shadow: 0 8px 24px rgba(15,118,110,.18); background: #f0fdfa; }}
.sr-route .sr-name {{ font-weight: 650; font-size: 1.05rem; color: {INK}; display:flex; justify-content:space-between;
  align-items:center; gap:.5rem; flex-wrap: wrap; }}
.sr-route .sr-sub {{ color: {MUTED}; font-size: .86rem; margin-top: .15rem; }}
.sr-score {{ font-size: 2.3rem; font-weight: 700; line-height: 1.1; margin: .6rem 0 .1rem 0; }}
.sr-score span {{ font-size: 1rem; color: {MUTED}; font-weight: 500; }}
.sr-pill {{ display:inline-block; padding: .15rem .6rem; border-radius: 999px; font-size: .78rem; font-weight: 600; }}
.sr-badge-rec {{ background: {ACCENT}; color: #fff; padding: .18rem .6rem; border-radius: 999px; font-size: .72rem;
  font-weight: 700; letter-spacing: .04em; }}
.sr-tag {{ display:inline-block; background:#f1f5f9; color:#475569; padding:.1rem .5rem; border-radius:6px;
  font-size:.72rem; font-weight:600; margin-right:.3rem; }}
.sr-factors {{ font-size: .84rem; color: #334155; margin-top: .6rem; line-height: 1.5; }}
.sr-factors b {{ color: {INK}; }}
.sr-agent {{ background: #fff; border: 1px solid {BORDER}; border-left: 4px solid {ACCENT}; border-radius: 12px;
  padding: 1rem 1.2rem; }}
.sr-agent .sr-agent-head {{ font-weight: 650; color: {INK}; display:flex; align-items:center; gap:.6rem; margin-bottom:.4rem; }}
.sr-agent p {{ margin: 0; color: #1e293b; line-height: 1.6; font-size: .96rem; }}
.sr-chip {{ background:#ecfeff; color:#155e75; border-radius:6px; padding:.15rem .55rem; font-size:.8rem; font-weight:600; }}
.sr-chip.off {{ background:#f1f5f9; color:#475569; }}
.sr-note {{ font-size: .8rem; color: {MUTED}; }}
.sr-disclaimer {{ background:#f8fafc; border:1px dashed #cbd5e1; border-radius:10px; padding:.7rem .9rem;
  font-size:.82rem; color:#475569; margin-top:1.4rem; }}
.sr-banner {{ border-radius: 12px; padding: .8rem 1rem; font-size: .95rem; margin: .6rem 0; }}
.sr-hot {{ background:#fff; border:1px solid {BORDER}; border-radius:12px; padding:.7rem .9rem; margin-bottom:.5rem; }}
[data-testid="stMetric"] {{ background:#fff; border:1px solid {BORDER}; border-radius:12px; padding:.7rem .9rem; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: .82rem; }}
[data-testid="stMetricValue"] {{ font-size: 1.3rem; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"] {{ border-radius: 10px; font-weight: 600; }}
.stButton > button, .stFormSubmitButton > button {{ border-radius: 10px; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown(f'<div class="sr-disclaimer"><b>Note.</b> {esc(DISCLAIMER)} Routes shown are a '
                f'<b>demo simulation</b>, not real roads.</div>', unsafe_allow_html=True)


def pill(level: str) -> str:
    return (f'<span class="sr-pill" style="background:{RISK_BG[level]};color:{RISK_COLORS[level]}">'
            f'{esc(level)} risk</span>')


# --------------------------------------------------------------------------- #
# Cards
# --------------------------------------------------------------------------- #
def agent_card(msg: Optional[dict], title: str = "SafeRoute AI Agent") -> None:
    if not msg:
        return
    chip = (f'<span class="sr-chip">LLM &middot; {esc(msg.get("model") or "")}</span>' if msg["mode"] == "llm"
            else '<span class="sr-chip off">Offline agent (no API key set)</span>')
    st.markdown(f'<div class="sr-agent"><div class="sr-agent-head">{esc(title)} {chip}</div>'
                f'<p>{esc(msg["text"])}</p></div>', unsafe_allow_html=True)


def route_card(r: dict, fastest_id: str, safest_id: str) -> str:
    rec = r["recommended"]
    color = RISK_COLORS[r["risk_level"]]
    tags = ""
    if r["id"] == fastest_id:
        tags += '<span class="sr-tag">Fastest</span>'
    if r["id"] == safest_id:
        tags += '<span class="sr-tag">Highest score</span>'
    badge = '<span class="sr-badge-rec">&#9733; RECOMMENDED</span>' if rec else ""
    concerns = esc(", ".join(r["concerns"][:2]) or "None significant")
    strengths = esc(", ".join(r["strengths"][:2]) or "None significant")
    return (f'<div class="sr-route{" rec" if rec else ""}">'
            f'<div class="sr-name"><span>{esc(r["name"])}</span>{badge}</div>'
            f'<div class="sr-sub">{esc(r["label"])}</div>'
            f'<div class="sr-score" style="color:{color}">{r["safety_score"]:.0f}<span>/100</span></div>'
            f'<div>{pill(r["risk_level"])}</div>'
            f'<div class="sr-sub" style="margin-top:.6rem"><b>{r["duration_min"]} min</b> &nbsp;|&nbsp; '
            f'<b>{r["distance_km"]:.1f} km</b></div>'
            f'<div style="margin-top:.4rem">{tags}</div>'
            f'<div class="sr-factors"><b>Helping:</b> {strengths}<br><b>Main risk factors:</b> {concerns}</div>'
            f'</div>')


def step_strip(steps: List[tuple]) -> None:
    cols = st.columns(len(steps))
    for col, (title, sub) in zip(cols, steps):
        col.markdown(f'<div class="sr-step">{esc(title)}<small>{esc(sub)}</small></div>', unsafe_allow_html=True)


def info_card(title: str, body: str) -> str:
    return f'<div class="sr-card"><h4>{esc(title)}</h4><p>{esc(body)}</p></div>'


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _base(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=36, b=10),
                      font=dict(family=FONT, size=13, color=INK), showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0))
    return fig


def _risk_bands(fig: go.Figure, orientation: str = "h") -> None:
    for lo, hi, level in ((0, 40, "High"), (40, 70, "Medium"), (70, 100, "Low")):
        kw = dict(fillcolor=RISK_COLORS[level], opacity=0.06, line_width=0, layer="below")
        if orientation == "h":
            fig.add_hrect(y0=lo, y1=hi, **kw)
        else:
            fig.add_vrect(x0=lo, x1=hi, **kw)


def score_bars(routes: List[dict]) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[r["name"] + (" ★" if r["recommended"] else "") for r in routes],
        y=[r["safety_score"] for r in routes],
        marker_color=[RISK_COLORS[r["risk_level"]] for r in routes],
        text=[f'{r["safety_score"]:.0f}' for r in routes], textposition="outside",
        hovertemplate="%{x}<br>Predicted safety score %{y}/100<extra></extra>"))
    _risk_bands(fig)
    fig.update_yaxes(range=[0, 108], title="Predicted safety score", gridcolor="#f1f5f9")
    fig.update_layout(title="Predicted safety score")
    return _base(fig, legend=False)


def tradeoff_scatter(routes: List[dict]) -> go.Figure:
    fig = go.Figure()
    for r in routes:
        fig.add_trace(go.Scatter(
            x=[r["duration_min"]], y=[r["safety_score"]], mode="markers+text", name=r["name"],
            text=[r["name"] + (" ★" if r["recommended"] else "")], textposition="top center",
            marker=dict(size=26 if r["recommended"] else 18, color=RISK_COLORS[r["risk_level"]],
                        line=dict(width=3 if r["recommended"] else 1, color=ACCENT if r["recommended"] else "#fff")),
            hovertemplate=f'{r["name"]}<br>%{{x}} min, score %{{y}}/100<extra></extra>'))
    _risk_bands(fig)
    xs = [r["duration_min"] for r in routes]
    pad = max(3, (max(xs) - min(xs)) * 0.35)
    fig.update_xaxes(title="Estimated duration (min)", range=[min(xs) - pad, max(xs) + pad], gridcolor="#f1f5f9")
    fig.update_yaxes(title="Predicted safety score", range=[0, 108], gridcolor="#f1f5f9")
    fig.update_layout(title="Time versus predicted safety")
    return _base(fig, legend=False)


def factor_bars(factors: List[dict], title: str = "") -> go.Figure:
    items = [f for f in factors if f["kind"] == "route" and abs(f["effect"]) >= 0.05][:8]
    items = list(reversed(items))
    fig = go.Figure(go.Bar(
        x=[f["effect"] for f in items], y=[f["text"] for f in items], orientation="h",
        marker_color=[RISK_COLORS["Low"] if f["effect"] > 0 else RISK_COLORS["High"] for f in items],
        text=[f'{f["effect"]:+.1f}' for f in items], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>%{x:+.1f} points<extra></extra>"))
    span = max([abs(f["effect"]) for f in items] + [1]) * 1.35
    fig.update_xaxes(range=[-span, span], zeroline=True, zerolinecolor="#94a3b8", gridcolor="#f1f5f9",
                     title="Effect on predicted score (points)")
    fig.update_layout(title=title)
    return _base(fig, height=max(240, 38 * len(items) + 90), legend=False)


def importance_bar(imp: Dict[str, float], top: int = 12, title: str = "") -> go.Figure:
    items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:top][::-1]
    fig = go.Figure(go.Bar(x=[v * 100 for _, v in items], y=[FEATURE_LABELS.get(k, k) for k, _ in items],
                           orientation="h", marker_color=ACCENT,
                           text=[f"{v * 100:.1f}%" for _, v in items], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(title="Share of total importance (%)", gridcolor="#f1f5f9")
    fig.update_layout(title=title)
    return _base(fig, height=max(300, 30 * len(items) + 100), legend=False)


def before_after_bars(routes: List[dict]) -> go.Figure:
    names = [r["name"] for r in routes]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Original plan", x=names, y=[r["baseline_score"] for r in routes],
                         marker_color="#94a3b8", text=[f'{r["baseline_score"]:.0f}' for r in routes],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="What-if scenario", x=names, y=[r["scenario_score"] for r in routes],
                         marker_color=[RISK_COLORS[r["scenario_risk"]] for r in routes],
                         text=[f'{r["scenario_score"]:.0f}' for r in routes], textposition="outside"))
    _risk_bands(fig)
    fig.update_yaxes(range=[0, 108], title="Predicted safety score", gridcolor="#f1f5f9")
    fig.update_layout(barmode="group", title="Before and after")
    return _base(fig)


def time_profile_chart(profile: dict, plan_hour: float) -> go.Figure:
    palette = {"A": "#475569", "B": ACCENT, "C": "#b45309"}
    fig = go.Figure()
    for rid, series in profile["scores"].items():
        fig.add_trace(go.Scatter(x=profile["hours"], y=series, mode="lines+markers", name=f"Route {rid}",
                                 line=dict(color=palette.get(rid, "#334155"), width=3), marker=dict(size=5),
                                 hovertemplate=f"Route {rid}<br>%{{x}}:00, score %{{y}}<extra></extra>"))
    _risk_bands(fig)
    fig.add_vline(x=plan_hour, line_dash="dot", line_color=INK,
                  annotation_text="Planned time", annotation_position="top")
    fig.update_xaxes(title="Hour of day", dtick=2, range=[-0.5, 23.5], gridcolor="#f1f5f9")
    fig.update_yaxes(title="Predicted safety score", range=[0, 100], gridcolor="#f1f5f9")
    fig.update_layout(title="Predicted safety score across the day")
    return _base(fig, height=360)


def driver_bars(drivers: List[dict], title: str = "") -> go.Figure:
    items = list(reversed([d for d in drivers if abs(d["effect"]) >= 0.05]))
    fig = go.Figure(go.Bar(
        x=[d["effect"] for d in items], y=[d["label"] for d in items], orientation="h",
        marker_color=[RISK_COLORS["Low"] if d["effect"] > 0 else RISK_COLORS["High"] for d in items],
        text=[f'{d["effect"]:+.1f}' for d in items], textposition="outside", cliponaxis=False))
    span = max([abs(d["effect"]) for d in items] + [1]) * 1.35
    fig.update_xaxes(range=[-span, span], zeroline=True, zerolinecolor="#94a3b8", gridcolor="#f1f5f9",
                     title="Points added to or removed from the score")
    fig.update_layout(title=title)
    return _base(fig, height=max(220, 44 * len(items) + 90), legend=False)


def gauge(score: float, level: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(suffix=" / 100", font=dict(size=34)),
        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=RISK_COLORS[level], thickness=0.3),
                   steps=[dict(range=[0, 40], color=RISK_BG["High"]), dict(range=[40, 70], color=RISK_BG["Medium"]),
                          dict(range=[70, 100], color=RISK_BG["Low"])])))
    return _base(fig, height=230, legend=False)


# --------------------------------------------------------------------------- #
# Maps (no external map service: a schematic canvas in km)
# --------------------------------------------------------------------------- #
def _xy_factory(lat0: float, lon0: float):
    k = math.cos(math.radians(lat0)) * 111.32

    def xy(lat, lon):
        return (lon - lon0) * k, (lat - lat0) * 110.57
    return xy


def _map_layout(fig: go.Figure, title: str, height: int = 470) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=44, b=90),
                      font=dict(family=FONT, size=12, color=INK), plot_bgcolor="#f1f5f9",
                      legend=dict(orientation="h", yanchor="top", y=-0.17, x=0), title=title)
    fig.update_xaxes(title="km (east-west)", gridcolor="#e2e8f0", zeroline=False)
    fig.update_yaxes(title="km (north-south)", gridcolor="#e2e8f0", zeroline=False, scaleanchor="x", scaleratio=1)
    fig.add_annotation(text="Demo Route Simulation", xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False,
                       font=dict(size=12, color="#475569"), bgcolor="rgba(255,255,255,0.85)", borderpad=4,
                       xanchor="left", yanchor="top")
    return fig


def route_map(result: dict) -> go.Figure:
    ctx = result["context"]
    o, d = ctx["origin_xy"], ctx["destination_xy"]
    xy = _xy_factory((o[0] + d[0]) / 2, (o[1] + d[1]) / 2)
    fig = go.Figure()
    for r in sorted(result["routes"], key=lambda r: r["recommended"]):
        pts = [xy(*c) for c in r["coords"]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        color = RISK_COLORS[r["risk_level"]]
        label = f'{r["name"]} · {r["safety_score"]:.0f}/100' + (" ★ recommended" if r["recommended"] else "")
        if r["recommended"]:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=16, color="rgba(15,118,110,0.22)"),
                                     hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label,
                                 line=dict(width=5 if r["recommended"] else 3, color=color,
                                           dash="solid" if r["recommended"] else "dot"),
                                 hovertemplate=f'{r["name"]}<br>{r["duration_min"]} min, {r["distance_km"]} km<extra></extra>'))
    hot_names = {h["name"] for h in result["hotspots"]}
    normal = [j for r in result["routes"] for j in r["junctions"] if j["name"] not in hot_names]
    if normal:
        px = [xy(j["lat"], j["lon"]) for j in normal]
        fig.add_trace(go.Scatter(x=[p[0] for p in px], y=[p[1] for p in px], mode="markers", name="Junction",
                                 marker=dict(size=9, color=[RISK_COLORS[j["risk_level"]] for j in normal],
                                             line=dict(width=1.5, color="#fff")),
                                 text=[f'{j["name"]}<br>{j["safety_score"]:.0f}/100 ({j["risk_level"]})<br>{j["main_factor"]}'
                                       for j in normal], hovertemplate="%{text}<extra></extra>"))
    hots = result["hotspots"]
    if hots:
        px = [xy(j["lat"], j["lon"]) for j in hots]
        fig.add_trace(go.Scatter(x=[p[0] for p in px], y=[p[1] for p in px], mode="markers", name="Safety hotspot",
                                 marker=dict(size=17, symbol="diamond", color="#fff",
                                             line=dict(width=3, color=RISK_COLORS["High"])),
                                 text=[f'{j["name"]}<br>{j["safety_score"]:.0f}/100<br>{j["main_factor"]}' for j in hots],
                                 hovertemplate="%{text}<extra></extra>"))
    sx, sy = xy(*o)
    ex, ey = xy(*d)
    fig.add_trace(go.Scatter(x=[sx], y=[sy], mode="markers+text", name="Start", text=[ctx["origin"]],
                             textposition="bottom center", marker=dict(size=16, color="#166534", line=dict(width=2, color="#fff"))))
    fig.add_trace(go.Scatter(x=[ex], y=[ey], mode="markers+text", name="Destination", text=[ctx["destination"]],
                             textposition="top center", marker=dict(size=16, symbol="square", color=INK,
                                                                    line=dict(width=2, color="#fff"))))
    return _map_layout(fig, "Route alternatives (schematic)")


def hotspot_map(junctions: List[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[j["x"] for j in junctions], y=[j["y"] for j in junctions], mode="markers+text",
        text=[j["name"].replace("Junction ", "") for j in junctions], textposition="middle center",
        textfont=dict(color="#fff", size=11),
        marker=dict(size=34, color=[RISK_COLORS[j["risk_level"]] for j in junctions], line=dict(width=2, color="#fff")),
        customdata=[[j["name"], j["safety_score"], j["risk_level"], j["main_factor"]] for j in junctions],
        hovertemplate="%{customdata[0]}<br>%{customdata[1]}/100 (%{customdata[2]})<br>%{customdata[3]}<extra></extra>",
        showlegend=False))
    fig = _map_layout(fig, "Synthetic demo junctions (abstract grid)", height=430)
    fig.update_xaxes(range=[-0.3, 8.3])
    fig.update_yaxes(range=[-0.3, 6.3])
    return fig
