"""The SafeRoute AI Agent.

Pipeline for every message:
  1. understand the question (intent + any "what if" conditions it mentions)
  2. call tools: re-run the ML risk model for the plan, and for a scenario when asked
  3. write the answer from those real numbers, using an OpenAI-compatible LLM when a key is
     configured, otherwise a built-in offline writer that uses the same facts
The agent only receives numbers the system actually produced and never claims a route is "safe".
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Dict, List, Optional, Tuple

from backend.models.schemas import ChatTurn, RouteRequest, Scenario
from backend.services import llm_client, risk_service as rs, whatif_service
from backend.utils import config

SYSTEM_PROMPT = """You are the SafeRoute AI Agent, a decision-support assistant that helps people compare routes.
Rules:
- Use ONLY the JSON facts provided. Never invent numbers, places, incidents or conditions.
- The scores are PREDICTED safety scores (0-100, higher = lower predicted risk) from a machine learning model trained on SYNTHETIC demo data.
- Never say a route is safe, secure or guaranteed. Say things like "lower predicted risk based on the available historical and contextual factors".
- Mention the traveler profile and time when they matter, name the main factors, and be honest about trade-offs such as extra minutes.
- If the user asks something the facts do not cover, say you do not have that information.
- Be concise: at most about 150 words, plain English, no markdown headings."""

_UNSAFE = re.compile(r"\b(definitely|absolutely|guaranteed?|completely|totally|100\s*%)\s+(safe|secure)\b|"
                     r"\bno risk\b|\brisk[- ]free\b|\bwill be safe\b|\bis safe\b|\bguarantee[sd]?\b", re.I)

DRIVER_TEXT = {
    "time": ("that time of day carries more risk in the model", "that time of day carries less risk in the model"),
    "visibility": ("visibility is lower (darkness, rain or fog)", "visibility is better"),
    "weather": ("rain or fog adds risk", "drier weather lowers the risk"),
    "other": ("other route features differ", "other route features differ"),
    "traffic": ("heavier traffic raises the predicted risk", "lighter traffic lowers the predicted risk"),
    "pedestrian_density": ("fewer people are around, so there is less street activity",
                           "more people around means more street activity"),
    "trip_length": ("the trip takes longer, so exposure time grows", "the trip is shorter, so exposure time drops"),
    "profile": ("this profile or mode is more exposed to the conditions on these routes",
                "this profile or mode is less exposed to the conditions on these routes"),
}
_cache: Dict[str, Tuple[str, str]] = {}


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #
def build_facts(ev: dict) -> dict:
    fastest = next(r for r in ev["routes"] if r["id"] == ev["fastest_id"])
    ctx = ev["context"]
    return {
        "context": {k: ctx[k] for k in ("origin", "destination", "time", "day", "weather", "mode", "mode_key",
                                         "traveler", "priorities", "traffic_change_pct")},
        "routes": [{"id": r["id"], "name": r["name"], "distance_km": r["distance_km"],
                    "duration_min": r["duration_min"], "predicted_safety_score": r["safety_score"],
                    "risk_level": r["risk_level"], "extra_minutes_vs_fastest": r["duration_min"] - fastest["duration_min"],
                    "positive_factors": r["strengths"], "negative_factors": r["concerns"],
                    "recommended": r["recommended"]} for r in ev["routes"]],
        "recommended_route": ev["recommended_id"], "fastest_route": ev["fastest_id"],
        "highest_score_route": ev["safest_id"], "recommendation_basis": ev["recommendation_basis"],
        "elevated_risk_junctions": [{"name": h["name"], "predicted_score": h["safety_score"],
                                     "main_factor": h["main_factor"]} for h in ev["hotspots"]],
        "data_note": "Synthetic demo data and simulated routes; not real-world safety measurements.",
    }


def _route(ev: dict, rid: str) -> dict:
    return next(r for r in ev["routes"] if r["id"] == rid)


def _lc(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s


def _join(items: List[str]) -> str:
    items = [_lc(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


# --------------------------------------------------------------------------- #
# Offline writer (uses the same facts the LLM would get)
# --------------------------------------------------------------------------- #
MODE_PHRASE = {"walking": "walking", "cycling": "cycling", "public_transport": "using public transport",
               "car": "driving"}
CLOSING = ("This reflects a lower predicted risk based on the available historical and contextual factors "
           "(synthetic demo data), not a guarantee of safety.")


def offline_recommendation(ev: dict) -> str:
    rec, fast = _route(ev, ev["recommended_id"]), _route(ev, ev["fastest_id"])
    best = _route(ev, ev["safest_id"])
    ctx = ev["context"]
    parts = [f"I recommend {rec['name']}."]
    if rec["id"] != fast["id"]:
        extra = rec["duration_min"] - fast["duration_min"]
        parts.append(f"It takes about {extra} minutes longer than {fast['name']}, the fastest option, but its "
                     f"predicted safety score is {rec['safety_score']:.0f}/100 compared with "
                     f"{fast['safety_score']:.0f}/100.")
    else:
        parts.append(f"It is the fastest option and has a predicted safety score of {rec['safety_score']:.0f}/100.")
        if best["id"] != rec["id"]:
            parts.append(f"{best['name']} scores slightly higher ({best['safety_score']:.0f}/100), but its extra "
                         f"travel time outweighs that gain.")
    if rec["strengths"]:
        parts.append(f"The main reasons are {_join(rec['strengths'])}.")
    if rec["concerns"]:
        parts.append(f"Points to watch: {_join(rec['concerns'])}.")
    parts.append(f"For a {ctx['traveler'].lower()} {MODE_PHRASE[ctx['mode_key']]} at {ctx['time']}, "
                 f"I weight {ctx['priorities']} most heavily.")
    if all(r["risk_level"] == "High" for r in ev["routes"]):
        parts.append("All options have a high predicted risk under these conditions. Try a different time or "
                     "travel mode on the What If? page.")
    elif rec["risk_level"] != "Low":
        parts.append(f"Its predicted risk is still {rec['risk_level'].lower()}, so extra care is sensible.")
    parts.append(CLOSING)
    return " ".join(parts)


def offline_why_risky(ev: dict, rid: str) -> str:
    r = _route(ev, rid)
    fast = _route(ev, ev["fastest_id"])
    parts = [f"{r['name']} has a predicted safety score of {r['safety_score']:.0f}/100 ({r['risk_level']} risk) "
             f"for a {ev['context']['traveler'].lower()} at {ev['context']['time']}."]
    if r["concerns"]:
        parts.append(f"The factors pulling its score down are {_join(r['concerns'])}.")
    else:
        parts.append("No single factor stands out as a major concern.")
    if r["strengths"]:
        parts.append(f"In its favour: {_join(r['strengths'])}.")
    if r["id"] == fast["id"]:
        parts.append(f"It is also the fastest option ({r['duration_min']} min), so you are trading some predicted "
                     f"safety for time.")
    parts.append("These are model predictions on synthetic demo data, not observed incidents.")
    return " ".join(parts)


def offline_tradeoff(ev: dict) -> str:
    rec, fast = _route(ev, ev["recommended_id"]), _route(ev, ev["fastest_id"])
    if rec["id"] == fast["id"]:
        best = _route(ev, ev["safest_id"])
        if best["id"] == rec["id"]:
            return (f"{rec['name']} is both the fastest and the highest-scoring option "
                    f"({rec['safety_score']:.0f}/100), so there is no trade-off here. {CLOSING}")
        extra = best["duration_min"] - rec["duration_min"]
        gain = best["safety_score"] - rec["safety_score"]
        return (f"The longer {best['name']} adds {extra} minutes for {gain:+.0f} points of predicted safety "
                f"({rec['safety_score']:.0f} to {best['safety_score']:.0f}). The model judges that gain too small to "
                f"justify the time, so {rec['name']} is recommended. You can weigh it differently. {CLOSING}")
    extra = rec["duration_min"] - fast["duration_min"]
    gain = rec["safety_score"] - fast["safety_score"]
    verdict = ("The gain is substantial, so the longer route looks worth it."
               if gain >= 10 else "The gain is modest, so it is a closer call.")
    return (f"{rec['name']} adds about {extra} minutes compared with {fast['name']} and raises the predicted safety "
            f"score by {gain:.0f} points ({fast['safety_score']:.0f} to {rec['safety_score']:.0f}). {verdict} "
            f"For a {ev['context']['traveler'].lower()} at {ev['context']['time']}, the main benefits are "
            f"{_join(rec['strengths']) or 'a lower predicted risk overall'}. The final choice is yours.")


def whatif_narrative(res: dict) -> str:
    focus = next(r for r in res["routes"] if r["id"] == res["recommended_before"])
    changes = "; ".join(res["changes"]) or "no changes"
    delta = round(focus["scenario_score"]) - round(focus["baseline_score"])
    parts = [f"Scenario ({changes}):",
             f"{focus['name']} goes from {focus['baseline_score']:.0f} to {focus['scenario_score']:.0f} "
             f"({delta:+d})."]
    same_dir = [d for d in focus["drivers"] if abs(d["effect"]) >= 0.5 and d["effect"] * focus["delta"] > 0]
    drivers = (same_dir or [d for d in focus["drivers"] if abs(d["effect"]) >= 0.5])[:2]
    if drivers:
        why = []
        for d in drivers:
            pair = DRIVER_TEXT.get(d["key"], ("conditions change", "conditions change"))
            why.append(pair[0] if d["effect"] < 0 else pair[1])
        parts.append("Why? " + "; ".join(why) + ".")
    else:
        parts.append("These changes barely move the predicted score for this route.")
    others = ", ".join(f"{r['name']} {r['baseline_score']:.0f} to {r['scenario_score']:.0f}"
                       for r in res["routes"] if r["id"] != focus["id"])
    parts.append(f"Other routes: {others}.")
    after = next(r for r in res["routes"] if r["id"] == res["recommended_after"])
    if res["recommendation_changed"]:
        parts.append(f"The recommendation changes from {focus['name']} to {after['name']} "
                     f"({after['scenario_score']:.0f}/100, {after['scenario_risk'].lower()} risk).")
    else:
        parts.append(f"The recommendation stays {after['name']} ({after['scenario_score']:.0f}/100, "
                     f"{after['scenario_risk'].lower()} risk).")
    parts.append(CLOSING)
    return " ".join(parts)


def offline_hotspots(ev: dict) -> str:
    if not ev["hotspots"]:
        return ("None of the simulated junctions on these routes fall in the elevated-risk range under the "
                "current conditions. They are synthetic demo junctions, so treat this as an illustration.")
    lines = "; ".join(f"{h['name']} ({h['safety_score']:.0f}/100, mainly {_lc(h['main_factor'])})"
                      for h in ev["hotspots"])
    return f"The lowest-scoring simulated junctions are: {lines}. They come from synthetic demo data."


def offline_model_info() -> str:
    from ml import predict as ml
    info = ml.model_info()
    top = sorted(info["feature_importance"].items(), key=lambda kv: -kv[1])[:5]
    names = ", ".join(k.replace("_", " ") for k, _ in top)
    return (f"The safety score comes from a Random Forest trained on {info['trained_rows']:,} synthetic rows "
            f"(held-out R² {info['metrics_holdout']['r2']}). The features it relies on most are {names}. "
            f"Scores of 70 or more are Low risk, 40-69 Medium and below 40 High. It is a decision-support "
            f"demo, not a measurement of real safety.")


# --------------------------------------------------------------------------- #
# Understanding the question
# --------------------------------------------------------------------------- #
_CUE = re.compile(r"\b(what if|what happens|what about|how about|if i|if it|if the|instead|as a|for a|for an|"
                  r"better for|switch|suppose|change)\b", re.I)


def _hour_from_text(q: str) -> Optional[float]:
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)", q, re.I)
    if m:
        h, mins = int(m.group(1)) % 12, int(m.group(2) or 0)
        if m.group(3).lower().startswith("p"):
            h += 12
        return h + mins / 60
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", q)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60
    for word, hour in (("midnight", 0.0), ("noon", 12.0), ("late night", 23.5), ("tonight", 22.0),
                       ("at night", 22.0), ("morning", 8.0), ("afternoon", 15.0), ("evening", 19.0)):
        if word in q:
            return hour
    return None


def parse_scenario(question: str) -> Tuple[Optional[Scenario], List[str]]:
    q = question.lower()
    data, notes = {}, []
    hour = _hour_from_text(q)
    if hour is not None:
        data["travel_time"] = f"{int(hour)}:{int(round((hour - int(hour)) * 60)) % 60:02d}"
        notes.append(f"time={rs.fmt_hour(hour)}")
    if re.search(r"heavy rain|storm|downpour", q):
        data["weather"] = "heavy_rain"
    elif re.search(r"\brain|raining|rainy|drizzle|\bwet\b", q):
        data["weather"] = "light_rain"
    elif re.search(r"\bfog|foggy|\bmist", q):
        data["weather"] = "fog"
    elif re.search(r"\bclear\b|sunny", q):
        data["weather"] = "clear"
    if "weather" in data:
        notes.append(f"weather={data['weather']}")
    m = re.search(r"traffic[^%\d]{0,25}(\d{1,3})\s*%|(\d{1,3})\s*%[^.?]{0,25}traffic", q)
    if m:
        pct = float(m.group(1) or m.group(2))
        if re.search(r"decrease|drop|lower|less|reduce|fall|down", q):
            pct = -pct
        data["traffic_change_pct"] = pct
    elif re.search(r"(heavy|more|worse|busier|rush) traffic|traffic (jam|increase)", q):
        data["traffic_change_pct"] = 30.0
    if "traffic_change_pct" in data:
        notes.append(f"traffic={data['traffic_change_pct']:+.0f}%")
    if _CUE.search(q):
        left = q.split("instead of")[0] if "instead of" in q else q
        if re.search(r"cyclist|cycling|bicycle|\bbike\b", left):
            data.update(traveler_type="cyclist", travel_mode="cycling")
        elif re.search(r"senior|elderly|old age|grandparent", left):
            data["traveler_type"] = "senior_citizen"
        elif re.search(r"\bstudent", left):
            data["traveler_type"] = "student"
        elif re.search(r"pedestrian", left):
            data["traveler_type"] = "pedestrian"
        if re.search(r"driving|\bcar\b|\bdrive\b", left):
            data["travel_mode"] = "car"
        elif re.search(r"walking|\bwalk\b|on foot", left):
            data["travel_mode"] = "walking"
        elif re.search(r"public transport|\bbus\b|metro|transit", left):
            data["travel_mode"] = "public_transport"
        for k in ("traveler_type", "travel_mode"):
            if k in data:
                notes.append(f"{k}={data[k]}")
    if not data:
        return None, []
    return Scenario(**data), notes


def detect_intent(question: str, scenario: Optional[Scenario]) -> Tuple[str, Optional[str]]:
    q = question.lower()
    route_m = re.search(r"route\s*([abc])\b", q)
    rid = route_m.group(1).upper() if route_m else None
    if scenario is not None:
        return "scenario", rid
    if re.search(r"risky|unsafe|dangerous|\bbad\b|low score|why.*(score|low)|worst", q):
        return "why_risky", rid
    if re.search(r"longer|worth|extra (time|minutes)|slower|trade-?off|faster|shortest|quickest", q):
        return "tradeoff", rid
    if re.search(r"hotspot|junction|danger spot|black ?spot|where", q):
        return "hotspots", rid
    if re.search(r"model|accuracy|how .*(work|calculat|score)|feature|random forest|synthetic|trained", q):
        return "model_info", rid
    if re.search(r"why|recommend|which route|best|safest|should i", q):
        return "why_recommended", rid
    return "summary", rid


def _scenario_summary(res: dict) -> dict:
    return {"changes": res["changes"],
            "routes": [{"id": r["id"], "score_before": r["baseline_score"], "score_after": r["scenario_score"],
                        "change": r["delta"], "risk_after": r["scenario_risk"], "minutes_after": r["scenario_duration"],
                        "main_drivers": [{"factor": d["label"], "effect_points": d["effect"]}
                                         for d in r["drivers"][:3]]} for r in res["routes"]],
            "recommended_before": res["recommended_before"], "recommended_after": res["recommended_after"]}


# --------------------------------------------------------------------------- #
# Writing the answer
# --------------------------------------------------------------------------- #
def _llm_answer(user_payload: dict, question: str, history: List[ChatTurn]) -> Optional[str]:
    if not llm_client.configured():
        return None
    key = hashlib.sha1(json.dumps([user_payload, question, [h.content for h in history[-4:]]],
                                  sort_keys=True, default=str).encode()).hexdigest()
    if key in _cache:
        return _cache[key][0]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": h.role, "content": h.content} for h in history[-4:]]
    messages.append({"role": "user", "content": f"FACTS (JSON):\n{json.dumps(user_payload, default=str)}\n\n"
                                                 f"REQUEST: {question}"})
    try:
        text = llm_client.chat(messages)
    except llm_client.LLMError:
        return None
    if _UNSAFE.search(text):
        return None
    if len(_cache) > 200:
        _cache.clear()
    _cache[key] = (text, config.llm_model())
    return text


def _pack(text: str, llm_text: Optional[str]) -> dict:
    if llm_text:
        return {"text": llm_text, "mode": "llm", "model": config.llm_model()}
    return {"text": text, "mode": "offline", "model": None}


def recommendation_message(ev: dict) -> dict:
    llm = _llm_answer(build_facts(ev), "Write the route recommendation for this traveler: name the recommended "
                                      "route, compare it with the fastest option, explain the main factors and "
                                      "the personalization, then add the caution that scores are predictions.", [])
    return _pack(offline_recommendation(ev), llm)


def whatif_message(res: dict, ev_after: dict) -> dict:
    payload = {"scenario_result": _scenario_summary(res), "facts_after_change": build_facts(ev_after)}
    llm = _llm_answer(payload, "Explain how the predicted safety scores change under this scenario and why, "
                               "using the drivers listed. Say whether the recommendation changes.", [])
    return _pack(whatif_narrative(res), llm)


def chat(plan: RouteRequest, question: str, history: List[ChatTurn]) -> dict:
    tools: List[dict] = []
    ev = rs.evaluate(plan, geometry=False)
    tools.append({"tool": "score_routes", "args": {"origin": plan.origin, "destination": plan.destination,
                                                   "time": ev["context"]["time"], "traveler": plan.traveler_type,
                                                   "mode": plan.travel_mode},
                  "summary": "Re-ran the ML model: " + ", ".join(
                      f"{r['name']} {r['safety_score']:.0f}/100" for r in ev["routes"])})
    scenario, notes = parse_scenario(question)
    intent, rid = detect_intent(question, scenario)
    facts = build_facts(ev)
    payload: dict = {"plan_facts": facts}
    answer = ""
    if intent == "scenario":
        res = whatif_service.run(plan, scenario)
        tools.append({"tool": "run_what_if", "args": {"scenario": scenario.model_dump(exclude_none=True,
                                                                                      exclude_defaults=True)},
                      "summary": f"Recomputed features and scores: {'; '.join(res['changes']) or 'no change'}. "
                                 f"Recommended route: {res['recommended_before']} -> {res['recommended_after']}"})
        payload["scenario_result"] = _scenario_summary(res)
        answer = whatif_narrative(res)
    elif intent == "why_risky":
        target = rid or min(ev["routes"], key=lambda r: r["safety_score"])["id"]
        answer = offline_why_risky(ev, target)
        payload["focus_route"] = target
    elif intent == "tradeoff":
        answer = offline_tradeoff(ev)
    elif intent == "hotspots":
        answer = offline_hotspots(ev)
    elif intent == "model_info":
        answer = offline_model_info()
    elif intent == "why_recommended":
        answer = offline_recommendation(ev)
        if rid and rid != ev["recommended_id"]:
            answer = (f"{_route(ev, rid)['name']} is not the recommended route here "
                      f"({_route(ev, rid)['safety_score']:.0f}/100, {_route(ev, rid)['risk_level']} risk). " + answer)
    else:
        answer = offline_recommendation(ev) + (" You can ask, for example: Why is Route C risky? Is the longer "
                                               "route worth it? What happens if I travel at 11 PM?")
    llm = _llm_answer(payload, question, history)
    out = _pack(answer, llm)
    out["answer"] = out.pop("text")
    out.update({"intent": intent, "tools_used": tools, "disclaimer": config.DISCLAIMER})
    return out
