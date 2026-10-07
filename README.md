# SafeRoute AI

**An AI-powered route recommendation system that combines machine learning and an AI agent to help users choose routes based on predicted safety risk rather than travel time alone.**

> The shortest route isn't always the safest route.

Supports **SDG 11 (Sustainable Cities and Communities, Target 11.2)** and, secondarily, **SDG 3 (Good Health and Well-Being)**.

> **Important:** the data in this project is **synthetic** and the routes are a **demo simulation**. Scores are *predicted safety scores based on available historical and contextual factors* in that synthetic data. This system provides decision support and should not be treated as a guarantee of real-world safety.

## Problem

Navigation apps rank routes by time or distance. The fastest route can run through poorly lit streets, busy junctions or accident-prone stretches, and what is risky depends on who is travelling and when. A student walking at 10 PM faces different risks from a senior citizen crossing at noon.

## Solution

SafeRoute AI compares route alternatives on a predicted safety score next to travel time, explains the score, and has an AI agent turn the numbers into a personalised recommendation.

```
User input -> Route data -> ML risk prediction -> Explainability -> AI agent -> Recommendation
```

## Features

- **Safe Route Planner:** three route alternatives with distance, duration, safety score (0-100), risk level, main risk factors and a clearly marked recommendation.
- **ML risk model:** Random Forest trained on a synthetic dataset (17 features + traveler profile), scores routes and individual junctions.
- **Explainability:** model feature importance, per-route factor effects (green adds, red removes points), exact Shapley attribution for What If? changes.
- **What If?:** change the time, weather, traffic, travel mode or traveler profile; every route is re-scored and the agent explains what moved. Includes a 24-hour safety profile chart.
- **AI Agent chat:** answers questions such as "Why is Route C risky?" or "What happens if I travel at 11 PM?" from the real model output, and shows which tools it used.
- **Safety hotspots:** lowest-scoring simulated junctions on your routes, plus a 12-junction demo grid you can explore by hour, profile and weather.
- **Personalisation:** the model takes the traveler type as an input and the agent states which factors it weights for that profile.
- **Model playground:** set conditions by hand and see the model's prediction and explanation.
- **Works with no API key and no map service** (offline agent, schematic map).

## Technology stack (all free)

| Layer | Tools |
|---|---|
| Frontend | Streamlit, Plotly, custom CSS |
| Backend | FastAPI, Pydantic, Uvicorn |
| ML | pandas, numpy, scikit-learn (Random Forest), joblib |
| AI agent | Any OpenAI-compatible API (Groq free tier suggested) with an offline fallback agent |
| Maps | None needed: simulated routes drawn with Plotly |
| Hosting | Streamlit Community Cloud (frontend), Render free web service (backend) |

## ML approach

- **Data:** 20,000 synthetic rows from `ml/generate_dataset.py`. Infrastructure attributes depend on road type; traffic, pedestrians, visibility and rainfall depend on time of day, day of week and weather; the score is produced by documented rules plus noise, with different weights for each traveler type and travel mode.
- **Features:** traffic_density, accident_frequency, street_lighting, pedestrian_density, road_condition, crime_rate, weather_condition, rainfall, visibility, road_type, intersection_density, vehicle_density, time_of_day, day_of_week, distance, route_duration, traveler_type, travel_mode.
- **Model:** `RandomForestRegressor` in a pipeline with one-hot encoding. Hold-out R² about 0.93, MAE about 3.8 points. Gradient boosting and a Ridge baseline are compared on the Safety Insights page.
- **Risk bands:** **70-100 Low**, **40-69 Medium**, **0-39 High**.
- **Recommendation rule:** recommendation score = predicted safety score minus a small penalty per extra minute versus the fastest route (0.35 point/min walking, 0.40 cycling, 0.50 public transport, 0.60 car).
- **Explainability:** for each group of related features (for example lighting, or traffic = traffic density + vehicle density) the values are replaced by typical ones and the change in predicted score is measured. For What If? changes, exact Shapley values over the changed groups are used, so the effects add up to the total change.

## AI agent

For every message the agent (1) works out the intent and any conditions mentioned (a time, rain, "traffic +30%", a profile or mode), (2) calls its tools: re-score the plan with the ML model and, if needed, run a What If? scenario, and (3) writes the answer from those numbers. With `LLM_API_KEY` set it uses the LLM; otherwise a built-in offline writer uses the same facts. Guardrails: the LLM only sees system-produced facts, is told never to call a route safe, and any answer containing absolute claims ("definitely safe", "guaranteed") is replaced by the offline answer.

## Architecture

```
SafeRoute-AI/
├── backend/            FastAPI app
│   ├── main.py
│   ├── routes/         health, risk, routing, agent endpoints
│   ├── services/       route simulation, risk scoring, what-if, hotspots, agent, LLM client
│   ├── models/         Pydantic schemas
│   └── utils/          config (environment variables)
├── frontend/           Streamlit app
│   ├── app.py          navigation + sidebar
│   ├── views/          home, planner, what_if, insights, agent, about (named views/, not pages/, so Streamlit's
│   │                   automatic page discovery cannot skip app.py when someone opens a deep link first)
│   └── components/     API client, UI helpers and charts, state
├── ml/                 features.py, generate_dataset.py, train_model.py, predict.py, model/
├── data/               saferoute_dataset.csv (synthetic)
├── tests/              pytest suite (ML, API, agent)
├── utils/smoke_test.py quick check of a running backend
├── requirements.txt    everything (local)   backend/ and frontend/ have their own for deployment
├── render.yaml  Dockerfile  .env.example  .gitignore
```

## Running locally

Requires Python 3.10 or newer.

```bash
# 1. get the code and create an environment
cd SafeRoute-AI
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (optional) configure the LLM for the agent; skip this to use the offline agent
cp .env.example .env               # then edit LLM_API_KEY

# 3. start the backend (terminal 1). The model trains automatically on first start if missing.
uvicorn backend.main:app --reload --port 8000

# 4. start the frontend (terminal 2)
streamlit run frontend/app.py
```

Open http://localhost:8501. API docs are at http://localhost:8000/docs.

Useful extras:

```bash
python -m ml.generate_dataset      # regenerate data/saferoute_dataset.csv
python -m ml.train_model           # retrain and print metrics
pip install -r requirements-dev.txt && pytest      # run the tests
python utils/smoke_test.py         # check a running backend
```

Try it: From **CBIT** To **Kokapet** (or Gachibowli), Walking, Student, 10:00 PM, then open **What If?** and **AI Agent**. Note that CBIT to Gachibowli is a long walk (about 7 km), so exposure time lowers every score; a shorter pair shows bigger differences between routes.

## Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `LLM_API_KEY` | backend | Key for an OpenAI-compatible provider. Empty = offline agent. Never commit it. |
| `LLM_BASE_URL` | backend | Default `https://api.groq.com/openai/v1` |
| `LLM_MODEL` | backend | Default `llama-3.1-8b-instant`; check your provider's console for current model names |
| `LLM_TIMEOUT` | backend | Seconds before falling back to the offline agent (default 25) |
| `BACKEND_URL` | frontend | Default `http://localhost:8000`; set to your Render URL when deployed |
| `CORS_ORIGINS` | backend | Optional comma-separated allowed origins (default `*`) |

To switch provider, change `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`. `.env.example` lists OpenAI, OpenRouter, Gemini and Ollama examples. Free-tier limits and model names change, so check the provider's site.

## Deployment (free tiers)

**Backend on Render**

https://saferoute-ai-ltzg.onrender.com

**Frontend on Streamlit Community Cloud**

https://saferoute-ai-parimal.streamlit.app/

## Limitations

- All data is synthetic and the routes are simulated, so scores say nothing about real streets, crime or accidents.
- Place names map to approximate Hyderabad coordinates; other text is placed at a stable simulated location.
- No live traffic, weather, incident or closure data.
- Explanations approximate the model's behaviour and are not causal proof.
- A real system would need real, regularly refreshed data (for example open road-safety and transport datasets) and validation with local authorities and users.

## SDG connection

**SDG 11, Target 11.2:** access to safe, affordable, accessible and sustainable transport for all, with special attention to people in vulnerable situations. SafeRoute AI helps people see risk before they travel and adapt to their own situation. **SDG 3:** fewer unsafe journeys supports better health and fewer injuries.
