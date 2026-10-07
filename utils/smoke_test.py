"""Quick check of a running backend:  python utils/smoke_test.py [http://localhost:8000]"""
import sys

import requests

base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
plan = {"origin": "CBIT", "destination": "Gachibowli", "travel_mode": "walking", "traveler_type": "student",
        "travel_time": "22:00"}
print("health      ", requests.get(f"{base}/health", timeout=90).json()["status"])
r = requests.post(f"{base}/recommend-route", json=plan, timeout=90).json()
for x in r["routes"]:
    print(f"{x['name']}: {x['safety_score']:.0f}/100 {x['risk_level']:6} {x['duration_min']} min"
          + ("  <- recommended" if x["recommended"] else ""))
print("what-if     ", requests.post(f"{base}/what-if", json={"plan": plan, "scenario": {"travel_time": "18:00"}},
                                    timeout=90).json()["changes"])
print("agent       ", requests.post(f"{base}/agent/chat", json={"plan": plan, "question": "Why is Route C risky?"},
                                    timeout=90).json()["answer"][:120], "...")
print("OK")
