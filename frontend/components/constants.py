"""Labels and options shared by the Streamlit pages."""

MODES = {"walking": "Walking", "cycling": "Cycling", "public_transport": "Public Transport", "car": "Car"}
TRAVELERS = {"student": "Student", "general": "General", "senior_citizen": "Senior Citizen",
             "cyclist": "Cyclist", "pedestrian": "Pedestrian"}
WEATHER = {"clear": "Clear", "cloudy": "Cloudy", "light_rain": "Light rain", "heavy_rain": "Heavy rain",
           "fog": "Fog"}

RISK_COLORS = {"Low": "#15803d", "Medium": "#b45309", "High": "#b91c1c"}
RISK_BG = {"Low": "#dcfce7", "Medium": "#fef3c7", "High": "#fee2e2"}
ACCENT = "#0f766e"

FALLBACK_PLACES = ["CBIT", "Gachibowli", "HITEC City", "Madhapur", "Kondapur", "Financial District", "Kokapet"]

DISCLAIMER = ("This system provides decision support and should not be treated as a guarantee of real-world "
              "safety. Scores are predicted from synthetic demo data.")
SCORE_NOTE = "Predicted safety score based on available historical and contextual factors."

DEMO_PLAN = {"origin": "CBIT", "destination": "Gachibowli", "travel_mode": "walking",
             "traveler_type": "student", "travel_time": "22:00", "weather": "clear", "include_agent": True}

SAMPLE_QUESTIONS = [
    "Why did you recommend this route?",
    "Why is Route C risky?",
    "Is the longer route worth taking?",
    "What happens if I travel at 11 PM?",
    "Which route is better for a cyclist?",
    "What if it starts raining and traffic increases by 30%?",
]

FEATURE_LABELS = {
    "traffic_density": "Traffic density", "accident_frequency": "Accident frequency",
    "street_lighting": "Street lighting", "pedestrian_density": "Pedestrian density",
    "road_condition": "Road condition", "crime_rate": "Crime rate (synthetic index)",
    "rainfall": "Rainfall", "visibility": "Visibility", "intersection_density": "Intersection density",
    "vehicle_density": "Vehicle density", "time_of_day": "Time of day", "day_of_week": "Day of week",
    "distance": "Distance", "route_duration": "Route duration", "weather_condition": "Weather condition",
    "road_type": "Road type", "traveler_type": "Traveler type", "travel_mode": "Travel mode",
}
