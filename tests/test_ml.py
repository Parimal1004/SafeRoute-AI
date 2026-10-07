import numpy as np

from ml import predict as ml
from ml.features import FEATURES, risk_level

ROW = dict(traffic_density=45, accident_frequency=2.0, street_lighting=85, pedestrian_density=60,
           road_condition=75, crime_rate=25, rainfall=0, visibility=7.4, intersection_density=9,
           vehicle_density=40, time_of_day=22, day_of_week=0, distance=2.2, route_duration=28,
           weather_condition="clear", road_type="commercial", traveler_type="student", travel_mode="walking")


def test_scores_are_between_0_and_100():
    s = ml.score([ROW, dict(ROW, street_lighting=5, accident_frequency=9.5, crime_rate=90)])
    assert np.all((s >= 0) & (s <= 100))


def test_worse_conditions_lower_the_score():
    good, bad = ml.score([ROW, dict(ROW, street_lighting=20, accident_frequency=8, pedestrian_density=10,
                                    crime_rate=70, road_type="service_lane")])
    assert good > bad + 10


def test_risk_bands():
    assert risk_level(85) == "Low" and risk_level(70) == "Low"
    assert risk_level(69.9) == "Medium" and risk_level(40) == "Medium"
    assert risk_level(39.9) == "High"


def test_explanation_has_positive_and_negative_factors():
    factors = ml.explain([ROW])[0]
    assert {f["impact"] for f in factors} >= {"positive", "negative"}
    assert all(f["key"] for f in factors)


def test_shapley_attribution_adds_up():
    new = dict(ROW, time_of_day=12, pedestrian_density=80, visibility=10, traffic_density=60)
    drivers = ml.attribute_change(ROW, new)
    total = float(ml.score([new])[0] - ml.score([ROW])[0])
    assert abs(sum(d["effect"] for d in drivers) - total) < 0.2


def test_model_info_has_metrics_and_all_features():
    info = ml.model_info()
    assert info["metrics_holdout"]["r2"] > 0.8
    assert set(info["feature_importance"]) == set(FEATURES)
