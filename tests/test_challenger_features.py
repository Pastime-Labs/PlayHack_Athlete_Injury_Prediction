import numpy as np
import pandas as pd

from src.challenger_features import build_challenger_feature_packs


def test_challenger_features_use_only_the_30_day_history():
    load = np.array([0, 0, 2, 1, 0, 0, 2] + [1] * 23, dtype=float)
    session_types = np.zeros((30, 3))
    active = load > 0
    session_types[active, 0] = 1
    session_types[2, 1] = 1
    session_types[2, 0] = 0
    sleep = np.r_[400.0, 400.0 + 10.0 * load[:-1]]
    daily = pd.DataFrame({
        "athlete_id": 1,
        "day": np.arange(1, 31),
        "training_hours": load,
        "session_count": active.astype(float),
        "practice_count": session_types[:, 0],
        "gym_count": session_types[:, 1],
        "scrimmage_count": session_types[:, 2],
        "TotalMinutesAsleep": sleep,
        "sleep_efficiency": 0.9,
        "hr_AvgHeartRate_mean": 60.0 + 2.0 * np.arange(30),
        "VeryActiveMinutes": np.arange(30, dtype=float),
    })

    packs = build_challenger_feature_packs({"daily": daily})

    assert packs["workload_shape"].loc[1, "shape_max_rest_streak"] == 2
    assert packs["workload_shape"].loc[1, "shape_session_type_switches"] >= 1
    assert np.isclose(packs["cross_modal"].loc[1, "cross_sleep_after_load_slope"], 10.0)
