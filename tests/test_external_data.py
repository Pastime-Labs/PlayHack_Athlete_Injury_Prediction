import json

import numpy as np
import pandas as pd

from src import external_data
from src.external_data import (
    RUNFITCHECK_FEATURES,
    SEQUENCE_CHANNELS,
    _daily_value,
    build_playhack_shared_features,
    load_pmdata_daily,
    load_pmdata_sequences,
    load_runfitcheck_samples,
)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_daily_value_vectorizes_mixed_dates_and_numeric_coercion(tmp_path, monkeypatch):
    path = tmp_path / "minute.json"
    _write(path, [
        {"dateTime": "2020-01-01T01:00:00", "value": "2"},
        {"dateTime": "bad", "startTime": "2020-01-01 02:00:00", "value": {"bpm": "4"}},
        {"start": "2020-01-02", "value": "3"},
        {"dateTime": "2020-01-02", "value": "bad"},
    ])
    original = external_data.pd.to_datetime
    calls = []

    def tracked(values, **kwargs):
        calls.append(kwargs)
        return original(values, **kwargs)

    monkeypatch.setattr(external_data.pd, "to_datetime", tracked)
    result = _daily_value(path)

    assert calls == [{"format": "mixed", "errors": "coerce"}]
    assert result.to_dict() == {
        external_data.pd.Timestamp("2020-01-01"): 6.0,
        external_data.pd.Timestamp("2020-01-02"): 3.0,
    }


def test_pmdata_loader_builds_semantic_daily_channels_and_masks(tmp_path):
    participant = tmp_path / "p01"
    fitbit = participant / "fitbit"
    days = [f"2020-01-{day:02d}" for day in range(1, 32)]
    for filename, value in (
        ("steps.json", "100"),
        ("calories.json", "20"),
        ("very_active_minutes.json", "5"),
        ("moderately_active_minutes.json", "7"),
        ("lightly_active_minutes.json", "30"),
    ):
        _write(fitbit / filename, [{"dateTime": day, "value": value} for day in days])
    _write(
        fitbit / "sleep.json",
        [
            {
                "dateOfSleep": day,
                "minutesAsleep": 420,
                "timeInBed": 480,
                "efficiency": 50,
            }
            for day in days
        ],
    )
    _write(
        fitbit / "exercise.json",
        [{"startTime": day, "activeDuration": 3_600_000} for day in days],
    )

    daily = load_pmdata_daily(participant)
    assert daily.loc["2020-01-01", "TotalSteps"] == 100
    assert daily.loc["2020-01-01", "sleep_efficiency"] == 0.875
    assert daily.loc["2020-01-01", "training_hours"] == 1.0
    assert daily[["hr_AvgHeartRate_mean", "hr_MaxHeartRate_max"]].isna().all().all()

    sequences = load_pmdata_sequences(tmp_path, stride=1)
    assert sequences.values.shape == (2, len(SEQUENCE_CHANNELS), 30)
    assert not sequences.observed[:, -2:, :].any()
    assert np.array_equal(np.unique(sequences.subject_ids), np.array(["p01"]))


def test_runfitcheck_windows_are_chronological_and_groupable(tmp_path):
    rows = []
    for respondent in (11, 22):
        row = {
            "Respondent_ID": respondent,
            "Geslacht": "Vrouw" if respondent == 11 else "Man",
            "Leeftijd": "25 t/m 34 jaar",
            "Lengte": 170,
            "Gewicht": 68,
        }
        for wave in range(9):
            row[f"M{wave}HoevaakHard"] = 2 + wave
            row[f"M{wave}MinutenHard"] = 50
            row[f"Hard_expM{wave}"] = f"{100 + 50 * wave},00"
            row[f"M{wave}NieuwBles"] = "0,00"
            row[f"M{wave}NieuwAantal"] = "0,00"
        rows.append(row)
    rows[0]["M1NieuwBles"] = rows[0]["M1NieuwAantal"] = "1,00"
    rows[0]["M3NieuwBles"] = rows[0]["M3NieuwAantal"] = "1,00"
    path = tmp_path / "runfitcheck.csv"
    pd.DataFrame(rows).to_csv(path, sep=";", index=False)

    samples = load_runfitcheck_samples(path)
    first = samples.loc[
        samples.respondent_id.eq(11) & samples.window_start.eq(0)
    ].iloc[0]

    assert samples.groupby("respondent_id").size().to_dict() == {11: 6, 22: 6}
    assert first.weekly_training_minutes == 150
    assert first.exposure_change == 50
    assert first.prior_injury_count == 1
    assert first.target == 1
    assert set(RUNFITCHECK_FEATURES).issubset(samples.columns)


def test_playhack_shared_features_use_two_15_day_bins():
    daily = pd.DataFrame({
        "athlete_id": np.repeat([1], 30),
        "day": np.arange(1, 31),
        "training_hours": [1.0] * 15 + [2.0] * 15,
        "session_count": [1.0] * 30,
    })
    metadata = pd.DataFrame({
        "athlete_id": [1],
        "age": [20],
        "gender": ["female"],
        "height_cm": [170.0],
        "weight_kg_baseline": [68.0],
        "prior_season_injury_count": [2],
    })

    features = build_playhack_shared_features({"daily": daily, "metadata": metadata})

    assert features.loc[1, "weekly_training_minutes"] == 840
    assert features.loc[1, "weekly_training_frequency"] == 7
    assert features.loc[1, "exposure_change"] == 420
    assert features.loc[1, "prior_injury_count"] == 2
