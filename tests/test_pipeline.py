import numpy as np
import pandas as pd
import pytest

from src.data import CUTOFF, _hourly_daily, load_bundle
from src.evaluation import _best_threshold, score_predictions
from src.features import _ramp_match
from src.final_model import validate_submission
from src.matched import MatchedEventModel


def test_ramp_match_finds_change_point():
    signal = np.zeros(30)
    signal[17:] = np.arange(1, 14) * 0.5
    match = _ramp_match(signal)
    assert abs(match["cp"] - 18) <= 1
    assert match["gain"] > 0.95
    assert match["amplitude"] > 0


def test_threshold_optimizes_f1():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.45, 0.9])
    threshold, score = _best_threshold(y, p)
    assert 0.40 < threshold <= 0.45
    assert score == 1.0


def test_false_negative_costs_30_and_false_positive_only_hurts_f1():
    metrics = score_predictions(
        y_class=np.array([1, 0]), probability=np.array([0.1, 0.9]), threshold=0.5,
        true_onset=np.array([5.0, np.nan]), true_recovery=np.array([10.0, np.nan]),
        pred_onset=np.array([5.0, 20.0]), pred_recovery=np.array([10.0, 20.0]),
        baseline_onset=np.array([10.0, 10.0]), baseline_recovery=np.array([12.0, 12.0]),
    )
    assert metrics["f1"] == 0.0
    assert metrics["onset_mae_penalized"] == 30.0
    assert metrics["recovery_mae_penalized"] == 30.0


def test_matched_event_model_detects_visible_ramp():
    rng = np.random.default_rng(7)
    n = 80
    sequence = rng.normal(0, 0.02, (n, 30, 3)) + np.array([100.0, 80.0, 60.0])
    injury = np.zeros(n, dtype=int); injury[:40] = 1
    onset = np.full(n, 30.0); onset[:40] = np.tile(np.arange(1, 11), 4)
    for i in range(40):
        ramp = np.maximum(0, np.arange(1, 31) - (13 + onset[i]) + 1)
        sequence[i] += ramp[:, None] * np.array([2.0, -1.0, 1.0])
    model = MatchedEventModel(np.array([1, -1, 1])).fit(sequence, injury, onset)
    probability, predicted_onset, _ = model.predict(sequence)
    assert probability[:40].mean() > probability[40:].mean()
    assert np.mean(np.abs(predicted_onset[:40] - onset[:40])) < 3


def test_cached_daily_table_is_legal_and_ordered():
    daily = load_bundle()["daily"]
    assert daily.date.max() < CUTOFF
    assert daily.groupby("athlete_id").size().eq(30).all()
    expected = daily.groupby("athlete_id").cumcount().add(1)
    assert daily.day.equals(expected)


def test_hourly_aggregation_ignores_post_cutoff_values(tmp_path):
    base = pd.DataFrame({
        "Id": [1, 1, 1],
        "ActivityHour": ["2026-02-03 10:00", "2026-02-03 11:00", "2026-02-04 10:00"],
        "value": [60.0, 80.0, 100.0],
    })
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    base.to_csv(first, index=False)
    base.assign(value=[60.0, 80.0, 1_000_000.0]).to_csv(second, index=False)
    left = _hourly_daily(first, "ActivityHour", ["value"])
    right = _hourly_daily(second, "ActivityHour", ["value"])
    pd.testing.assert_frame_equal(left, right)


def test_perfect_predictions_score_perfectly():
    metrics = score_predictions(
        y_class=np.array([1, 1, 0]), probability=np.array([0.9, 0.8, 0.1]), threshold=0.5,
        true_onset=np.array([5.0, 15.0, np.nan]), true_recovery=np.array([8.0, 12.0, np.nan]),
        pred_onset=np.array([5.0, 15.0, 20.0]), pred_recovery=np.array([8.0, 12.0, 10.0]),
        baseline_onset=np.array([10.0, 10.0, 10.0]), baseline_recovery=np.array([10.0, 10.0, 10.0]),
    )
    assert metrics["f1"] == 1.0
    assert metrics["onset_skill"] == 1.0
    assert metrics["recovery_skill"] == 1.0


def test_submission_contract_rejects_invalid_ranges():
    valid = pd.DataFrame({
        "athlete_id": np.array([1, 2], dtype=int),
        "injured_in_risk_window": np.array([0, 1], dtype=int),
        "onset_day_offset": np.array([1, 30], dtype=int),
        "recovery_duration": np.array([5, 20], dtype=int),
    })
    validate_submission(valid, np.array([1, 2]))
    invalid = valid.copy()
    invalid.loc[0, "onset_day_offset"] = 31
    with pytest.raises(ValueError, match="Onset"):
        validate_submission(invalid)


def test_split_registry_is_disjoint_and_complete():
    splits = pd.read_csv("cache/representations/splits.csv")
    assert not splits.athlete_id.duplicated().any()
    assert set(splits.split) == {"development", "lockbox"}
    assert splits.loc[splits.split.eq("development"), "fold"].between(0, 4).all()
    assert splits.loc[splits.split.eq("lockbox"), "fold"].isna().all()
