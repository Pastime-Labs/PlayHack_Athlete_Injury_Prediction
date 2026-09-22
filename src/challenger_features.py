from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-6
SESSION_TYPES = ("practice_count", "gym_count", "scrimmage_count")


def _longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    x, y = x[valid], y[valid]
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator < EPS:
        return np.nan
    return float(np.dot(centered, y - y.mean()) / denominator)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) < EPS or np.std(y[valid]) < EPS:
        return np.nan
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _high_load_delta(load: np.ndarray, response: np.ndarray) -> float:
    valid = np.isfinite(load) & np.isfinite(response)
    if valid.sum() < 6:
        return np.nan
    load, response = load[valid], response[valid]
    high = load >= np.quantile(load, 0.75)
    if high.all() or (~high).all():
        return np.nan
    return float(response[high].mean() - response[~high].mean())


def _workload_shape(frame: pd.DataFrame) -> dict[str, float]:
    load = frame["training_hours"].to_numpy(dtype=float)
    sessions = frame["session_count"].to_numpy(dtype=float)
    active = np.isfinite(sessions) & (sessions > 0)
    active_load = load[np.isfinite(load) & (load > 0)]
    total = float(np.nansum(load))
    mean = float(np.nanmean(load))
    std = float(np.nanstd(load))
    post_rest = []
    rest_run = 0
    for value, is_active in zip(load, active):
        if is_active:
            if rest_run >= 2 and np.isfinite(value):
                post_rest.append(value)
            rest_run = 0
        else:
            rest_run += 1

    type_counts = frame.loc[:, SESSION_TYPES].sum(axis=0).to_numpy(dtype=float)
    type_total = float(type_counts.sum())
    type_probability = type_counts[type_counts > 0] / type_total if type_total else np.array([])
    types = np.argmax(frame.loc[:, SESSION_TYPES].to_numpy(dtype=float), axis=1)
    active_types = types[active]
    top3 = float(np.sort(active_load)[-3:].sum()) if len(active_load) else 0.0
    return {
        "shape_active_days": float(active.sum()),
        "shape_rest_days": float((~active).sum()),
        "shape_max_active_streak": float(_longest_run(active)),
        "shape_max_rest_streak": float(_longest_run(~active)),
        "shape_top3_load_share": top3 / (total + EPS),
        "shape_peak_to_active_mean": float(np.nanmax(active_load)) / (float(np.mean(active_load)) + EPS)
        if len(active_load) else np.nan,
        "shape_load_monotony": mean / (std + EPS),
        "shape_load_strain": total * mean / (std + EPS),
        "shape_mean_absolute_load_change": float(np.nanmean(np.abs(np.diff(load)))),
        "shape_post_rest_load_ratio": float(np.mean(post_rest)) / (float(np.mean(active_load)) + EPS)
        if post_rest and len(active_load) else 0.0,
        "shape_session_type_entropy": float(-np.sum(type_probability * np.log(type_probability)))
        if len(type_probability) else 0.0,
        "shape_session_type_switches": float(np.sum(active_types[1:] != active_types[:-1]))
        if len(active_types) > 1 else 0.0,
    }


def _cross_modal(frame: pd.DataFrame) -> dict[str, float]:
    load = frame["training_hours"].to_numpy(dtype=float)
    sleep = frame["TotalMinutesAsleep"].to_numpy(dtype=float)
    sleep_efficiency = frame["sleep_efficiency"].to_numpy(dtype=float)
    heart_rate = frame["hr_AvgHeartRate_mean"].to_numpy(dtype=float)
    activity = frame["VeryActiveMinutes"].to_numpy(dtype=float)
    lag_load, lag_sleep = load[:-1], sleep[1:]
    _, lag_sleep_efficiency = load[:-1], sleep_efficiency[1:]
    _, lag_heart_rate = load[:-1], heart_rate[1:]
    activity_slope = _slope(activity, heart_rate)
    valid = np.isfinite(activity) & np.isfinite(heart_rate)
    residual_delta = np.nan
    if valid.sum() >= 6 and np.isfinite(activity_slope):
        intercept = float(heart_rate[valid].mean() - activity_slope * activity[valid].mean())
        residual = heart_rate - (intercept + activity_slope * activity)
        residual_delta = float(np.nanmean(residual[-7:]) - np.nanmean(residual[:23]))
    sleep_threshold = np.nanquantile(sleep[:-1], 0.25) if np.isfinite(sleep[:-1]).any() else np.nan
    low_sleep = sleep[:-1] <= sleep_threshold
    valid_after_sleep = low_sleep & np.isfinite(load[1:])
    other_after_sleep = ~low_sleep & np.isfinite(load[1:])
    training_after_low_sleep = (
        float(np.mean(load[1:][valid_after_sleep]) - np.mean(load[1:][other_after_sleep]))
        if valid_after_sleep.any() and other_after_sleep.any() else np.nan
    )
    return {
        "cross_sleep_after_load_slope": _slope(lag_load, lag_sleep),
        "cross_sleep_after_high_load_delta": _high_load_delta(lag_load, lag_sleep),
        "cross_sleep_eff_after_load_slope": _slope(lag_load, lag_sleep_efficiency),
        "cross_sleep_eff_after_high_load_delta": _high_load_delta(lag_load, lag_sleep_efficiency),
        "cross_training_after_low_sleep_delta": training_after_low_sleep,
        "cross_hr_after_load_slope": _slope(lag_load, lag_heart_rate),
        "cross_hr_after_high_load_delta": _high_load_delta(lag_load, lag_heart_rate),
        "cross_hr_activity_slope": activity_slope,
        "cross_hr_activity_correlation": _correlation(activity, heart_rate),
        "cross_hr_activity_residual_recent_delta": residual_delta,
        "cross_sleep_hr_correlation": _correlation(sleep, heart_rate),
    }


def build_challenger_feature_packs(bundle: dict[str, object]) -> dict[str, pd.DataFrame]:
    """Create isolated prospective feature packs for challenger-only experiments."""
    daily = bundle["daily"].sort_values(["athlete_id", "day"])
    required = {
        "athlete_id", "day", "training_hours", "session_count", *SESSION_TYPES,
        "TotalMinutesAsleep", "sleep_efficiency", "hr_AvgHeartRate_mean", "VeryActiveMinutes",
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"Daily data lacks challenger columns: {missing}")
    counts = daily.groupby("athlete_id").size()
    if counts.ne(30).any():
        raise ValueError("Challenger features require exactly 30 days per athlete")

    shape_rows, cross_rows = {}, {}
    for athlete_id, frame in daily.groupby("athlete_id", sort=True):
        shape_rows[athlete_id] = _workload_shape(frame)
        cross_rows[athlete_id] = _cross_modal(frame)
    return {
        "workload_shape": pd.DataFrame.from_dict(shape_rows, orient="index"),
        "cross_modal": pd.DataFrame.from_dict(cross_rows, orient="index"),
    }
