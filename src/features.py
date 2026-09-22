from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-6


def _slope(values: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return np.nan
    x = np.arange(len(values), dtype=float)[mask]
    y = values[mask].astype(float)
    return float(np.polyfit(x, y, 1)[0])


def _robust_residual(values: np.ndarray) -> np.ndarray:
    # Remove each athlete's weekly schedule before measuring prospective drift.
    weekday_baseline = np.nanmedian(values[:14].reshape(2, 7), axis=0)
    expected = np.tile(weekday_baseline, 5)[:30]
    relative = (values - expected) / (np.abs(expected) + EPS)
    baseline = relative[:14]
    center = np.nanmedian(baseline)
    scale = np.nanmedian(np.abs(baseline - center)) * 1.4826
    if not np.isfinite(scale) or scale < EPS:
        scale = np.nanstd(baseline)
    if not np.isfinite(scale) or scale < EPS:
        scale = 1.0
    return (relative - center) / scale


def _ramp_match(signal: np.ndarray) -> dict[str, float]:
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    baseline_sse = float(np.sum((signal - signal.mean()) ** 2)) + EPS
    best = {"gain": 0.0, "cp": 30.0, "amplitude": 0.0, "corr": 0.0}
    t = np.arange(1, 31, dtype=float)
    for cp in range(14, 31):
        ramp = np.maximum(0.0, t - cp + 1.0)
        design = np.column_stack([np.ones(30), ramp])
        coef, *_ = np.linalg.lstsq(design, signal, rcond=None)
        fitted = design @ coef
        sse = float(np.sum((signal - fitted) ** 2))
        gain = max(0.0, 1.0 - sse / baseline_sse)
        corr = float(np.corrcoef(signal, ramp)[0, 1]) if signal.std() > EPS else 0.0
        if gain > best["gain"]:
            best = {"gain": gain, "cp": float(cp), "amplitude": float(coef[1]), "corr": corr}
    return best


def build_features(bundle: dict[str, object]) -> dict[str, pd.DataFrame]:
    daily = bundle["daily"].copy()
    metadata = bundle["metadata"].set_index("athlete_id")
    channels = list(bundle["channels"])
    groups = daily.groupby("athlete_id", sort=True)

    blocks: dict[str, pd.DataFrame] = {}
    basic_parts = []
    temporal_parts = []
    for channel in channels:
        stats = groups[channel].agg(["mean", "std", "min", "max", "median"])
        stats.columns = [f"{channel}_{c}" for c in stats.columns]
        basic_parts.append(stats)

        rows = {}
        for athlete_id, frame in groups:
            values = frame[channel].to_numpy(dtype=float)
            early, recent = values[:23], values[23:]
            rows[athlete_id] = {
                f"{channel}_slope": _slope(values),
                f"{channel}_early_mean": np.nanmean(early),
                f"{channel}_recent_mean": np.nanmean(recent),
                f"{channel}_recent_delta": np.nanmean(recent) - np.nanmean(early),
                f"{channel}_recent_ratio": np.nanmean(recent) / (abs(np.nanmean(early)) + EPS),
                f"{channel}_recent_std": np.nanstd(recent),
            }
        temporal_parts.append(pd.DataFrame.from_dict(rows, orient="index"))

    basic = metadata.join(pd.concat(basic_parts, axis=1))
    temporal = basic.join(pd.concat(temporal_parts, axis=1))

    structural_rows = {}
    signal_channels = {
        "steps": ("TotalSteps", 1.0),
        "calories": ("Calories", 1.0),
        "very_active": ("VeryActiveMinutes", 1.0),
        "light_active": ("LightlyActiveMinutes", -1.0),
        "sleep": ("TotalMinutesAsleep", -1.0),
        "sleep_eff": ("sleep_efficiency", -1.0),
        "heart_rate": ("hr_AvgHeartRate_mean", 1.0),
    }
    for athlete_id, frame in groups:
        residuals, row, matches = [], {}, []
        for name, (channel, direction) in signal_channels.items():
            if channel not in frame:
                continue
            residual = direction * _robust_residual(frame[channel].to_numpy(dtype=float))
            residuals.append(residual)
            match = _ramp_match(residual)
            matches.append(match)
            row.update({f"match_{name}_{key}": value for key, value in match.items()})
        strain = np.nanmedian(np.vstack(residuals), axis=0)
        match = _ramp_match(strain)
        row.update({f"strain_{key}": value for key, value in match.items()})
        row["strain_last7"] = float(np.nanmean(strain[-7:]))
        row["strain_first12"] = float(np.nanmean(strain[:12]))
        row["strain_acceleration"] = _slope(strain[-10:]) - _slope(strain[:12])
        gains = np.array([max(0.0, m["gain"] * max(0.0, m["amplitude"])) for m in matches])
        cps = np.array([m["cp"] for m in matches])
        row["match_consensus_cp"] = float(np.average(cps, weights=gains)) if gains.sum() > EPS else 30.0
        row["match_consensus_strength"] = float(gains.sum())
        row["match_cp_spread"] = float(np.sqrt(np.average((cps - row["match_consensus_cp"]) ** 2, weights=gains))) if gains.sum() > EPS else 0.0
        structural_rows[athlete_id] = row
    structural_only = pd.DataFrame.from_dict(structural_rows, orient="index")
    structural = temporal.join(structural_only)

    activity_tokens = ("Step", "Distance", "Active", "Intensity", "Calories", "training", "session", "strain", "match_steps", "match_calories", "match_very")
    recovery_tokens = ("Sleep", "sleep", "Heart", "hr_", "weight", "BMI", "strain", "match_sleep", "match_heart")
    blocks["basic"] = basic
    blocks["temporal"] = temporal
    blocks["structural"] = structural
    blocks["mechanistic"] = metadata.join(structural_only)
    blocks["activity_view"] = structural[[c for c in structural if any(t in c for t in activity_tokens)]]
    blocks["recovery_view"] = structural[[c for c in structural if any(t in c for t in recovery_tokens)]]
    return blocks


def build_sequence(bundle: dict[str, object]) -> tuple[np.ndarray, list[str]]:
    daily = bundle["daily"]
    preferred = [
        "TotalSteps", "Calories", "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes",
        "TotalMinutesAsleep", "sleep_efficiency", "training_hours", "hr_AvgHeartRate_mean",
        "hr_MaxHeartRate_max", "hsteps_StepTotal_std", "hint_TotalIntensity_std",
    ]
    channels = [c for c in preferred if c in daily]
    sequence = np.stack([
        frame[channels].to_numpy(dtype=np.float32)
        for _, frame in daily.groupby("athlete_id", sort=True)
    ])
    return sequence, channels
