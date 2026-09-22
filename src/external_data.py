from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SEQUENCE_CHANNELS = (
    "TotalSteps",
    "Calories",
    "VeryActiveMinutes",
    "FairlyActiveMinutes",
    "LightlyActiveMinutes",
    "TotalMinutesAsleep",
    "sleep_efficiency",
    "training_hours",
    "hr_AvgHeartRate_mean",
    "hr_MaxHeartRate_max",
)

RUNFITCHECK_FEATURES = (
    "age",
    "sex",
    "bmi",
    "prior_injury_count",
    "weekly_training_minutes",
    "weekly_training_frequency",
    "exposure_change",
    "exposure_ratio",
    "missing_fraction",
)


@dataclass(frozen=True)
class ExternalSequences:
    values: np.ndarray
    observed: np.ndarray
    subject_ids: np.ndarray
    end_dates: np.ndarray
    channels: tuple[str, ...] = SEQUENCE_CHANNELS


def _comma_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype("string").str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _age_midpoint(value: Any) -> float:
    numbers = [float(item) for item in re.findall(r"\d+(?:[.,]\d+)?", str(value))]
    return float(np.mean(numbers[:2])) if numbers else np.nan


def _shared_sex(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower().map(
        {"vrouw": "female", "female": "female", "man": "male", "male": "male"}
    )


def load_runfitcheck_samples(path: str | Path) -> pd.DataFrame:
    """Build leakage-safe 28-day-history/28-day-target respondent windows."""
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    wave_columns = {
        f"M{wave}{suffix}"
        for wave in range(9)
        for suffix in ("HoevaakHard", "MinutenHard", "NieuwBles", "NieuwAantal")
    }
    required = {"Respondent_ID", "Geslacht", "Leeftijd", "Lengte", "Gewicht"} | {
        f"Hard_expM{wave}" for wave in range(9)
    } | wave_columns
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"RunFitCheck columns missing: {missing}")

    age = raw["Leeftijd"].map(_age_midpoint)
    sex = _shared_sex(raw["Geslacht"])
    height = _comma_numeric(raw["Lengte"])
    weight = _comma_numeric(raw["Gewicht"])
    bmi = weight / (height / 100.0) ** 2
    frequency = {wave: _comma_numeric(raw[f"M{wave}HoevaakHard"]) for wave in range(9)}
    duration = {wave: _comma_numeric(raw[f"M{wave}MinutenHard"]) for wave in range(9)}
    exposure = {
        wave: _comma_numeric(raw[f"Hard_expM{wave}"]).fillna(
            frequency[wave] * duration[wave]
        )
        for wave in range(9)
    }
    new_injury = {wave: _comma_numeric(raw[f"M{wave}NieuwBles"]) for wave in range(9)}
    injury_count = {
        wave: _comma_numeric(raw[f"M{wave}NieuwAantal"]).combine_first(new_injury[wave])
        for wave in range(9)
    }

    rows = []
    for start in range(6):
        early, recent = start, start + 1
        target = pd.concat(
            [new_injury[start + 2], new_injury[start + 3]], axis=1
        )
        observed_target = target.notna().all(axis=1)
        prior = pd.concat(
            [injury_count[wave] for wave in range(recent + 1)], axis=1
        ).fillna(0.0).sum(axis=1)
        missing_inputs = pd.concat(
            [
                age,
                sex,
                bmi,
                exposure[early],
                exposure[recent],
                frequency[early],
                frequency[recent],
            ],
            axis=1,
        )
        frame = pd.DataFrame({
            "respondent_id": raw["Respondent_ID"],
            "window_start": start,
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "prior_injury_count": prior,
            "weekly_training_minutes": exposure[recent],
            "weekly_training_frequency": frequency[recent],
            "exposure_change": exposure[recent] - exposure[early],
            "exposure_ratio": (exposure[recent] + 30.0) / (exposure[early] + 30.0),
            "missing_fraction": missing_inputs.isna().mean(axis=1),
            "target": target.max(axis=1),
        })
        rows.append(frame.loc[observed_target])

    samples = pd.concat(rows, ignore_index=True)
    if samples["respondent_id"].isna().any() or not samples["target"].isin([0.0, 1.0]).all():
        raise ValueError("RunFitCheck respondent IDs or target labels are invalid")
    if samples.duplicated(["respondent_id", "window_start"]).any():
        raise ValueError("RunFitCheck respondent-window keys are not unique")
    return samples


def build_playhack_shared_features(bundle: dict[str, object]) -> pd.DataFrame:
    """Map PlayHack's two 15-day training bins to the RunFitCheck contract."""
    metadata = bundle["metadata"].set_index("athlete_id").sort_index()
    daily = bundle["daily"].sort_values(["athlete_id", "day"])
    required = {"athlete_id", "day", "training_hours", "session_count"}
    if not required.issubset(daily.columns):
        raise ValueError(f"PlayHack daily columns missing: {sorted(required - set(daily.columns))}")
    counts = daily.groupby("athlete_id").size().reindex(metadata.index)
    if counts.ne(30).any():
        raise ValueError("PlayHack shared features require 30 days per athlete")

    def weekly(mask: pd.Series) -> tuple[pd.Series, pd.Series]:
        grouped = daily.loc[mask].groupby("athlete_id")
        minutes = grouped.training_hours.sum().reindex(metadata.index, fill_value=0.0) * 60.0 * 7.0 / 15.0
        frequency = grouped.session_count.sum().reindex(metadata.index, fill_value=0.0) * 7.0 / 15.0
        return minutes, frequency

    early_minutes, early_frequency = weekly(daily.day.le(15))
    recent_minutes, recent_frequency = weekly(daily.day.gt(15))
    height = pd.to_numeric(metadata["height_cm"], errors="coerce")
    weight = pd.to_numeric(metadata["weight_kg_baseline"], errors="coerce")
    bmi = weight / (height / 100.0) ** 2
    sex = _shared_sex(metadata["gender"])
    missing_inputs = pd.concat(
        [metadata["age"], sex, bmi, early_minutes, recent_minutes, early_frequency, recent_frequency],
        axis=1,
    )
    frame = pd.DataFrame({
        "age": pd.to_numeric(metadata["age"], errors="coerce"),
        "sex": sex,
        "bmi": bmi,
        "prior_injury_count": pd.to_numeric(
            metadata["prior_season_injury_count"], errors="coerce"
        ),
        "weekly_training_minutes": recent_minutes,
        "weekly_training_frequency": recent_frequency,
        "exposure_change": recent_minutes - early_minutes,
        "exposure_ratio": (recent_minutes + 30.0) / (early_minutes + 30.0),
        "missing_fraction": missing_inputs.isna().mean(axis=1),
    }, index=metadata.index)
    frame.index.name = "athlete_id"
    return frame.loc[:, RUNFITCHECK_FEATURES]


def _records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list) and (not value or isinstance(value[0], dict)):
                return value
    raise ValueError(f"Unsupported JSON record structure in {path}")


def _date(record: dict[str, Any]) -> pd.Timestamp | None:
    for key in ("dateTime", "dateOfSleep", "startTime", "originalStartTime", "start"):
        if key in record:
            value = pd.to_datetime(record[key], errors="coerce")
            if pd.notna(value):
                if getattr(value, "tzinfo", None) is not None:
                    value = value.tz_localize(None)
                return value.normalize()
    return None


def _number(value: Any) -> float:
    if isinstance(value, dict):
        for key in ("value", "bpm", "minutes", "duration"):
            if key in value:
                return _number(value[key])
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _daily_value(path: Path, aggregation: str = "sum") -> pd.Series:
    records = _records(path)
    if not records:
        return pd.Series(dtype=float)
    frame = pd.DataFrame.from_records(records)
    date_keys = tuple(
        key
        for key in ("dateTime", "dateOfSleep", "startTime", "originalStartTime", "start")
        if key in frame
    )
    if not date_keys:
        return pd.Series(dtype=float)
    candidates = frame.reindex(columns=date_keys).to_numpy(dtype=object)
    parsed = pd.to_datetime(candidates.reshape(-1), format="mixed", errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.tz_localize(None)
    if not isinstance(parsed, pd.DatetimeIndex):
        parsed = pd.DatetimeIndex([
            value.tz_localize(None)
            if isinstance(value, pd.Timestamp) and value.tzinfo is not None
            else value
            for value in parsed
        ])
    parsed = parsed.to_numpy().reshape(candidates.shape)
    valid_dates = ~pd.isna(parsed)
    first_valid = valid_dates.argmax(axis=1)
    dates = parsed[np.arange(len(frame)), first_valid]
    dates[~valid_dates.any(axis=1)] = np.datetime64("NaT")
    dates = pd.DatetimeIndex(dates).normalize()

    def scalar(value: Any) -> Any:
        while isinstance(value, dict):
            value = next(
                (value[key] for key in ("value", "bpm", "minutes", "duration") if key in value),
                np.nan,
            )
        return value

    raw_values = frame["value"] if "value" in frame else pd.Series(np.nan, index=frame.index)
    values = pd.to_numeric(raw_values.map(scalar), errors="coerce").to_numpy(dtype=float)
    keep = ~dates.isna() & np.isfinite(values)
    if not keep.any():
        return pd.Series(dtype=float)
    daily = pd.DataFrame({"date": dates[keep], "value": values[keep]})
    grouped = daily.groupby("date").value
    if aggregation == "sum":
        return grouped.sum()
    if aggregation == "mean":
        return grouped.mean()
    raise ValueError(aggregation)


def _daily_sleep(path: Path) -> pd.DataFrame:
    rows = []
    for record in _records(path):
        date = _date(record)
        asleep = _number(record.get("minutesAsleep"))
        in_bed = _number(record.get("timeInBed"))
        efficiency = _number(record.get("efficiency"))
        if np.isfinite(asleep) and in_bed > 0:
            efficiency = asleep / in_bed
        elif np.isfinite(efficiency) and efficiency > 1:
            efficiency /= 100.0
        if date is not None:
            rows.append((date, asleep, in_bed, efficiency))
    if not rows:
        return pd.DataFrame(columns=["TotalMinutesAsleep", "sleep_efficiency"])
    frame = pd.DataFrame(
        rows,
        columns=["date", "TotalMinutesAsleep", "time_in_bed", "sleep_efficiency"],
    )
    return frame.groupby("date").agg(
        TotalMinutesAsleep=("TotalMinutesAsleep", "sum"),
        sleep_efficiency=("sleep_efficiency", "mean"),
    )


def _daily_training(path: Path) -> pd.Series:
    rows = []
    for record in _records(path):
        date = _date(record)
        duration = _number(record.get("activeDuration", record.get("duration")))
        if date is not None and np.isfinite(duration):
            # Fitbit exercise durations are milliseconds; tolerate seconds or minutes in variants.
            hours = duration / 3_600_000.0 if duration > 100_000 else duration / 3600.0
            rows.append((date, hours))
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows, columns=["date", "value"]).groupby("date").value.sum()


def _assign(frame: pd.DataFrame, column: str, values: pd.Series) -> None:
    if not values.empty:
        frame.loc[values.index, column] = values.to_numpy(dtype=float)


def load_pmdata_daily(participant_dir: str | Path) -> pd.DataFrame:
    participant = Path(participant_dir)
    fitbit = participant / "fitbit"
    required = {
        "TotalSteps": ("steps.json", "sum"),
        "Calories": ("calories.json", "sum"),
        "VeryActiveMinutes": ("very_active_minutes.json", "sum"),
        "FairlyActiveMinutes": ("moderately_active_minutes.json", "sum"),
        "LightlyActiveMinutes": ("lightly_active_minutes.json", "sum"),
    }
    available_dates: list[pd.Timestamp] = []
    series: dict[str, pd.Series] = {}
    for column, (filename, aggregation) in required.items():
        path = fitbit / filename
        if path.exists():
            values = _daily_value(path, aggregation)
            series[column] = values
            available_dates.extend(values.index.tolist())
    sleep_path = fitbit / "sleep.json"
    sleep = _daily_sleep(sleep_path) if sleep_path.exists() else pd.DataFrame()
    available_dates.extend(sleep.index.tolist())
    exercise_path = fitbit / "exercise.json"
    training = _daily_training(exercise_path) if exercise_path.exists() else pd.Series(dtype=float)
    available_dates.extend(training.index.tolist())
    if not available_dates:
        raise ValueError(f"No usable PMData records in {participant}")
    index = pd.date_range(min(available_dates), max(available_dates), freq="D")
    frame = pd.DataFrame(index=index, columns=SEQUENCE_CHANNELS, dtype=float)
    for column, values in series.items():
        _assign(frame, column, values)
    if not sleep.empty:
        for column in ("TotalMinutesAsleep", "sleep_efficiency"):
            _assign(frame, column, sleep[column].dropna())
    _assign(frame, "training_hours", training)
    # PMData's selected lightweight files do not provide semantically equivalent
    # daily mean/max heart-rate channels, so those channels remain explicitly missing.
    frame.index.name = "date"
    return frame


def _windows(
    daily: pd.DataFrame,
    subject_id: str,
    length: int,
    stride: int,
    min_day_coverage: float,
) -> Iterable[tuple[np.ndarray, np.ndarray, str, np.datetime64]]:
    values = daily.loc[:, SEQUENCE_CHANNELS].to_numpy(dtype=np.float32)
    observed = np.isfinite(values)
    for start in range(0, len(daily) - length + 1, stride):
        stop = start + length
        day_coverage = observed[start:stop].any(axis=1).mean()
        if day_coverage < min_day_coverage:
            continue
        yield (
            values[start:stop].T,
            observed[start:stop].T,
            subject_id,
            daily.index[stop - 1].to_datetime64(),
        )


def load_pmdata_sequences(
    root: str | Path,
    length: int = 30,
    stride: int = 3,
    min_day_coverage: float = 0.70,
) -> ExternalSequences:
    root = Path(root)
    items = []
    for participant in sorted(path for path in root.glob("p[0-9][0-9]") if path.is_dir()):
        daily = load_pmdata_daily(participant)
        items.extend(_windows(daily, participant.name, length, stride, min_day_coverage))
    if not items:
        raise ValueError(f"No valid {length}-day PMData windows found under {root}")
    values, observed, subjects, dates = zip(*items)
    return ExternalSequences(
        values=np.stack(values),
        observed=np.stack(observed),
        subject_ids=np.asarray(subjects),
        end_dates=np.asarray(dates),
    )
