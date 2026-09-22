from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


CUTOFF = pd.Timestamp("2026-02-04")
CACHE_VERSION = 4


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(data_dir: Path) -> str:
    payload = {
        "version": CACHE_VERSION,
        "cutoff": str(CUTOFF.date()),
        "files": {path.name: _file_digest(path) for path in sorted(data_dir.glob("*.csv"))},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _hourly_daily(path: Path, time_col: str, value_cols: list[str]) -> pd.DataFrame:
    partials: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=300_000):
        timestamp = pd.to_datetime(chunk[time_col], format="mixed", errors="coerce")
        chunk = chunk.loc[timestamp.notna() & (timestamp < CUTOFF), ["Id", *value_cols]].copy()
        timestamp = timestamp.loc[chunk.index]
        chunk["date"] = timestamp.dt.normalize()
        keys = ["Id", "date"]
        squared = []
        for col in value_cols:
            sq = f"_{col}_sq"
            chunk[sq] = chunk[col].astype(float) ** 2
            squared.append(sq)
        grouped = chunk.groupby(keys)
        out = grouped[value_cols].agg(["sum", "count", "min", "max"])
        out.columns = [f"{col}_{stat}" for col, stat in out.columns]
        out["hours_observed"] = grouped.size()
        for col, sq in zip(value_cols, squared):
            out[f"{col}_sum_sq"] = grouped[sq].sum()
        partials.append(out.reset_index())

    merged = pd.concat(partials, ignore_index=True)
    sum_cols = [c for c in merged if c not in {"Id", "date"} and not c.endswith(("_min", "_max"))]
    min_cols = [c for c in merged if c.endswith("_min")]
    max_cols = [c for c in merged if c.endswith("_max")]
    aggregations = {c: "sum" for c in sum_cols} | {c: "min" for c in min_cols} | {c: "max" for c in max_cols}
    out = merged.groupby(["Id", "date"], as_index=False).agg(aggregations)
    for col in value_cols:
        count = out[f"{col}_count"].clip(lower=1)
        mean = out[f"{col}_sum"] / count
        variance = (out[f"{col}_sum_sq"] / count - mean**2).clip(lower=0)
        out[f"{col}_mean"] = mean
        out[f"{col}_std"] = np.sqrt(variance)
        out.drop(columns=[f"{col}_sum_sq", f"{col}_count"], inplace=True)
    return out.rename(columns={"Id": "athlete_id"})


def _build_bundle(data_dir: Path) -> dict[str, object]:
    metadata = pd.read_csv(data_dir / "athlete_metadata.csv").sort_values("athlete_id")
    labels_path = data_dir / "train_labels.csv"
    labels = pd.read_csv(labels_path).sort_values("athlete_id") if labels_path.exists() else None

    activity = pd.read_csv(data_dir / "dailyActivity_merged.csv", parse_dates=["ActivityDate"])
    activity = activity.loc[activity.ActivityDate < CUTOFF].rename(
        columns={"Id": "athlete_id", "ActivityDate": "date"}
    ).sort_values(["athlete_id", "date"])
    activity["day"] = activity.groupby("athlete_id").cumcount() + 1

    sleep = pd.read_csv(data_dir / "sleepDay_merged.csv", parse_dates=["SleepDay"])
    sleep = sleep.loc[sleep.SleepDay < CUTOFF].rename(columns={"Id": "athlete_id", "SleepDay": "date"})
    sleep["sleep_efficiency"] = sleep.TotalMinutesAsleep / sleep.TotalTimeInBed.replace(0, np.nan)

    sessions = pd.read_csv(data_dir / "training_sessions.csv", parse_dates=["date"])
    sessions = sessions.loc[sessions.date < CUTOFF].copy()
    sessions["training_hours"] = sessions.end_hour - sessions.start_hour
    session_daily = sessions.groupby(["athlete_id", "date"]).agg(
        training_hours=("training_hours", "sum"),
        session_count=("session_id", "size"),
        practice_count=("sport_session_type", lambda x: int((x == "practice").sum())),
        gym_count=("sport_session_type", lambda x: int((x == "gym").sum())),
        scrimmage_count=("sport_session_type", lambda x: int((x == "scrimmage").sum())),
    ).reset_index()

    daily = activity.merge(sleep, on=["athlete_id", "date"], how="left").merge(
        session_daily, on=["athlete_id", "date"], how="left"
    )
    session_cols = ["training_hours", "session_count", "practice_count", "gym_count", "scrimmage_count"]
    daily[session_cols] = daily[session_cols].fillna(0)

    # Daily files already contain steps, calories and intensity. Heart rate is the
    # only nonredundant hourly modality required by the core experiment suite.
    hourly_specs = [
        ("hourlyHeartrate_merged.csv", "ActivityHour", ["AvgHeartRate", "MinHeartRate", "MaxHeartRate"], "hr"),
    ]
    for filename, time_col, value_cols, prefix in hourly_specs:
        hourly = _hourly_daily(data_dir / filename, time_col, value_cols)
        hourly = hourly.rename(columns={c: f"{prefix}_{c}" for c in hourly if c not in {"athlete_id", "date"}})
        daily = daily.merge(hourly, on=["athlete_id", "date"], how="left")

    weight = pd.read_csv(data_dir / "weightLogInfo_merged.csv", parse_dates=["Date"])
    weight = weight.loc[weight.Date < CUTOFF].sort_values(["Id", "Date"])
    weight_latest = weight.groupby("Id").tail(1).set_index("Id")
    weight_counts = weight.groupby("Id").size()
    metadata = metadata.set_index("athlete_id")
    metadata["legal_weight_records"] = weight_counts
    metadata["legal_weight_missing"] = (~metadata.index.isin(weight_counts.index)).astype(int)
    for source, target in [("WeightKg", "latest_weight_kg"), ("Fat", "latest_fat"), ("BMI", "latest_bmi")]:
        metadata[target] = weight_latest[source]
    metadata = metadata.reset_index()

    if daily.groupby("athlete_id").size().ne(30).any():
        raise ValueError("Every athlete must have exactly 30 legal observation days")
    if daily.date.max() >= CUTOFF or not daily[["athlete_id", "date"]].duplicated().sum() == 0:
        raise ValueError("Observation cutoff or athlete-day uniqueness violated")
    if set(daily.athlete_id) != set(metadata.athlete_id):
        raise ValueError("Metadata/observation athlete IDs differ")
    if labels is not None and set(labels.athlete_id) != set(metadata.athlete_id):
        raise ValueError("Metadata/label athlete IDs differ")

    numeric_channels = [
        c for c in daily.select_dtypes(include="number").columns
        if c not in {"athlete_id", "day", "SedentaryMinutes"}
    ]
    bundle = {
        "metadata": metadata.reset_index(drop=True),
        "daily": daily.sort_values(["athlete_id", "day"]).reset_index(drop=True),
        "channels": numeric_channels,
    }
    if labels is not None:
        bundle["labels"] = labels.reset_index(drop=True)
    return bundle


def load_bundle(data_dir: str | Path = "DataSet", cache_dir: str | Path = "cache") -> dict[str, object]:
    data_path, cache_path = Path(data_dir), Path(cache_dir)
    fingerprint = _fingerprint(data_path)
    target = cache_path / fingerprint / "bundle.joblib"
    if target.exists():
        return joblib.load(target)
    bundle = _build_bundle(data_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, target, compress=3)
    (target.parent / "manifest.json").write_text(
        json.dumps({"fingerprint": fingerprint, "cutoff": str(CUTOFF.date())}, indent=2), encoding="utf-8"
    )
    return bundle
