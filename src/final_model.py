from __future__ import annotations

import hashlib
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score

from run_tcn import HazardTCN, hazard_loss, predict as predict_tcn
from src.data import CUTOFF
from src.evaluation import _best_threshold, _classifier, _regressor
from src.features import build_features, build_sequence


SEED = 2026
ONSET_HIST_WEIGHT = 0.65


def _sequence_ids(bundle: dict[str, object]) -> np.ndarray:
    return bundle["daily"]["athlete_id"].drop_duplicates().to_numpy(dtype=int)


def _normalise_sequence(sequence: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    median = np.nanmedian(sequence, axis=(0, 2))
    mean = np.nanmean(sequence, axis=(0, 2))
    std = np.nanstd(sequence, axis=(0, 2)) + 1e-6
    values = np.where(np.isfinite(sequence), sequence, median[None, :, None])
    values = (values - mean[None, :, None]) / std[None, :, None]
    return values.astype(np.float32), {"median": median, "mean": mean, "std": std}


def _apply_sequence_stats(sequence: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    values = np.where(np.isfinite(sequence), sequence, stats["median"][None, :, None])
    return ((values - stats["mean"][None, :, None]) / stats["std"][None, :, None]).astype(np.float32)


def _fit_tcn(sequence: np.ndarray, event: np.ndarray, day: np.ndarray, epochs: int) -> HazardTCN:
    torch.manual_seed(SEED)
    model = HazardTCN(sequence.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    rng = np.random.default_rng(SEED)
    for _ in range(epochs):
        model.train()
        for batch in np.array_split(rng.permutation(len(sequence)), max(1, len(sequence) // 64)):
            logits = model(torch.tensor(sequence[batch], dtype=torch.float32))
            loss = hazard_loss(logits, torch.tensor(event[batch]), torch.tensor(day[batch]))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
    return model


def _threshold_policies(oof_path: str | Path) -> dict[str, float]:
    oof = pd.read_csv(oof_path)
    y = oof.y_injury.to_numpy(dtype=int)
    probability = oof.p_injury.to_numpy(dtype=float)
    f1_threshold, _ = _best_threshold(y, probability)
    candidates = np.linspace(0.0, 1.0, 1001)
    eligible = [
        (f1_score(y, probability >= threshold), threshold)
        for threshold in candidates
        if recall_score(y, probability >= threshold) >= 0.99
    ]
    recall_threshold = max(eligible)[1]
    return {"f1": float(f1_threshold), "recall99": float(recall_threshold)}


def fit_final(bundle: dict[str, object], classifier_oof: str | Path, tcn_epochs: int = 90) -> dict[str, object]:
    if "labels" not in bundle:
        raise ValueError("Final training requires train_labels.csv")
    labels = bundle["labels"].set_index("athlete_id").sort_index()
    ids = labels.index.to_numpy(dtype=int)
    features = build_features(bundle)["structural"].loc[ids]

    classifier = _classifier("extra_regular", features, SEED)
    classifier.fit(features, labels.injured_in_risk_window)
    positive = labels.injured_in_risk_window.eq(1)
    onset_model = _regressor("hist", features.loc[positive], SEED)
    onset_model.fit(features.loc[positive], labels.loc[positive, "onset_day_offset"])

    metadata = bundle["metadata"].set_index("athlete_id").loc[ids]
    recovery_by_sport = (
        labels.loc[positive]
        .assign(sport=metadata.loc[positive, "sport"])
        .groupby("sport")
        .recovery_duration.median()
        .to_dict()
    )
    recovery_default = float(labels.loc[positive, "recovery_duration"].median())

    raw_sequence, channels = build_sequence(bundle)
    sequence_ids = _sequence_ids(bundle)
    if not np.array_equal(sequence_ids, ids):
        position = {athlete_id: index for index, athlete_id in enumerate(sequence_ids)}
        raw_sequence = raw_sequence[[position[athlete_id] for athlete_id in ids]]
    sequence, sequence_stats = _normalise_sequence(raw_sequence.transpose(0, 2, 1).astype(float))
    event = labels.injured_in_risk_window.to_numpy(dtype=bool)
    day = labels.onset_day_offset.fillna(30).to_numpy(dtype=int)
    tcn = _fit_tcn(sequence, event, day, tcn_epochs)

    return {
        "version": 1,
        "cutoff": str(CUTOFF.date()),
        "seed": SEED,
        "training_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "feature_columns": features.columns.tolist(),
        "classifier": classifier,
        "onset_model": onset_model,
        "onset_hist_weight": ONSET_HIST_WEIGHT,
        "recovery_by_sport": recovery_by_sport,
        "recovery_default": recovery_default,
        "sequence_channels": channels,
        "sequence_stats": sequence_stats,
        "tcn_state": {name: value.detach().cpu().numpy() for name, value in tcn.state_dict().items()},
        "thresholds": _threshold_policies(classifier_oof),
        "tcn_epochs": tcn_epochs,
    }


def predict_bundle(artifact: dict[str, object], bundle: dict[str, object], policy: str = "f1") -> pd.DataFrame:
    if policy not in artifact["thresholds"]:
        raise ValueError(f"Unknown policy {policy!r}; choose from {sorted(artifact['thresholds'])}")
    features = build_features(bundle)["structural"]
    ids = features.index.to_numpy(dtype=int)
    expected = artifact["feature_columns"]
    if set(features.columns) != set(expected):
        missing = sorted(set(expected) - set(features.columns))
        extra = sorted(set(features.columns) - set(expected))
        raise ValueError(f"Feature schema changed; missing={missing}, extra={extra}")
    features = features[expected]
    probability = artifact["classifier"].predict_proba(features)[:, 1]
    hist_onset = artifact["onset_model"].predict(features)

    raw_sequence, channels = build_sequence(bundle)
    if channels != artifact["sequence_channels"]:
        raise ValueError("Sequence channel schema changed")
    sequence_ids = _sequence_ids(bundle)
    if not np.array_equal(sequence_ids, ids):
        position = {athlete_id: index for index, athlete_id in enumerate(sequence_ids)}
        raw_sequence = raw_sequence[[position[athlete_id] for athlete_id in ids]]
    sequence = _apply_sequence_stats(raw_sequence.transpose(0, 2, 1).astype(float), artifact["sequence_stats"])
    tcn = HazardTCN(sequence.shape[1])
    tcn.load_state_dict({name: torch.from_numpy(value) for name, value in artifact["tcn_state"].items()})
    _, tcn_onset = predict_tcn(tcn, sequence)
    weight = artifact["onset_hist_weight"]
    onset = np.rint(np.clip(weight * hist_onset + (1 - weight) * tcn_onset, 1, 30)).astype(int)

    sport = bundle["metadata"].set_index("athlete_id").loc[ids, "sport"]
    recovery = np.rint(
        sport.map(artifact["recovery_by_sport"]).fillna(artifact["recovery_default"])
    ).clip(5, 20).astype(int)
    threshold = artifact["thresholds"][policy]
    return pd.DataFrame({
        "athlete_id": ids,
        "injured_in_risk_window": (probability >= threshold).astype(int),
        "onset_day_offset": onset,
        "recovery_duration": recovery.to_numpy(),
        "injury_probability": probability,
    })


def validate_submission(frame: pd.DataFrame, expected_ids: np.ndarray | None = None) -> None:
    required = ["athlete_id", "injured_in_risk_window", "onset_day_offset", "recovery_duration"]
    if frame.columns.tolist() != required:
        raise ValueError(f"Submission columns must be exactly {required}")
    if frame.athlete_id.duplicated().any() or frame.isna().any().any():
        raise ValueError("Submission contains duplicate IDs or missing values")
    if expected_ids is not None and set(frame.athlete_id) != set(np.asarray(expected_ids)):
        raise ValueError("Submission athlete IDs do not match the input data")
    if not frame.injured_in_risk_window.isin([0, 1]).all():
        raise ValueError("Injury predictions must be binary")
    if not frame.onset_day_offset.between(1, 30).all():
        raise ValueError("Onset predictions must be within 1..30")
    if not frame.recovery_duration.between(5, 20).all():
        raise ValueError("Recovery predictions must be within 5..20")
    if not all(np.issubdtype(frame[column].dtype, np.integer) for column in required):
        raise ValueError("All submission columns must contain integers")
