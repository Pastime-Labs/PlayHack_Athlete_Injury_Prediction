from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class Experiment:
    name: str
    features: str
    classifier: str
    onset: str
    recovery: str


def experiments() -> list[Experiment]:
    return [
        Experiment("01_global_baseline", "basic", "prior", "mean", "mean"),
        Experiment("02_sport_baseline", "basic", "sport_prior", "mean", "sport_median"),
        Experiment("03_logistic_basic", "basic", "logistic", "ridge", "sport_median"),
        Experiment("04_logistic_temporal", "temporal", "logistic", "ridge", "sport_median"),
        Experiment("05_logistic_balanced", "temporal", "logistic_balanced", "ridge", "sport_median"),
        Experiment("06_logistic_mechanistic", "mechanistic", "logistic", "mechanistic", "sport_median"),
        Experiment("07_extra_basic", "basic", "extra_regular", "extra", "sport_median"),
        Experiment("08_extra_temporal", "temporal", "extra_regular", "extra", "sport_median"),
        Experiment("09_extra_structural", "structural", "extra_regular", "extra", "sport_median"),
        Experiment("10_extra_fine", "structural", "extra_fine", "extra_fine", "sport_median"),
        Experiment("11_hist_temporal", "temporal", "hist", "hist", "sport_median"),
        Experiment("12_hist_structural", "structural", "hist", "hist", "sport_median"),
        Experiment("13_gradient_structural", "structural", "gradient", "extra", "sport_median"),
        Experiment("14_forest_structural", "structural", "forest", "extra", "sport_median"),
        Experiment("15_activity_expert", "activity_view", "logistic", "mechanistic", "sport_median"),
        Experiment("16_recovery_expert", "recovery_view", "logistic", "ridge", "sport_median"),
    ]


def _preprocessor(frame: pd.DataFrame, scale: bool) -> ColumnTransformer:
    categorical = frame.select_dtypes(exclude="number").columns.tolist()
    numeric = [c for c in frame if c not in categorical]
    numeric_steps = [SimpleImputer(strategy="median")]
    if scale:
        numeric_steps.append(StandardScaler())
    return ColumnTransformer([
        ("num", make_pipeline(*numeric_steps), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])


def _classifier(kind: str, frame: pd.DataFrame, seed: int):
    scale = kind.startswith("logistic")
    pre = _preprocessor(frame, scale=scale)
    if kind == "logistic":
        model = LogisticRegression(C=0.12, max_iter=3000)
    elif kind == "logistic_balanced":
        model = LogisticRegression(C=0.12, class_weight="balanced", max_iter=3000)
    elif kind == "extra_regular":
        model = ExtraTreesClassifier(n_estimators=180, min_samples_leaf=7, max_features=0.65, class_weight="balanced", n_jobs=1, random_state=seed)
    elif kind == "extra_fine":
        model = ExtraTreesClassifier(n_estimators=180, min_samples_leaf=3, max_features=0.8, class_weight="balanced", n_jobs=1, random_state=seed)
    elif kind == "hist":
        model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.035, max_leaf_nodes=12, min_samples_leaf=25, l2_regularization=10, random_state=seed)
    elif kind == "gradient":
        model = GradientBoostingClassifier(n_estimators=160, learning_rate=0.03, max_depth=2, min_samples_leaf=15, subsample=0.75, random_state=seed)
    elif kind == "forest":
        model = RandomForestClassifier(n_estimators=180, min_samples_leaf=6, max_features=0.55, class_weight="balanced_subsample", n_jobs=1, random_state=seed)
    else:
        raise KeyError(kind)
    return make_pipeline(pre, model)


def _regressor(kind: str, frame: pd.DataFrame, seed: int):
    scale = kind == "ridge"
    pre = _preprocessor(frame, scale=scale)
    if kind == "ridge":
        model = Ridge(alpha=100.0)
    elif kind == "extra":
        model = ExtraTreesRegressor(n_estimators=180, min_samples_leaf=8, max_features=0.7, n_jobs=1, random_state=seed)
    elif kind == "extra_fine":
        model = ExtraTreesRegressor(n_estimators=180, min_samples_leaf=4, max_features=0.8, n_jobs=1, random_state=seed)
    elif kind == "hist":
        model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=220, learning_rate=0.035, max_leaf_nodes=10, min_samples_leaf=20, l2_regularization=12, random_state=seed)
    else:
        raise KeyError(kind)
    return make_pipeline(pre, model)


def _best_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.10, 0.85, 151)
    scores = np.array([f1_score(y, probability >= threshold) for threshold in candidates])
    index = int(scores.argmax())
    return float(candidates[index]), float(scores[index])


def score_predictions(
    y_class: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    true_onset: np.ndarray,
    true_recovery: np.ndarray,
    pred_onset: np.ndarray,
    pred_recovery: np.ndarray,
    baseline_onset: np.ndarray,
    baseline_recovery: np.ndarray,
) -> dict[str, float]:
    pred_class = probability >= threshold
    positive = y_class.astype(bool)
    onset_hit_error = np.abs(pred_onset[positive] - true_onset[positive])
    recovery_hit_error = np.abs(pred_recovery[positive] - true_recovery[positive])
    onset_error = np.where(pred_class[positive], onset_hit_error, 30.0)
    recovery_error = np.where(pred_class[positive], recovery_hit_error, 30.0)
    onset_baseline_mae = mean_absolute_error(true_onset[positive], baseline_onset[positive])
    recovery_baseline_mae = mean_absolute_error(true_recovery[positive], baseline_recovery[positive])
    return {
        "threshold": float(threshold),
        "f1": f1_score(y_class, pred_class),
        "recall": recall_score(y_class, pred_class),
        "roc_auc": roc_auc_score(y_class, probability),
        "pr_auc": average_precision_score(y_class, probability),
        "onset_mae_raw": float(onset_hit_error.mean()),
        "recovery_mae_raw": float(recovery_hit_error.mean()),
        "onset_mae_penalized": float(onset_error.mean()),
        "recovery_mae_penalized": float(recovery_error.mean()),
        "onset_skill": max(0.0, 1.0 - float(onset_error.mean()) / onset_baseline_mae),
        "recovery_skill": max(0.0, 1.0 - float(recovery_error.mean()) / recovery_baseline_mae),
    }


def _mechanistic_onset(train_x: pd.DataFrame, train_y: pd.Series, valid_x: pd.DataFrame) -> np.ndarray:
    cp_col, gain_col = "strain_cp", "strain_gain"
    direct_train = np.clip(train_x[cp_col].to_numpy() - 13.0, 1.0, 17.0)
    direct_valid = np.clip(valid_x[cp_col].to_numpy() - 13.0, 1.0, 17.0)
    best_mae, best_threshold, best_fallback = np.inf, 0.0, float(train_y.median())
    for threshold in np.quantile(train_x[gain_col], np.linspace(0.2, 0.9, 15)):
        weak = train_x[gain_col].to_numpy() < threshold
        fallback = float(train_y.loc[weak].median()) if weak.any() else float(train_y.median())
        pred = np.where(weak, fallback, direct_train)
        score = mean_absolute_error(train_y, pred)
        if score < best_mae:
            best_mae, best_threshold, best_fallback = score, float(threshold), fallback
    return np.where(valid_x[gain_col].to_numpy() < best_threshold, best_fallback, direct_valid)


def run_experiment(
    spec: Experiment,
    feature_sets: dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    metadata: pd.DataFrame,
    development_ids: np.ndarray,
    fold_map: dict[int, int],
    seed: int = 2026,
) -> tuple[dict[str, float], pd.DataFrame]:
    x = feature_sets[spec.features].loc[development_ids]
    target = labels.set_index("athlete_id").loc[development_ids]
    sport = metadata.set_index("athlete_id").loc[development_ids, "sport"]
    probability = np.zeros(len(x)); onset = np.zeros(len(x)); recovery = np.zeros(len(x))
    onset_baseline = np.zeros(len(x)); recovery_baseline = np.zeros(len(x))
    folds = np.array([fold_map[int(i)] for i in development_ids])

    for fold in sorted(np.unique(folds)):
        train_mask, valid_mask = folds != fold, folds == fold
        train_x, valid_x = x.iloc[train_mask], x.iloc[valid_mask]
        train_y, valid_y = target.iloc[train_mask], target.iloc[valid_mask]
        train_sport, valid_sport = sport.iloc[train_mask], sport.iloc[valid_mask]

        if spec.classifier == "prior":
            probability[valid_mask] = train_y.injured_in_risk_window.mean()
        elif spec.classifier == "sport_prior":
            priors = train_y.assign(sport=train_sport).groupby("sport").injured_in_risk_window.mean()
            probability[valid_mask] = valid_sport.map(priors).fillna(train_y.injured_in_risk_window.mean())
        else:
            classifier = _classifier(spec.classifier, train_x, seed + int(fold))
            classifier.fit(train_x, train_y.injured_in_risk_window)
            probability[valid_mask] = classifier.predict_proba(valid_x)[:, 1]

        positive = train_y.injured_in_risk_window.eq(1)
        positive_x = train_x.loc[positive]
        onset_baseline[valid_mask] = train_y.loc[positive, "onset_day_offset"].mean()
        recovery_baseline[valid_mask] = train_y.loc[positive, "recovery_duration"].mean()
        if spec.onset == "mean":
            onset[valid_mask] = train_y.loc[positive, "onset_day_offset"].mean()
        elif spec.onset == "mechanistic":
            onset[valid_mask] = _mechanistic_onset(
                positive_x,
                train_y.loc[positive, "onset_day_offset"],
                valid_x,
            )
        else:
            model = _regressor(spec.onset, positive_x, seed + int(fold))
            model.fit(positive_x, train_y.loc[positive, "onset_day_offset"])
            onset[valid_mask] = model.predict(valid_x)

        positive_recovery = train_y.loc[positive, "recovery_duration"]
        if spec.recovery == "mean":
            recovery[valid_mask] = positive_recovery.mean()
        elif spec.recovery == "sport_median":
            medians = train_y.loc[positive].assign(sport=train_sport.loc[positive]).groupby("sport").recovery_duration.median()
            recovery[valid_mask] = valid_sport.map(medians).fillna(positive_recovery.median())
        else:
            model = _regressor(spec.recovery, positive_x, seed + int(fold))
            model.fit(positive_x, positive_recovery)
            recovery[valid_mask] = model.predict(valid_x)

    y_class = target.injured_in_risk_window.to_numpy(dtype=int)
    threshold, best_f1 = _best_threshold(y_class, probability)
    pred_class = (probability >= threshold).astype(int)
    positive = y_class == 1
    onset = np.clip(onset, 1, 30)
    recovery = np.clip(recovery, 5, 20)
    onset_error = np.where(pred_class[positive] == 1, np.abs(onset[positive] - target.onset_day_offset.to_numpy()[positive]), 30.0)
    recovery_error = np.where(pred_class[positive] == 1, np.abs(recovery[positive] - target.recovery_duration.to_numpy()[positive]), 30.0)
    baseline_onset_mae = mean_absolute_error(target.onset_day_offset.to_numpy()[positive], onset_baseline[positive])
    baseline_recovery_mae = mean_absolute_error(target.recovery_duration.to_numpy()[positive], recovery_baseline[positive])
    metrics = {
        "f1": best_f1,
        "threshold": threshold,
        "recall": recall_score(y_class, pred_class),
        "roc_auc": roc_auc_score(y_class, probability),
        "pr_auc": average_precision_score(y_class, probability),
        "onset_mae_raw": mean_absolute_error(target.onset_day_offset.to_numpy()[positive], onset[positive]),
        "recovery_mae_raw": mean_absolute_error(target.recovery_duration.to_numpy()[positive], recovery[positive]),
        "onset_mae_penalized": float(onset_error.mean()),
        "recovery_mae_penalized": float(recovery_error.mean()),
        "onset_skill": max(0.0, 1.0 - float(onset_error.mean()) / baseline_onset_mae),
        "recovery_skill": max(0.0, 1.0 - float(recovery_error.mean()) / baseline_recovery_mae),
    }
    metrics["research_score"] = metrics["f1"] + 0.25 * (metrics["onset_skill"] + metrics["recovery_skill"])
    oof = pd.DataFrame({
        "athlete_id": development_ids,
        "fold": folds,
        "y_injury": y_class,
        "y_onset": target.onset_day_offset.to_numpy(),
        "y_recovery": target.recovery_duration.to_numpy(),
        "p_injury": probability,
        "pred_injury": pred_class,
        "pred_onset": onset,
        "pred_recovery": recovery,
    })
    return metrics, oof


def save_run(root: Path, spec: Experiment, metrics: dict[str, float], oof: pd.DataFrame) -> None:
    run_dir = root / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    oof.to_csv(run_dir / "oof.csv", index=False)
