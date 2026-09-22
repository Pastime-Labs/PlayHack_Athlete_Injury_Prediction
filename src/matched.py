from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error


class MatchedEventModel:
    """Fold-fitted weekly detrending plus 31-hypothesis event matching."""

    def __init__(self, directions: np.ndarray):
        self.directions = np.asarray(directions, dtype=float)

    def _fit_seasonality(self, sequence: np.ndarray) -> None:
        early = sequence[:, :12, :].astype(float)
        initial_level = np.nanmedian(early, axis=1)
        normalized = early / (np.abs(initial_level[:, None, :]) + 1e-6)
        factors = np.ones((7, sequence.shape[2]), dtype=float)
        for weekday in range(7):
            positions = [i for i in range(12) if i % 7 == weekday]
            factors[weekday] = np.nanmedian(normalized[:, positions, :], axis=(0, 1))
        self.weekday_factors_ = np.where(np.isfinite(factors), factors, 1.0)
        expected_factor = self.weekday_factors_[np.arange(12) % 7]
        self.training_levels_ = np.nanmedian(early / (expected_factor[None, :, :] + 1e-6), axis=1)

    def _residuals(self, sequence: np.ndarray) -> np.ndarray:
        early = sequence[:, :12, :].astype(float)
        early_factor = self.weekday_factors_[np.arange(12) % 7]
        levels = np.nanmedian(early / (early_factor[None, :, :] + 1e-6), axis=1)
        factors = self.weekday_factors_[np.arange(30) % 7]
        expected = levels[:, None, :] * factors[None, :, :]
        residual = (sequence - expected) / (np.abs(levels[:, None, :]) + 1e-6)
        return np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, sequence: np.ndarray, injury: np.ndarray, onset: np.ndarray) -> "MatchedEventModel":
        self._fit_seasonality(sequence)
        residual = self._residuals(sequence)
        visible = (injury == 1) & (onset <= 17)
        slopes = []
        days = np.arange(1, 31, dtype=float)
        for values, event_day in zip(residual[visible], onset[visible]):
            ramp = np.maximum(0.0, days - (13.0 + event_day) + 1.0)
            slopes.append(np.sum(values * ramp[:, None], axis=0) / (np.sum(ramp**2) + 1e-6))
        slope = np.nanmedian(np.asarray(slopes), axis=0)
        self.template_slope_ = np.abs(slope) * self.directions

        baseline = residual[:, :12, :].reshape(-1, residual.shape[2])
        center = np.nanmedian(baseline, axis=0)
        noise = np.nanmedian(np.abs(baseline - center), axis=0) * 1.4826
        self.noise_ = np.maximum(np.where(np.isfinite(noise), noise, 0.0), 0.01)

        features, hypothesis = self._features(sequence)
        self.calibrator_ = LogisticRegression(C=0.25, class_weight="balanced", max_iter=2000)
        self.calibrator_.fit(features, injury)

        positive_features = features[injury == 1]
        positive_hypothesis = hypothesis[injury == 1]
        positive_onset = onset[injury == 1]
        evidence = positive_features[:, 0]
        best_day = positive_hypothesis[:, :17].argmax(axis=1) + 1
        late = positive_onset > 17
        default_fallback = float(np.median(positive_onset[late])) if late.any() else float(np.median(positive_onset))
        self.onset_threshold_, self.onset_fallback_ = -np.inf, default_fallback
        best_mae = np.inf
        for threshold in np.quantile(evidence, np.linspace(0.1, 0.9, 33)):
            weak = evidence < threshold
            fallback = float(np.median(positive_onset[weak])) if weak.any() else default_fallback
            prediction = np.where(weak, fallback, best_day)
            mae = mean_absolute_error(positive_onset, prediction)
            if mae < best_mae:
                best_mae, self.onset_threshold_, self.onset_fallback_ = mae, float(threshold), fallback
        return self

    def _hypotheses(self, sequence: np.ndarray) -> np.ndarray:
        residual = self._residuals(sequence)
        days = np.arange(1, 31, dtype=float)
        null_sse = np.sum((residual / self.noise_[None, None, :]) ** 2, axis=(1, 2))
        scores = []
        for onset in range(1, 31):
            ramp = np.maximum(0.0, days - (13.0 + onset) + 1.0)
            template = ramp[:, None] * self.template_slope_[None, :]
            event_sse = np.sum(((residual - template[None, :, :]) / self.noise_[None, None, :]) ** 2, axis=(1, 2))
            scores.append(0.5 * (null_sse - event_sse))
        return np.column_stack(scores)

    def _features(self, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = self._hypotheses(sequence)
        visible = scores[:, :17]
        ordered = np.sort(visible, axis=1)
        features = np.column_stack([
            visible.max(axis=1),
            logsumexp(visible, axis=1),
            ordered[:, -1] - ordered[:, -2],
            visible.argmax(axis=1) + 1,
            np.maximum(visible, 0).sum(axis=1),
        ])
        return features, scores

    def predict(self, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features, scores = self._features(sequence)
        probability = self.calibrator_.predict_proba(features)[:, 1]
        best_day = scores[:, :17].argmax(axis=1) + 1
        onset = np.where(features[:, 0] < self.onset_threshold_, self.onset_fallback_, best_day).astype(float)
        return probability, onset, scores

