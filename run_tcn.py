from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.data import load_bundle
from src.evaluation import _best_threshold, Experiment, save_run, score_predictions
from src.features import build_sequence


torch.set_num_threads(1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Dropout(0.20),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GELU(), nn.Dropout(0.20),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class HazardTCN(nn.Module):
    def __init__(self, inputs: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(inputs, 32, 1), nn.GELU(),
            ResidualBlock(32, 1), ResidualBlock(32, 2), ResidualBlock(32, 4),
        )
        self.hazard = nn.Sequential(nn.Linear(64, 48), nn.GELU(), nn.Dropout(0.15), nn.Linear(48, 30))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        pooled = torch.cat([encoded.mean(dim=2), encoded.amax(dim=2)], dim=1)
        return self.hazard(pooled)


def hazard_loss(logits: torch.Tensor, event: torch.Tensor, day: torch.Tensor) -> torch.Tensor:
    time = torch.arange(1, 31, device=logits.device)[None, :]
    before = time < day[:, None]
    at_event = (time == day[:, None]) & event[:, None]
    survived = before | ((~event[:, None]) & (time <= 30))
    loss = F.softplus(logits) * survived + F.softplus(-logits) * at_event
    nll = loss.sum(dim=1).mean()
    hazard = torch.sigmoid(logits)
    survival_before = torch.cumprod(torch.cat([torch.ones((len(logits), 1), device=logits.device), 1 - hazard[:, :-1]], dim=1), dim=1)
    mass = hazard * survival_before
    conditional = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
    mean_day = (conditional * time).sum(dim=1)
    auxiliary = F.smooth_l1_loss(mean_day[event] / 30.0, day[event] / 30.0) if event.any() else 0.0
    return nll + 0.25 * auxiliary


def predict(model: nn.Module, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(sequence, dtype=torch.float32))
        hazard = torch.sigmoid(logits)
        survival_before = torch.cumprod(torch.cat([torch.ones((len(logits), 1)), 1 - hazard[:, :-1]], dim=1), dim=1)
        mass = hazard * survival_before
        event_probability = 1 - torch.prod(1 - hazard, dim=1)
        conditional = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
        cdf = torch.cumsum(conditional, dim=1)
        onset = (cdf < 0.5).sum(dim=1) + 1
    return event_probability.numpy(), onset.numpy().astype(float)


def main() -> None:
    seed = 2026
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    bundle = load_bundle()
    sequence, channels = build_sequence(bundle)
    sequence = sequence.transpose(0, 2, 1)
    all_ids = bundle["labels"].athlete_id.to_numpy(); position = {int(v): i for i, v in enumerate(all_ids)}
    labels = bundle["labels"].set_index("athlete_id"); metadata = bundle["metadata"].set_index("athlete_id")
    splits = pd.read_csv("cache/representations/splits.csv")
    ids = splits.loc[splits.split == "development", "athlete_id"].astype(int).to_numpy()
    folds = splits.set_index("athlete_id").loc[ids, "fold"].astype(int).to_numpy(); target = labels.loc[ids]
    probability = np.zeros(len(ids)); onset = np.zeros(len(ids)); recovery = np.zeros(len(ids))
    onset_baseline = np.zeros(len(ids)); recovery_baseline = np.zeros(len(ids))

    for fold in sorted(np.unique(folds)):
        train_mask, valid_mask = folds != fold, folds == fold
        train_ids, valid_ids = ids[train_mask], ids[valid_mask]
        train = sequence[[position[int(i)] for i in train_ids]].astype(float)
        valid = sequence[[position[int(i)] for i in valid_ids]].astype(float)
        median = np.nanmedian(train, axis=(0, 2)); mean = np.nanmean(train, axis=(0, 2)); std = np.nanstd(train, axis=(0, 2)) + 1e-6
        train = np.where(np.isfinite(train), train, median[None, :, None]); valid = np.where(np.isfinite(valid), valid, median[None, :, None])
        train = (train - mean[None, :, None]) / std[None, :, None]; valid = (valid - mean[None, :, None]) / std[None, :, None]
        train_y = target.iloc[train_mask]
        event = train_y.injured_in_risk_window.to_numpy().astype(bool)
        day = train_y.onset_day_offset.fillna(30).to_numpy().astype(int)
        model = HazardTCN(train.shape[1]); optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
        rng = np.random.default_rng(seed + int(fold))
        for _ in range(90):
            model.train()
            for batch in np.array_split(rng.permutation(len(train)), max(1, len(train) // 64)):
                logits = model(torch.tensor(train[batch], dtype=torch.float32))
                loss = hazard_loss(logits, torch.tensor(event[batch]), torch.tensor(day[batch]))
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        probability[valid_mask], onset[valid_mask] = predict(model, valid)
        positive = train_y.injured_in_risk_window.eq(1)
        sport_medians = train_y.loc[positive].assign(sport=metadata.loc[train_ids].loc[positive, "sport"]).groupby("sport").recovery_duration.median()
        recovery[valid_mask] = metadata.loc[valid_ids, "sport"].map(sport_medians).fillna(train_y.loc[positive, "recovery_duration"].median())
        onset_baseline[valid_mask] = train_y.loc[positive, "onset_day_offset"].mean(); recovery_baseline[valid_mask] = train_y.loc[positive, "recovery_duration"].mean()

    y_class = target.injured_in_risk_window.to_numpy(); threshold, _ = _best_threshold(y_class, probability)
    onset = np.rint(np.clip(onset, 1, 30)); recovery = np.rint(np.clip(recovery, 5, 20))
    metrics = score_predictions(y_class, probability, threshold, target.onset_day_offset.to_numpy(), target.recovery_duration.to_numpy(), onset, recovery, onset_baseline, recovery_baseline)
    metrics["parameters"] = sum(p.numel() for p in model.parameters()); metrics["research_score"] = metrics["f1"] + 0.25 * (metrics["onset_skill"] + metrics["recovery_skill"])
    oof = pd.DataFrame({"athlete_id": ids, "fold": folds, "y_injury": y_class, "y_onset": target.onset_day_offset.to_numpy(), "y_recovery": target.recovery_duration.to_numpy(), "p_injury": probability, "pred_injury": (probability >= threshold).astype(int), "pred_onset": onset, "pred_recovery": recovery})
    save_run(Path("runs"), Experiment("20_tcn_hazard", "sequence", "tcn_hazard", "posterior_median", "sport_median"), metrics, oof)
    print(json.dumps({"channels": channels, **metrics}, indent=2))


if __name__ == "__main__":
    main()
