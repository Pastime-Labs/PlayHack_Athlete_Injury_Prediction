from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def rebuild_leaderboard(runs: str | Path = "runs") -> pd.DataFrame:
    root = Path(runs)
    records = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config_path = metrics_path.with_name("config.json")
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        internal = metrics.pop("research_score", None)
        records.append({
            "experiment": metrics_path.parent.name,
            **config,
            **metrics,
            "internal_heuristic_unofficial": internal,
            "has_oof": metrics_path.with_name("oof.csv").exists(),
        })
    if not records:
        raise FileNotFoundError(f"No completed runs found under {root}")
    leaderboard = pd.DataFrame(records).sort_values(
        ["f1", "roc_auc"], ascending=False, na_position="last"
    )
    root.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(root / "leaderboard.csv", index=False)
    json_records = json.loads(leaderboard.to_json(orient="records"))
    (root / "leaderboard.json").write_text(json.dumps(json_records, indent=2), encoding="utf-8")
    return leaderboard
