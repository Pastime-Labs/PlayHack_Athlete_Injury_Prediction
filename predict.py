from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.data import load_bundle
from src.final_model import predict_bundle, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a validated injury prediction submission.")
    parser.add_argument("--data", required=True, help="Directory containing observation-window CSV files.")
    parser.add_argument("--artifact", default="artifacts/final_pipeline.joblib")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--policy", choices=("f1", "recall99"), default="f1")
    args = parser.parse_args()

    bundle = load_bundle(args.data)
    predictions = predict_bundle(joblib.load(args.artifact), bundle, policy=args.policy)
    submission = predictions.drop(columns="injury_probability")
    validate_submission(submission, bundle["metadata"].athlete_id.to_numpy())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"wrote {len(submission)} validated rows to {output}")


if __name__ == "__main__":
    main()
