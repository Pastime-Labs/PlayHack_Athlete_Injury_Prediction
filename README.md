# PlayHack Injury Forecasting

Inference package for predicting injury occurrence, onset day, and recovery
duration from a 30-day athlete observation window.

## Included

- `predict.py`: command-line prediction entry point
- `src/`: feature construction, validation, and model inference code
- `artifacts/final_pipeline.joblib`: frozen trained model
- `artifacts/final_pipeline.manifest.json`: model provenance and checksum
- `submission.csv`: generated prediction output

The datasets and research experiments are intentionally not versioned.

## Setup

Python 3.12 on Windows:

```powershell
py -3.12 -m pip install -r requirements.txt
```

## Generate Predictions

Provide a directory containing the organizer CSV files:

```powershell
py -3.12 predict.py --data <data-directory> --policy f1 --output submission.csv
```

Use `--policy recall99` when the recall-focused operating point is required.
