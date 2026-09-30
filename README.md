# PlayHack Injury Forecasting

Our entry for the PlayHack sports prediction challenge. Given 30 days of activity, sleep, heart-rate and training data, we predict whether an athlete will be injured during the following month, when that injury will occur, and how long recovery will take.

We built the models using data from 3,000 athletes and generated predictions for the organizers' 1,100 test athletes. Much of the feature engineering measures changes relative to each athlete's usual workload, sleep and heart rate.

The final model uses ExtraTrees for injury risk, a combination of histogram gradient boosting and a temporal convolutional network (TCN) for onset, and a recovery estimate based on sport.

## Results

These results come from five-fold out-of-fold (OOF) predictions on 2,400 development athletes. Each athlete's prediction came from a model trained on the other folds.

| Metric | Development result |
|---|---:|
| Injury F1 | 0.6768 |
| Injury ROC-AUC | 0.7842 |
| Injury PR-AUC (average precision) | 0.7760 |
| Onset MAE | 2.385 days |
| Recovery MAE | 2.871 days |

The F1 threshold was selected using these OOF predictions. The timing errors above are measured on true injuries, before the challenge's missed-injury penalties.

We also checked the classifier across three split seeds and a separate evaluation that held out entire teams. F1 with cross-fitted thresholds was **0.6747 +/- 0.0075** and ROC-AUC was **0.7836 +/- 0.0025**, reported as mean +/- standard deviation across the four evaluations. On the team holdout alone, F1 was 0.6702 and ROC-AUC was 0.7853.

The official test labels have not been released, so we cannot report test-set performance. The saved comparisons are in [runs/leaderboard.csv](runs/leaderboard.csv) and [artifacts/robustness/](artifacts/robustness/).

## Run the model

The project was developed with Python 3.12. From the repository root:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 predict.py --data Test_DataSet --policy recall99 --output reproduced_submission.csv
```

On systems without the Windows Python launcher, use `python` from your Python 3.12 environment in place of `py -3.12`.

The command loads `artifacts/final_pipeline.joblib` and writes one prediction row per athlete. It checks the feature schema, athlete IDs, missing values, integer types and output ranges before saving. The `recall99` option reproduces the policy used by the current `submission.csv`; see [Submission and thresholds](#submission-and-thresholds) for the tradeoff.

You will need the organizer data. The input directory must contain:

```text
Test_DataSet/
|-- athlete_metadata.csv
|-- dailyActivity_merged.csv
|-- sleepDay_merged.csv
|-- training_sessions.csv
|-- hourlyHeartrate_merged.csv
`-- weightLogInfo_merged.csv
```

Training uses the same files under `DataSet/`, together with `train_labels.csv`.

For inference only, [PlayHack_Model_Submission.zip](PlayHack_Model_Submission.zip) contains the trained model, required Python modules, a smaller dependency list and the current prediction CSV. Its [README](PlayHack_Model_Submission/README.md) has standalone instructions. Raw data and presentation files are excluded from that archive.

## The data

Each athlete contributes observations from **5 January to 3 February 2026**. The prediction window runs from **4 February to 5 March 2026**.

The training set contains 1,050 athletes with an injury in the prediction window and 1,950 without one. The official test set uses different athlete IDs (`3001..4100`), with no overlap with training.

Our EDA covered class balance, sport and gender composition, date coverage, duplicate athlete-day records, missing measurements and differences between the training and test inputs. The test data matched the model's feature and sequence schemas.

### Cleaning and preparation

We join the sources by athlete and date, keeping only observations before 4 February. This filter also applies to weight records before selecting each athlete's most recent measurement.

Daily activity provides the base table. We add sleep duration and efficiency, aggregate sessions into training hours and session-type counts, and summarize hourly heart rate by day. Heart-rate files are processed in chunks to keep memory use manageable. The loader checks that each athlete has 30 unique observation days and that IDs agree across the input tables.

Missing measurements needed different treatment depending on the source:

- An absent session row becomes zero recorded training sessions for that day.
- Missing physiological measurements stay missing until numeric imputation. We do not interpolate them or remove an athlete just because a measurement is absent.
- Sparse weight, BMI and body-fat records contribute their latest available values, along with a weight-availability indicator and measurement count.
- Numeric medians and categorical encodings are fitted on the training portion of each fold. Sequence normalization also uses training-fold statistics.

The EDA and preparation figures are available in [artifacts/presentation_visuals/](artifacts/presentation_visuals/).

## Features

The tabular model takes 405 columns per athlete before one-hot encoding. The TCN receives the same observation window as a sequence of 30 days and 10 channels.

| Feature group | What we calculate |
|---|---|
| Athlete context | Sport, age, gender, position, playing history, prior injuries and other metadata |
| Levels and variability | Means, medians, standard deviations, minima and maxima across daily signals |
| Recent changes | Full-window slopes, last-seven-day averages, and differences and ratios against the earlier 23 days |
| Training and recovery | Session duration and type counts, sleep efficiency, heart-rate summaries and available body measurements |
| Personal baseline | Deviations from each athlete's weekly activity, sleep and heart-rate pattern |
| Changes across signals | Ramp strength, estimated change point, agreement between signals and strain acceleration |

A high step count alone says little about whether an athlete's routine has changed. We therefore use the first two observation weeks to estimate that athlete's usual weekly pattern, then measure deviations from it.

For selected signals, we fit candidate ramps starting between observation days 14 and 30. The resulting features describe how strongly a signal changes, when the change begins, and whether other signals change around the same time. The combined strain features align rising activity and heart rate with declining sleep duration or efficiency.

The feature comparisons in the research runs helped us decide which representation to use. Later features for rest gaps, consecutive training days and lagged responses did not improve results consistently enough to include in the final model.

The implementation is in [src/features.py](src/features.py).

## Models

### Injury probability

ExtraTrees performed well on the structural features and held up across the repeated evaluations. The final classifier uses 180 trees, a minimum leaf size of 7, `max_features=0.65` and balanced class weights.

We also tried blending its probabilities with the TCN. The estimated AUC improvement was small, and its paired bootstrap interval included zero, so we kept ExtraTrees on its own for this target.

### Injury onset

The histogram gradient boosting regressor learns from the injured athletes' tabular features. The TCN uses the daily sequences to estimate a hazard for each of the 30 future days, from which we obtain a conditional median onset day.

The TCN has 32 hidden channels, residual blocks with dilations of 1, 2 and 4, and mean/max pooling before its hazard head. We trained it with AdamW, dropout, weight decay and gradient clipping; the final fit used 90 epochs.

We combine the two onset estimates as:

```text
onset_day = round(clip(0.65 * HGB_prediction + 0.35 * TCN_prediction, 1, 30))
```

Their errors were complementary: blending reduced rounded onset MAE from 2.4679 days for HGB alone to 2.3845 days. The paired bootstrap interval for the change was `[-0.1333, -0.0369]` days.

### Recovery duration

The recovery models we tested did not find enough additional signal to improve reliably on a sport-specific median. We use the median recovery duration among training injuries in the athlete's sport, with an overall median for an unseen sport. Predictions are rounded and clipped to the observed training range of 5 to 20 days.

All three components were refit on the full 3,000 labelled athletes after model selection. The saved pipeline includes the preprocessing, sequence statistics, feature order and thresholds needed for inference. Its source hashes, dependency versions and SHA-256 are recorded in [final_pipeline.manifest.json](artifacts/final_pipeline.manifest.json).

## Validation

We split the 3,000 labelled athletes into 2,400 for development and 600 for an initial holdout. The development set has five fixed folds, stratified by injury label. All days belonging to an athlete stay together.

The 600-athlete holdout was examined during the original research. We treated it as used after that point and excluded it from later tuning and model-selection decisions.

Models, imputers, encoders and timing baselines are fitted within each fold. The initial F1 result uses a threshold chosen on pooled OOF predictions. In the repeated evaluations, thresholds are cross-fitted: the threshold for one fold is chosen from the other folds' OOF predictions.

We repeated the evaluation with sport/onset stratification using seeds 2026, 2037 and 2048, then held out complete teams using `StratifiedGroupKFold`. We also inspected sport, gender and missingness subgroups and used paired athlete bootstraps when comparing candidates.

These checks support performance on other athletes from a similar setting. Generalization to a different sport or dataset still needs its own evaluation.

## Other experiments

The initial comparison covered 21 configurations: simple global and sport baselines, logistic regression, ExtraTrees, gradient boosting, random forest, CatBoost, a matched-event model, MiniROCKET and the hazard TCN. We also compared activity-only and recovery-focused feature sets.

After selecting the current model, we tested a few further ideas:

| Experiment | What we found |
|---|---|
| Nested hyperparameter search | Five outer folds and three inner folds did not find better ExtraTrees or HGB settings. |
| PMData pretraining | Pretraining the encoder on 446 external windows increased blended onset MAE to 2.4393 days. It showed no reliable benefit over a matched random-initialization control. |
| RunFitCheck transfer | An external injury-risk score added 0.00060 AUC and 0.00557 cross-fitted F1, below our acceptance thresholds. |
| ExtraTrees + TCN stacker | A regularized logistic stacker improved calibration, but its AUC gain was small and F1 fell. |
| Workload and cross-modal feature pack | Early F1 gains did not hold up on another athlete split and the team holdout. |

Those experiments left the final model selection unchanged. We did not pool external and organizer rows: source-classification AUC remained 0.9632 after normalization, showing substantial dataset differences. PMData's downloaded license is CC BY-NC 4.0; its research checkpoint is excluded from the submission.

The [full report](FINAL_REPORT.md), [external-data study](EXTERNAL_DATA_RESEARCH.md) and [challenger results](artifacts/challenger_confirmation.json) contain the experiment details.

## Submission and thresholds

The prediction CSV has four integer columns:

```csv
athlete_id,injured_in_risk_window,onset_day_offset,recovery_duration
```

Onset and recovery are supplied for every athlete, including those assigned an injury flag of zero.

Choosing the injury threshold matters because the challenge assigns a 30-day error to both timing predictions whenever a true injury is missed. We prepared two operating points:

| CLI policy | Threshold | Development F1 | Development recall | Athletes flagged in the test set |
|---|---:|---:|---:|---:|
| `f1` | 0.460 | 0.6768 | 0.5321 | 268 |
| `recall99` | 0.146 | 0.5332 | 0.9905 | 1,064 |

The lower threshold reduces missed injuries at the cost of many more false positives. Its name refers to recall on the development predictions.

The files are saved as `artifacts/final_submission_f1.csv` and `artifacts/final_submission_recall99.csv`. The root `submission.csv` currently matches the latter.

Both come from the same fitted model, and only the injury flag changes. The final upload should contain one policy CSV, selected once the organizers clarify the metric weights and sample-submission format. The validation code currently enforces onset in `1..30` and recovery in `5..20`, matching the training labels.

## Reproducing the work

Place the labelled data in `DataSet/` and install the root requirements. Run the core comparisons first; later scripts use the resulting fold assignments and OOF files.

```powershell
py -3.12 run_research.py
py -3.12 run_matched.py
py -3.12 run_catboost.py --features mechanistic
py -3.12 run_catboost.py --features structural
py -3.12 run_minirocket.py
py -3.12 run_tcn.py
py -3.12 rebuild_leaderboard.py
py -3.12 analyze_pareto.py
py -3.12 robust_validate.py
```

With those results in place, refit the selected model to a separate file:

```powershell
py -3.12 train_final.py --output artifacts/reproduced_pipeline.joblib
py -3.12 predict.py --data Test_DataSet --artifact artifacts/reproduced_pipeline.joblib --policy recall99 --output reproduced_submission.csv
```

The research scripts cache processed data using content fingerprints and save metrics and OOF predictions under `runs/` and `artifacts/`. Training records the data, code, dependency and model hashes alongside the artifact.

To run the tests:

```powershell
py -3.12 -m pytest -q --import-mode=importlib tests
```

The suite covers date filtering, split integrity, feature calculations, timing penalties, submission checks and the experimental adapters. The explicit test path and import mode also avoid name clashes with any copied tests in local staging directories.

## Repository layout

```text
src/                         Data loading, features, evaluation and final model
tests/                       Automated checks
runs/                        Experiment metrics, leaderboard and OOF predictions
artifacts/                   Trained model, manifests and analysis outputs
predict.py                   Generate predictions from a saved model
train_final.py               Fit the selected model on all labelled athletes
run_research.py              Run the initial tabular experiments
run_tcn.py                   TCN definition and experiment runner
robust_validate.py           Repeated-split and team-held evaluations
PlayHack_Model_Submission/   Standalone inference package
```
