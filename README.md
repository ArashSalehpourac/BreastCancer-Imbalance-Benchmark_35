# BreastCancer-Imbalance-Benchmark_35

Fresh, reproducible reconstruction of a breast-cancer class-imbalance benchmark using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## Project objective

Re-run the study from zero using a leakage-safe evaluation design, produce auditable machine-readable evidence, and write a new manuscript only from the newly generated results.

## Historical reference

The public repository `rzaroz/BreastCancer` is treated as a historical/reference implementation only. Its code and reported outputs are not accepted as evidence for the new manuscript unless independently reproduced by this repository.

## Core experimental factors

Oversampling / data-generation conditions:

- Original / no resampling
- ADASYN
- Borderline-SMOTE
- SMOTE
- SMOTE-Tomek
- CTGAN

Ensemble classifiers:

- Random Forest
- AdaBoost
- XGBoost
- LightGBM

## Non-negotiable validity rules

1. Split/fold assignment occurs before any resampling or synthetic-data generation.
2. Resampling is fitted only on the training partition of each fold.
3. Test/validation partitions remain untouched.
4. CTGAN is trained independently inside the corresponding training partition; no synthetic sample may be generated using held-out observations.
5. Repeated stratified cross-validation and explicit random seeds will be used instead of relying on one 80:20 split.
6. All preprocessing, tuning, resampling, and model fitting must be fold-local to prevent leakage.
7. Raw per-fold/per-seed outputs must be preserved before aggregate tables or figures are produced.
8. Manuscript numbers, tables, and figures must be generated from auditable experiment outputs rather than manually copied from historical reports.
9. No scientific claim is accepted merely because it appears in the previous manuscript or repository.
10. The manuscript will clearly distinguish benchmark evidence from clinical validation; WDBC results will not be presented as prospective clinical validation.

## Planned evidence

Primary metrics will include accuracy, balanced accuracy, macro-F1, malignant-class recall/sensitivity, specificity, MCC, ROC-AUC, PR-AUC, false-negative rate, and calibration measures when probabilities are available. Statistical comparisons and confidence intervals will be derived from repeated held-out folds/seeds.

## Storage policy

- GitHub: source code, configuration, tests, machine-readable experiment outputs needed for reproducibility, and project issues/PRs.
- Google Drive project folder: manuscript-relevant evidence, publication tables, final figures, statistical reports, manuscript drafts, supplementary material, and submission packages.

## Current status

Repository initialized. No experiment has yet been authorized or accepted as scientific evidence.
