# BreastCancer-Imbalance-Benchmark_35

Fresh, reproducible reconstruction of a breast-cancer class-imbalance benchmark using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## Project objective

Re-run the study from zero using a leakage-safe evaluation design, produce auditable machine-readable evidence, and write a new manuscript only from the newly generated results.

## Historical reference

The public repository `rzaroz/BreastCancer` is treated as a historical/reference implementation only. Its code and reported outputs are not accepted as evidence for the new manuscript unless independently reproduced by this repository.

## Frozen experimental protocol

Experimental Protocol v1.0 is frozen in `docs/EXPERIMENTAL_PROTOCOL_v1.0.md` and was merged at commit `00765102707ca0cce5b169af2b6436a91c643a87` before any fresh result-bearing experiment.

The primary benchmark is prespecified as six imbalance conditions (baseline, ADASYN, Borderline-SMOTE, SMOTE, SMOTE-Tomek, CTGAN) × four ensemble classifiers (Random Forest, AdaBoost, XGBoost, LightGBM) under shared 5×10 repeated stratified outer cross-validation and three deterministic stochastic replicates.

## Non-negotiable validity rules

1. Split/fold assignment occurs before any resampling or synthetic-data generation.
2. Resampling is fitted only on the training partition of each fold.
3. Test partitions remain untouched.
4. CTGAN is trained independently inside the corresponding training partition; no synthetic sample may be generated using held-out observations.
5. All preprocessing, resampling, model fitting, and any future tuning must be fold-local.
6. Raw per-fold/per-seed evidence must be preserved before aggregate tables or figures are produced.
7. Manuscript numbers, tables, and figures must be generated from auditable experiment outputs rather than manually copied from historical reports.
8. No scientific claim is accepted merely because it appears in the previous manuscript or repository.
9. WDBC benchmark evidence must not be presented as prospective clinical validation.

## P1 reproducibility foundation

Issue #3 and branch `p1/reproducibility-foundation` implement infrastructure only:

- exact dataset SHA-256 registration and schema validation;
- immutable shared 5×10 outer-fold artifact generation and hashing;
- deterministic 3,600-record seed registry;
- atomic evidence writers and non-result-bearing provenance manifests;
- typed leakage guards separating train and test partitions;
- unit/CI tests for hashes, schema gates, split invariants, seeds, evidence writes, and leakage protection.

No RF/AdaBoost/XGBoost/LightGBM scientific evaluation and no ADASYN/Borderline-SMOTE/SMOTE/SMOTE-Tomek/CTGAN execution are authorized in P1.

See `docs/P1_REPRODUCIBILITY_FOUNDATION.md`.

## Storage policy

- GitHub: source code, configuration, tests, machine-readable experiment outputs needed for reproducibility, and project issues/PRs.
- Google Drive project folder: manuscript-relevant evidence, publication tables, final figures, statistical reports, manuscript drafts, supplementary material, and submission packages.

## Current status

Protocol v1.0 is frozen. P1 foundation implementation is under review. Fresh result-bearing experiments: **0**.
