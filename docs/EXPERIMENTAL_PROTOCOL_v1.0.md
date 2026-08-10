# Experimental Protocol v1.0 — BreastCancer-Imbalance-Benchmark_35

**Status:** FROZEN FOR PRIMARY EXPERIMENTS  
**Protocol version:** 1.0  
**Freeze date:** 2026-08-10  
**Repository:** `ArashSalehpourac/BreastCancer-Imbalance-Benchmark_35`  
**Historical repository:** `rzaroz/BreastCancer` (reference/audit only; historical numerical results are not admissible as evidence)

## 1. Scientific objective

The primary objective is to determine whether the effect of class-imbalance treatment on breast-cancer diagnostic classification depends on the downstream ensemble classifier, with particular emphasis on malignant-case detection and false-negative control.

The experiment compares six imbalance conditions:

1. No resampling (baseline)
2. ADASYN
3. Borderline-SMOTE
4. SMOTE
5. SMOTE-Tomek
6. CTGAN-based conditional minority augmentation

against four classifiers:

1. Random Forest (RF)
2. AdaBoost
3. XGBoost
4. LightGBM

This yields 24 prespecified method × classifier conditions. No additional method may enter the primary benchmark after the first result-bearing run without a protocol amendment.

## 2. Dataset and feature policy

The canonical dataset is the Breast Cancer Wisconsin (Diagnostic) dataset (WDBC). The exact raw input file used for the experiment must be archived before any run and identified by SHA-256.

### 2.1 Permitted columns

- `diagnosis` is the target.
- `id` is retained only as a traceability key and MUST NOT be used as a predictor.
- Any empty export/index column such as `Unnamed: 32` is removed.
- All legitimate diagnostic predictor variables in the canonical WDBC table are retained.

### 2.2 Dataset gates

Before execution, the loader must fail closed if any of the following occurs:

- the dataset hash differs from the registered manifest without an approved amendment;
- required columns are absent;
- duplicate column names exist;
- unexpected missing or infinite predictor values are found;
- target values are not exactly the expected benign/malignant labels after canonical normalization;
- the stable row identifier is not unique.

No manual correction of the raw dataset is permitted inside an experimental run.

## 3. Cross-validation design

### 3.1 Outer evaluation

Primary evaluation uses **5 × 10 repeated stratified cross-validation**:

- `n_splits = 10`
- `n_repeats = 5`
- stratification variable = diagnosis
- one master random seed controls fold generation
- the resulting 50 outer test folds are generated exactly once and stored as an immutable split artifact

The identical outer folds MUST be reused for every imbalance method and every classifier.

### 3.2 Master seed and deterministic seed derivation

`MASTER_SEED = 20260810`

All stochastic components must derive their seed deterministically from:

`SeedSequence([MASTER_SEED, repeat_index, fold_index, method_code, model_code, replicate_index])`

The experiment uses **3 stochastic replicates per outer-fold condition** for model fitting. For CTGAN, the same replicate index also controls the CTGAN fit/sample stochasticity for that fold.

No failed or inconvenient seed may be silently replaced, removed, or rerun under a different seed. Failures are evidence and must be logged.

### 3.3 Test-set isolation

The outer test fold is immutable and must never be used for:

- resampler fitting;
- synthetic-data generation;
- scaler fitting;
- model fitting;
- hyperparameter selection;
- threshold selection;
- synthetic-sample filtering;
- early stopping based on test performance;
- deciding whether to rerun an experiment.

## 4. Preprocessing order

For every outer fold and every stochastic replicate, the sequence is fixed:

1. Select outer training and outer test rows from the frozen split artifact.
2. Separate traceability ID from predictors.
3. Fit any required preprocessing object on the outer training fold only.
4. Transform the outer training fold.
5. Apply the already-fitted transformation to the outer test fold.
6. Apply the assigned imbalance treatment to the transformed **training fold only**.
7. Fit the classifier on the resulting training data.
8. Predict malignant-class probabilities on the untouched transformed outer test fold.
9. Compute and store predictions and fold metrics.

Because ADASYN/SMOTE-family methods are distance-based, continuous predictors are standardized using a `StandardScaler` fit on the current training fold only. The same scaled representation is used across all six conditions to avoid giving any condition privileged feature scaling. Tree models therefore receive the same transformed predictors for comparability.

Unexpected missing values cause a run failure; no imputation is introduced unless a later protocol amendment explicitly defines it.

## 5. Resampling specifications

All non-baseline resampling occurs inside the current outer training fold only. The intended minority class is malignant.

### 5.1 Baseline

- No resampling.
- No class weighting.
- Training data remain at their naturally occurring class distribution.

### 5.2 ADASYN

Prespecified semantics:

- target minority/majority ratio: 1.0 where feasible;
- nearest-neighbor parameter: 5;
- deterministic random state derived from the protocol seed.

Actual class counts before and after resampling must be recorded.

### 5.3 Borderline-SMOTE

Prespecified semantics:

- target minority/majority ratio: 1.0;
- `k_neighbors = 5`;
- `m_neighbors = 10`;
- borderline variant: Borderline-1;
- deterministic random state derived from the protocol seed.

Actual class counts before and after resampling must be recorded.

### 5.4 SMOTE

Prespecified semantics:

- target minority/majority ratio: 1.0;
- `k_neighbors = 5`;
- deterministic random state derived from the protocol seed.

Actual class counts before and after resampling must be recorded.

### 5.5 SMOTE-Tomek

Prespecified semantics:

1. SMOTE first targets a minority/majority ratio of 1.0 using `k_neighbors = 5`.
2. Tomek-link cleaning is then applied with an explicitly configured policy; implementation must not rely on an undocumented library default.
3. The final post-cleaning class counts are accepted as produced by the method and are recorded rather than manually forced back to exact balance.

The implementation review must lock the exact library argument corresponding to this semantic policy before the first run; changing that argument after a result-bearing run requires a protocol amendment.

## 6. CTGAN protocol

The historical CTGAN implementation is rejected. CTGAN in this benchmark is a fold-local conditional generator.

### 6.1 Training scope

For each outer fold and CTGAN stochastic replicate:

- instantiate a new CTGAN model;
- fit it **only on the current outer training fold**;
- never reuse a CTGAN model across folds, repeats, or replicates;
- include `diagnosis` in the CTGAN training table as an explicitly declared discrete/categorical column;
- continuous predictors remain continuous.

### 6.2 Conditioning

Synthetic observations MUST be generated by an explicit condition equivalent to:

`diagnosis = malignant`

The generated label must come from valid conditional generation. It is forbidden to generate unconditional rows and then overwrite their diagnosis values.

### 6.3 Amount of synthetic data

Let:

- `N_B` = benign rows in the current outer training fold;
- `N_M` = malignant rows in the current outer training fold.

Then:

`N_synthetic = max(0, N_B - N_M)`

Exactly this many conditioned malignant rows are requested so that the CTGAN training set targets a 1:1 class ratio.

### 6.4 CTGAN training budget

Primary protocol configuration:

- epochs: 400
- batch size: 50
- PAC: 10
- 3 stochastic CTGAN replicates per outer fold, coupled to the 3 classifier replicate seeds

The software implementation must seed Python, NumPy, and the generator backend deterministically wherever the selected CTGAN package exposes stochastic state.

### 6.5 CTGAN validity gates

A CTGAN replicate fails closed if:

- the requested malignant conditional sample count is not returned;
- any generated row has the wrong target condition;
- required columns are missing or reordered unexpectedly;
- NaN/Inf values occur;
- schema/type conversion fails;
- synthetic data cannot be transformed into the exact classifier feature schema.

Quality diagnostics are recorded but cannot use outer-test performance to accept/reject synthetic rows. Test data must never affect CTGAN filtering.

Audit-only CTGAN diagnostics include:

- requested vs returned synthetic count;
- exact duplicate rate against the real training fold;
- feature range exceedance rate relative to the training fold;
- per-feature distributional discrepancy summaries;
- nearest-neighbor distance summaries to real training observations.

These diagnostics are descriptive and do not permit post-hoc seed deletion.

## 7. Classifier policy

The primary experiment is intended to isolate imbalance-treatment effects rather than combine resampling with adaptive hyperparameter search.

Therefore:

- no hyperparameter tuning using outer test data is permitted;
- no class weighting is used in the four primary classifiers;
- all stochastic classifier states are explicitly seeded;
- a single fixed parameterization per classifier is used across all six imbalance conditions;
- package versions and the full realized parameter dictionary returned by each fitted estimator must be written to the run manifest.

The exact classifier parameter dictionaries will be committed as a configuration artifact and reviewed before the first result-bearing run. Once the first result-bearing run begins, changing a classifier parameter is a protocol change and requires a new protocol version.

A later tuned-model sensitivity analysis, if desired, is outside the primary benchmark and must be labeled secondary/exploratory unless separately preregistered with nested training-only tuning.

## 8. Prediction policy and threshold

For every outer test observation, store the predicted probability of the malignant class.

Primary hard classification threshold:

`p(malignant) >= 0.50`

The 0.50 threshold is fixed before results and may not be changed after viewing comparative performance.

Threshold-optimized clinical operating points may be added only as a separately labeled secondary analysis using thresholds chosen exclusively from training data.

## 9. Outcomes

### 9.1 Primary outcome

**Malignant-class recall (sensitivity)** is the prespecified primary endpoint because false negatives are the principal diagnostic error of interest.

### 9.2 Key secondary outcomes

- malignant-class precision / PPV
- malignant-class PR-AUC
- macro F1
- balanced accuracy
- Matthews correlation coefficient (MCC)
- specificity
- ROC-AUC

### 9.3 Additional descriptive outcomes

- accuracy
- weighted F1
- benign-class precision/recall/F1
- false-negative count and false-negative rate
- false-positive count and false-positive rate
- Brier score
- log loss
- confusion-matrix counts
- wall-clock fit and prediction time
- peak-process memory where reliably measurable

Accuracy alone is never sufficient to declare a method superior.

## 10. Cross-fitted subject-level predictions

Each subject is held out once per repeat. For each of the 24 method × classifier conditions, the 3 stochastic replicate probabilities from each repeat are retained individually.

For manuscript-level subject-wise analyses, an out-of-fold ensemble probability is computed by averaging all valid prespecified out-of-fold replicate probabilities for that subject and condition. Because every planned replicate is mandatory, missing replicates cause the condition/run to fail the completeness gate rather than being silently ignored.

This subject-level cross-fitted prediction table is the authoritative source for final confusion matrices and subject-level inferential comparisons.

## 11. Statistical analysis plan

### 11.1 General principles

- All comparisons are paired because identical subjects/folds are used across conditions.
- Effect sizes and confidence intervals are reported alongside p-values.
- Multiplicity is controlled with Holm correction within each declared test family.
- `alpha = 0.05`, two-sided.
- No significance claim may be based on selecting the most favorable metric after results are known.

### 11.2 Primary planned comparisons

Within each classifier, compare each of the five imbalance treatments against the no-resampling baseline on malignant detection.

This produces 20 planned primary contrasts:

`5 imbalance treatments × 4 classifiers`.

For subject-level malignant detection at threshold 0.50:

- restrict to truly malignant subjects;
- compare paired detected/not-detected outcomes using an exact McNemar test where computationally applicable;
- apply Holm correction across the 20 primary contrasts.

Report absolute sensitivity difference with a paired bootstrap 95% confidence interval.

### 11.3 Interaction analysis

The core scientific claim concerns imbalance-treatment × classifier interaction.

For each imbalance treatment and classifier, define the subject-level malignant-detection gain relative to that classifier's baseline. Interaction is quantified through **difference-in-differences** between classifier-specific gains.

For each of the five imbalance treatments, all six pairwise classifier gain contrasts are prespecified, yielding 30 interaction contrasts.

For truly malignant subjects:

- compute each subject's paired gain difference;
- use a paired permutation/sign-flip test for the mean difference-in-differences;
- use 100,000 Monte Carlo permutations when an exact enumeration is impractical;
- apply Holm correction across the 30 interaction contrasts;
- report the difference-in-differences and its paired bootstrap 95% CI.

A statistically significant interaction contrast means that the change in malignant detection produced by a given imbalance treatment differs between two classifiers; it does not by itself establish universal superiority.

### 11.4 Secondary metric uncertainty

For PR-AUC, ROC-AUC, macro F1, balanced accuracy, MCC, specificity, Brier score, and accuracy:

- use a paired stratified subject bootstrap;
- 10,000 bootstrap resamples;
- preserve malignant/benign class composition during resampling;
- use the same bootstrap draw for both conditions in a pair;
- report 95% percentile confidence intervals for each metric and paired difference.

Secondary p-values, if reported, must be clearly separated from the primary test family and multiplicity-adjusted within their own declared family.

### 11.5 Descriptive fold-level summaries

For transparency, retain the full 50-fold × 3-replicate metric distribution and report means, standard deviations, medians, and interquartile ranges. These fold values are not to be treated as 150 independent patients for inferential claims.

## 12. Model-selection and interpretation rules

There is no prespecified requirement to identify one universal winner.

A condition may be described as preferable only if the statement is supported by the prespecified endpoint structure. In particular:

- a gain in accuracy cannot compensate silently for a clinically important loss in malignant recall;
- statistically indistinguishable methods must not be rank-ordered as definitively superior based on trivial decimal differences;
- effect size, uncertainty, and false-negative behavior take precedence over isolated maximum accuracy;
- any post-hoc exploratory ranking must be labeled exploratory.

## 13. Evidence schema

Every result-bearing run must have a unique immutable `run_id` and a manifest containing at minimum:

- protocol version
- Git commit SHA
- dataset SHA-256
- split artifact SHA-256
- master seed
- software/package versions
- operating system
- Python version
- hardware summary
- UTC start/end timestamps
- full classifier parameter dictionaries
- full resampler parameter dictionaries
- CTGAN parameter dictionary
- status of every validation gate

### 13.1 Required machine-readable artifacts

`evidence/manifests/run_manifest.json`

`evidence/splits/fold_assignments.csv`

`evidence/raw/predictions.csv` or lossless columnar equivalent, with at least:

- `run_id`
- `protocol_version`
- `git_commit`
- `dataset_sha256`
- `row_id`
- `repeat`
- `fold`
- `method`
- `classifier`
- `replicate`
- `seed`
- `y_true`
- `p_malignant`
- `threshold`
- `y_pred`

`evidence/raw/fold_metrics.csv` with all prespecified metrics.

`evidence/raw/resampling_audit.csv` with pre/post class counts and method parameters.

`evidence/raw/ctgan_quality.csv` with CTGAN validity/quality diagnostics.

`evidence/stats/primary_tests.csv`

`evidence/stats/interaction_tests.csv`

`evidence/stats/bootstrap_intervals.csv`

`evidence/publication/table_source_*.csv`

`evidence/publication/figure_source_*.csv`

### 13.2 No manual transcription rule

Every number appearing in a manuscript table, figure, abstract, Results section, Discussion comparison, or supplementary table must be reproducibly traceable to a machine-readable evidence artifact.

Publication tables and figures must be generated programmatically from frozen evidence. Manually typing performance numbers into manuscript tables is prohibited.

## 14. Google Drive evidence rule

Any artifact that may be used in the paper must also be copied to the designated project Drive structure:

- `01_Experiment_Evidence` — raw manifests, frozen splits, predictions, fold metrics, resampling and CTGAN audits
- `02_Statistical_Analysis` — statistical test outputs, bootstrap outputs, statistical logs
- `03_Publication_Tables_Figures` — final table-source files and publication-ready figures
- `04_Manuscript_Drafts` — manuscript versions generated from accepted evidence
- `05_Submission_Package` — final manuscript, supplementary files, reporting checklist, code/data statement, cover materials

A Drive artifact used in the paper must retain provenance to the corresponding Git commit and run ID.

## 15. Run acceptance gates

A result-bearing run is scientifically admissible only if all of the following pass:

1. Dataset hash gate
2. Schema gate
3. Split/fold hash gate
4. Train/test isolation gate
5. Fold-local preprocessing gate
6. Fold-local resampling gate
7. CTGAN conditioning gate for CTGAN runs
8. Seed completeness gate
9. Prediction completeness gate
10. Metric recomputation consistency gate
11. Manifest completeness gate
12. Evidence-file checksum gate

If any gate fails, the run is marked **INVALID** and no number from it may enter the manuscript.

## 16. Protocol-change policy

This document is frozen before experimental implementation.

After the freeze commit:

- bug fixes that merely implement the stated protocol are allowed if documented and tested;
- any scientific design change requires a GitHub issue stating the rationale and impact;
- any design change before the first result-bearing run increments at least the minor protocol version;
- any design change after results have been viewed that could change results must create a new protocol version and a new run family;
- old evidence is never overwritten;
- historical results from `rzaroz/BreastCancer` remain audit/reference material only.

## 17. Pre-implementation checkpoint

At protocol freeze:

- fresh experimental code written: **NO**
- fresh experimental runs executed: **NO**
- historical numerical results accepted: **NO**
- experimental protocol: **FROZEN v1.0**

The next permitted phase is implementation of data loading, split freezing, leakage gates, evidence writers, resampling wrappers, model wrappers, CTGAN conditioning, metrics, and tests strictly against this protocol. No scientific result may be generated until implementation tests pass.
