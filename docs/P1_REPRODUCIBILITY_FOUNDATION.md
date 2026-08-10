# P1 — Reproducibility Foundation

**Status:** implementation candidate on `p1/reproducibility-foundation`  
**Governing protocol:** `docs/EXPERIMENTAL_PROTOCOL_v1.0.md`  
**Base commit:** `00765102707ca0cce5b169af2b6436a91c643a87`  
**Issue:** #3

P1 implements infrastructure only. It does not train, fit, sample, evaluate, or compare any scientific model or imbalance-treatment method.

## Components

### Dataset registration and gates

`bcimbalance.dataset` hashes exact source bytes with SHA-256, validates the canonical WDBC CSV schema, removes only an empty `Unnamed: 32` export column, excludes `id` from predictors, normalizes `diagnosis` to B/M, and fails closed on unexpected columns, invalid labels, duplicate IDs, missing/non-numeric values, or NaN/Inf predictors.

The raw CSV is intentionally gitignored. The dataset registry JSON is designed to carry the exact byte hash, row count, feature count, class counts, and canonical column list.

### Immutable outer folds

`bcimbalance.folds` creates the protocol-fixed 5×10 repeated stratified outer-CV artifact using `MASTER_SEED=20260810`. The artifact stores row IDs rather than dataframe positions and is hashed over canonical JSON. Verification requires:

- exactly 50 folds;
- complete 5×10 repeat/fold grid;
- no train/test overlap;
- train∪test equals all registered rows for every fold;
- each row appears exactly once in a test fold per repeat;
- artifact hash integrity.

Once generated from the registered dataset, the resulting split artifact becomes immutable evidence and must be reused by every later method/classifier condition.

### Deterministic seed registry

`bcimbalance.seeds` implements the frozen formula:

`SeedSequence([MASTER_SEED, repeat_index, fold_index, method_code, model_code, replicate_index])`

The complete frozen key space contains 3,600 condition seeds: 5 repeats × 10 folds × 6 methods × 4 models × 3 replicates. Generation fails if any duplicate key or seed collision is detected.

### Evidence writers

`bcimbalance.evidence` provides atomic JSON, JSONL, and CSV writes, SHA-256 receipts, checksum sidecars, environment snapshots, and P1 manifests explicitly marked `result_bearing=false`.

### Leakage guards

`bcimbalance.guards` creates separate `TrainingPartition` and `TestingPartition` objects, rejects overlap or incomplete row coverage, and provides a preprocessing helper that can fit only on the training partition before transforming both partitions.

## CLI sequence after P1 approval

The following commands are infrastructure operations, not scientific experiments:

```bash
python -m pip install -e .
python scripts/register_dataset.py --source /path/to/WDBC.csv
python scripts/generate_folds.py
python scripts/generate_seed_registry.py
python -m unittest discover -s tests -v
```

Dataset registration/fold generation should not be performed until the exact candidate WDBC source file is selected for the project. No script in P1 invokes RF, AdaBoost, XGBoost, LightGBM, ADASYN, Borderline-SMOTE, SMOTE, SMOTE-Tomek, or CTGAN.

## P1 acceptance gate

P1 is ready for merge only when the PR diff remains within foundation scope and GitHub Actions passes all foundation tests. Merge does **not** authorize any scientific experiment. The next phase must separately freeze the exact dataset hash, split hash, seed-registry hash, and classifier parameter dictionaries before result-bearing execution.
