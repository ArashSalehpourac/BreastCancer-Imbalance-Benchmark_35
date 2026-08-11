# P3 — First Result-Bearing Execution Infrastructure

**Status:** infrastructure candidate only  
**Issue:** #9  
**Branch:** `p3/result-bearing-execution-infrastructure`  
**Authorized base:** `8c6ff728375de033b32fac8b32240a22e98401e1`

P3 prepares the fail-closed preflight, dry-run planning, evidence-schema, environment capture, and checkpoint primitives needed before a future result-bearing run. P3 itself does **not** execute a classifier, resampler, CTGAN model, prediction, scientific metric, or statistical test.

## Frozen identities enforced by preflight

The preflight requires all of the following to match before a future condition can be considered executable:

- Protocol v1.0;
- exact canonical WDBC SHA-256;
- exact P1 foundation-lock SHA-256;
- exact 50-fold split SHA-256 regenerated from the canonical dataset;
- exact 3,600-record seed-registry SHA-256 regenerated from the frozen seed formula;
- exact P2 experiment-config SHA-256;
- exact P3 result-evidence schema SHA-256;
- the 24 method×classifier primary conditions and 3 replicates;
- exact direct scientific package versions frozen in P2;
- an explicitly supplied exact Git commit equal to repository `HEAD`;
- a realized and SHA-256-receipted `pip freeze`.

Any mismatch is a hard failure.

## Dry-run execution plan

`bcimbalance.execution_harness.build_execution_plan` deterministically expands the frozen design into 3,600 planned execution tasks:

`5 repeats × 10 folds × 6 imbalance methods × 4 classifiers × 3 replicates = 3,600`.

Each task has a unique ID, frozen seed, primary-condition ID, and unique future evidence path. Planning is metadata-only and is marked `result_bearing=false`.

## Result-evidence schema

`config/RESULT_EVIDENCE_SCHEMA_v1.json` machine-locks the minimum artifact paths and prediction columns specified in Protocol v1.0 Section 13. The schema describes future result-bearing evidence but is itself a P3 infrastructure artifact marked non-result-bearing. Its frozen canonical SHA-256 is:

`7819dd71c49d3c5c686ca76079a92f0b5399596d0de71d2964eca8ece8af4686`

## Checkpoint/resume semantics

The infrastructure defines fail-closed checkpoint state:

- no marker → `PENDING`;
- one `.inprogress.json` marker → `INCOMPLETE`;
- one `.complete.json` marker → `COMPLETE`;
- simultaneous in-progress and complete markers → hard failure.

Checkpoint metadata is written atomically. P3 tests exercise these transitions using temporary metadata only; they do not execute any scientific condition.

## Colab-compatible preflight

After the exact P3 implementation revision is independently validated and separately authorized, Colab can run the preflight entry point against the canonical Drive dataset. The exact authorized Git commit must be supplied explicitly; the script does not trust a moving branch name.

Example shape only:

```bash
python scripts/p3_preflight.py \
  --dataset /content/drive/MyDrive/35/01_Experiment_Evidence/00_Dataset/wdbc_canonical_lf.csv \
  --repo-root /content/BreastCancer-Imbalance-Benchmark_35 \
  --expected-git-commit <SEPARATELY_AUTHORIZED_EXACT_SHA> \
  --output-dir /content/drive/MyDrive/35/01_Experiment_Evidence/P3_Preflight
```

The expected terminal gates are:

```text
P3_PREFLIGHT_GATE=PASS
P3_DRY_RUN_GATE=PASS
RESULT_BEARING=false
SCIENTIFIC_EXECUTION=0
```

A PASS authorizes nothing by itself. Starting the first result-bearing condition requires a separate controlled authorization after the P3 harness revision, CI, preflight evidence, and Colab environment have been reviewed.

## Explicitly absent from P3

No P3 source file contains an implementation call to `fit`, `fit_resample`, CTGAN sampling, `predict`, `predict_proba`, metric calculation, inferential statistics, publication-table generation, or manuscript-result generation. The P3 CI includes a static scope test for the primary scientific execution call names.
