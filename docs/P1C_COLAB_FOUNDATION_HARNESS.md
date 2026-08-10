# P1C — Google Colab Foundation Verification Harness

**Status:** implementation candidate  
**Issue:** #5  
**Branch:** `p1c/colab-foundation-harness`  
**Authorized base:** `2e6c405c6dce747fc86e3d72252f7a78831f1771`  
**Scientific experiments authorized:** none

## Purpose

P1C verifies that a Google Colab runtime is connected to the exact frozen P1 foundation before any future result-bearing work is considered. The harness mounts Google Drive, reads a user-supplied WDBC CSV from Drive, and fails closed unless the dataset, shared outer folds, and deterministic seed registry exactly match the P1 foundation lock.

## Frozen identity checked by the harness

- Dataset SHA-256: `27f219231dbb30eecbfc1361407ed641ea01be43316e2c707a1baf82c9795e23`
- Dataset bytes: `124635`
- Rows: `569`
- Predictors: `30`
- Class counts: `B=357`, `M=212`
- Outer folds: `50` under the frozen 5×10 repeated-stratified design
- Split SHA-256: `00114e7735fc4eb012ecf248010b18b441cce4a74e0d1540115bbaff543d764a`
- Master seed: `20260810`
- Seed records: `3600`
- Seed-registry SHA-256: `985f5614275ef880213bf775cded0b8e3867fb011f43df817b48d94bb5af73e2`

The authoritative machine-readable values remain in `data/registry/FOUNDATION_LOCK_v1.json`.

## Colab notebook

Use:

`colab/P1C_COLAB_FOUNDATION_GATE.ipynb`

The notebook is intentionally limited to:

1. mounting Google Drive;
2. cloning the private repository using the `GITHUB_TOKEN` Colab Secret without printing the token;
3. confirming the checked-out repository descends from the authorized P1 base commit;
4. installing the foundation package;
5. verifying the exact Drive dataset bytes and schema;
6. regenerating the frozen outer-fold artifact in memory and comparing its hash;
7. regenerating the deterministic seed registry in memory and comparing its hash;
8. writing non-result-bearing verification evidence to Drive;
9. emitting `COLAB_FOUNDATION_GATE=PASS` only after all checks succeed.

## User setup in Colab

Before running the notebook:

1. Upload the exact WDBC CSV to Google Drive.
2. Add a GitHub fine-grained read-only token to Colab **Secrets** using the name `GITHUB_TOKEN`. Never paste the token into a code cell.
3. Open the notebook in Colab.
4. Edit the `DATASET_PATH` line so it points to the uploaded CSV.
5. Optionally edit `OUTPUT_DIR`; it should remain inside the project Drive evidence area.
6. Run the notebook from top to bottom.

A recommended path is:

`/content/drive/MyDrive/BreastCancer-Imbalance-Benchmark_35/01_Experiment_Evidence/00_Dataset/wdbc.csv`

The actual mounted Drive path may differ; the SHA-256 gate, not the filename or folder name, determines dataset identity.

## Evidence emitted on PASS

The output directory contains:

- `colab_foundation_gate.json`
- `colab_foundation_gate.json.sha256`
- `colab_foundation_manifest.json`
- `colab_foundation_manifest.json.sha256`

All emitted evidence is marked `result_bearing=false`.

## Fail-closed behavior

The harness stops without a PASS if any of the following differs from the frozen foundation:

- dataset byte hash or byte count;
- schema, row count, feature count, or B/M class counts;
- outer-fold count or split hash;
- master seed, seed count, or seed-registry hash;
- repository ancestry relative to the authorized P1 base.

## Scope prohibition

P1C must contain no scientific classifier training/evaluation, no imbalance-method execution, no synthetic-data generation, no performance metric calculation, and no statistical comparison. A successful P1C run proves only operational reproducibility of the frozen foundation.

## Acceptance gate

P1C is ready for merge only when:

- the PR diff remains limited to the Colab foundation harness and its verification infrastructure;
- all unit tests pass in GitHub Actions;
- the notebook static scope guard passes;
- no result-bearing scientific code or output is introduced;
- merge is explicitly authorized after validation.

Merging P1C will not authorize the next scientific phase.
