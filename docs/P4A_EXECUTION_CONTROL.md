# P4A — First Result-Bearing Execution Control Layer

**Status:** control implementation candidate only  
**Issue:** #11  
**Branch:** `p4/first-result-bearing-execution-control`  
**Authorized base:** `57f721960f87ac189851024b477fbcdb771a7ed4`

P4A prepares the exact Colab control path that must pass before a later P4B scientific authorization. P4A is deliberately **non-result-bearing**. It does not fit classifiers, execute resamplers, fit/sample CTGAN, create predictions, calculate scientific metrics, run inferential statistics, or generate publication results.

## Machine-readable control policy

`config/P4A_EXECUTION_CONTROL_POLICY_v1.json` locks the Drive path policy, P0/P1/P2/P3 identities, 24-condition/50-fold/3-replicate design, 3,600-task count, threshold 0.50, malignant positive class, future run-root layout, and deterministic mapping from P3 logical task-record paths to P4 `raw/conditions/` paths.

Physical SHA-256 of the policy file:

`af3212b00c87afbd83032de5a86b1bb933f840a7e6b675e752278e55c2a80c0b`

Any policy mismatch fails closed.

## Colab control sequence

1. Start a fresh Colab runtime and mount Drive.
2. Use only `/content/drive/MyDrive/35/01_Experiment_Evidence/00_Dataset/wdbc_canonical_lf.csv`.
3. Authenticate to the private repository through the Colab `GITHUB_TOKEN` secret without printing the token.
4. Clone the repository and checkout a separately reviewed exact commit SHA.
5. Install the exact P2 environment through `requirements/p3-preflight.txt`, then install the project editable with `--no-deps`.
6. Run `scripts/p4a_control.py` with `--require-exact-colab-paths`.
7. P4A invokes the accepted P3 preflight/dry-run, captures `pip freeze`, verifies all frozen hashes and counts, writes control-only manifests and receipts, and requires all 3,600 scientific tasks to remain `PENDING`.
8. Store P4A validation evidence only under `/content/drive/MyDrive/35/01_Experiment_Evidence/P4A_Control_Validation/<P4A_CONTROL_SESSION>/`.
9. Do not create `/content/drive/MyDrive/35/01_Experiment_Evidence/P4_Result_Bearing_Runs/<RUN_ID>/` during P4A.

Expected terminal gates:

```text
P3_PREFLIGHT_GATE=PASS
P3_DRY_RUN_GATE=PASS
P4A_CONTROL_GATE=PASS
P4B_AUTHORIZED=false
RESULT_BEARING=false
SCIENTIFIC_EXECUTION=0
```

A PASS does not authorize P4B.

## Future P4B run-root policy — defined, not executed

A future separately authorized scientific run must use exactly one immutable root:

`/content/drive/MyDrive/35/01_Experiment_Evidence/P4_Result_Bearing_Runs/P4_RUN_<UTC>_<12-char-exact-git-sha>/`

Required direct children are:

- `preflight/`
- `manifests/`
- `splits/`
- `raw/`
- `checkpoints/`
- `logs/`
- `receipts/`

P3 logical task-record paths beginning with `evidence/conditions/` are preserved as plan metadata and deterministically mapped to `raw/conditions/` inside the future P4 run root. This mapping does not alter the frozen P3 plan.

## Checkpoint/resume gate

P4A audits the accepted P3 3,600-task plan against the checkpoint directory. During P4A every task must remain `PENDING`. Unknown marker filenames, unknown task IDs, simultaneous `.inprogress.json` and `.complete.json` markers, or any non-pending scientific task cause hard failure.

The future P4B procedure may use the accepted checkpoint semantics only after separate authorization; P4A itself does not write scientific `.inprogress` or `.complete` markers.

## Evidence produced by P4A

The P4A control session writes only non-result-bearing artifacts:

- `preflight/pip_freeze.txt` plus SHA-256 receipt
- `preflight/p3_preflight_report.json` plus receipt
- `preflight/p3_execution_plan.json` plus receipt
- `manifests/p4a_control_manifest.json` plus receipt
- `manifests/p4a_path_policy_report.json` plus receipt
- `receipts/p4a_receipt_index.json` plus receipt

Every P4A manifest contains `p4b_authorized=false`, `result_bearing=false`, and `scientific_execution=0`.

## Separate P4B authorization

No P4A merge, CI PASS, Colab PASS, or control manifest authorizes a scientific task. P4B requires a new explicit authorization at an exact reviewed and accepted revision after P4A merge and Colab validation.
