# P4B Result-Bearing Execution Package

## Status

This package prepares the future scientific executor and its authorization gate. **Implementation, CI, PR review, and Colab validation are non-result-bearing.**

Authoritative implementation base:

`ffd7a29c8c05d17ac4e20dae9db75492d1ed1dff`

Control issues:

- Parent P4 control: GitHub Issue #11
- P4B package: GitHub Issue #13

Mandatory review state:

```text
P4B_AUTHORIZED=false
RESULT_BEARING=false
SCIENTIFIC_EXECUTION=0
```

No P4B package-validation action may fit a WDBC model, execute a WDBC resampler, fit/sample CTGAN, generate WDBC predictions, compute WDBC scientific metrics, or create a `P4_Result_Bearing_Runs/<RUN_ID>` directory.

## Frozen identities

The package preserves Protocol v1.0 and the already accepted P1/P2/P3/P4A identities. The accepted P4A environment receipt is:

`9bdf50ed4d756c753579eb0e4ced6031299ffd6dab1d885f03dfdf58e9978a4e`

The canonical 3,600-task execution-plan identity is:

`3696604d936e103426fd0527cf8a37b1ec619043fcb32121cb55fe119b94c137`

The P4B policy file itself is intentionally hashed from its final exact bytes during review; that physical SHA-256 must be recorded with the exact validated PR head before any later scientific authorization is constructed.

## Package architecture

`config/P4B_EXECUTION_POLICY_v1.json` freezes the P4B authorization semantics, run-root policy, task mapping, checkpoint state machine, accepted environment identity, and one-task first-run limit.

`config/P4B_AUTHORIZATION_v1.schema.json` describes the future machine-readable authorization record. A real authorization record is **not committed to the repository**.

`src/bcimbalance/p4b_authorization.py` validates the policy and authorization record before any scientific path is entered. It binds one exact Git commit, one exact task ID, `max_tasks=1`, one immutable run root, the policy hash, all frozen scientific hashes, and the accepted realized environment receipt.

`src/bcimbalance/scientific_task.py` contains the future one-task executor. Its public entry point calls the authorization verifier before creating a result-bearing run directory. The internal executor implements the frozen train-fold-only pipeline and task evidence writing.

`scripts/p4b_execute.py` has two mutually exclusive modes:

- `--validate-only`: package/preflight validation only; always prints `P4B_AUTHORIZED=false`, `RESULT_BEARING=false`, `SCIENTIFIC_EXECUTION=0`.
- `--execute-one`: future result-bearing mode; unusable without a separately approved exact authorization artifact.

`colab/P4B_EXECUTION_GATE.ipynb` is validation-only during package review.

## Future one-task authorization record

After package merge and independent Colab validation, the first scientific authorization must be created outside the repository as an immutable JSON artifact with:

- `schema_version = p4b-authorization/v1`
- `result_bearing_authorized = true`
- exact reviewed/merged 40-character Git SHA
- exactly one frozen `task_id`
- `max_tasks = 1`
- one immutable `P4_RUN_YYYYMMDDTHHMMSSZ_<12-char-sha>` run ID/root
- explicit authorization reference and timezone-aware UTC authorization timestamp
- exact final P4B policy SHA-256
- exact dataset/foundation/split/seed/P2/P3/P4A hashes
- exact accepted P4A `pip freeze` SHA-256
- exact canonical execution-plan SHA-256

The authorization artifact's own SHA-256 is calculated and recorded in the P4B run manifest and task evidence.

## Future first-task procedure — defined, not authorized by this package

1. Start a fresh Colab runtime.
2. Mount Drive.
3. Checkout the separately approved exact merged P4B commit, never a moving branch.
4. Install the exact frozen direct package pins.
5. Capture a complete normalized `pip freeze`; its physical SHA-256 must equal the accepted P4A receipt.
6. Run non-result-bearing P3/P4A preflight and regenerate the exact 3,600-task plan.
7. Select the exact task ID specified by the separate authorization. Do not choose the next pending task automatically.
8. Generate one unique immutable P4 run root string but do not create the directory until authorization validation succeeds.
9. Place/read the separately approved authorization artifact and verify it against the exact policy bytes, exact task, exact Git commit, exact run root, and accepted environment receipt.
10. Only after the authorization verifier returns a grant may the launcher create the run root and its required subdirectories.
11. Write the result-bearing pre-execution manifest, including the authorization artifact SHA-256.
12. Confirm the task is `PENDING`; write a single `.inprogress.json` marker.
13. Execute exactly the authorized task using the frozen train/test fold, task seed, scaling, imbalance treatment, classifier, threshold 0.50, and malignant positive class.
14. Write task split evidence, prediction-level evidence, resampling/CTGAN audit information, task metrics, timings, and SHA-256 receipts atomically.
15. Convert the single checkpoint marker atomically to `.complete.json` only after evidence has been written and hashed.
16. Stop after this one task. Do not automatically continue to another task.
17. Independently verify the one-task evidence before any expansion authorization.

## Interrupted-run behavior

`PENDING` means no marker exists. `INCOMPLETE` means only `<task>.inprogress.json` exists. `COMPLETE` means only `<task>.complete.json` exists. Both marker types for the same task are a hard failure.

An interrupted task is never assigned a new seed and is never silently rerun or overwritten. Partial evidence and the `INCOMPLETE` marker are retained for independent review. Recovery requires a controlled decision against the same frozen task identity.

## Result-bearing task semantics

The future executor uses:

- the frozen repeated-stratified outer fold;
- `StandardScaler` fitted only on the current outer training fold;
- test-fold transformation only with the training-fitted scaler;
- imbalance treatment on training data only;
- exact P2 parameters for baseline, ADASYN, Borderline-SMOTE, SMOTE, SMOTE-Tomek, and conditional CTGAN;
- exact P2 parameters for RF, AdaBoost, XGBoost, and LightGBM;
- threshold 0.50;
- malignant (`M`) as the positive class;
- deterministic task seed controls;
- untouched held-out observations until prediction/evaluation.

For CTGAN, the generator is fitted only to the current training fold, using diagnosis plus the 30 scaled predictors. Conditional generation requests `diagnosis=M` only and verifies the exact requested row count and labels.

## Evidence boundaries

P4B one-task evidence is raw scientific evidence. It is not permission to generate manuscript-level aggregate claims.

No primary tests, interaction tests, bootstrap intervals, final aggregate tables, publication figures, or manuscript results may be produced until the separate completeness/statistics gate is satisfied.

## Review acceptance criteria

Before merge, the exact P4B branch head must demonstrate:

- branch ancestry from `ffd7a29c8c05d17ac4e20dae9db75492d1ed1dff`;
- unchanged P0/P1/P2/P3/P4A identities;
- exact final P4B policy SHA-256 recorded;
- authorization verifier fail-closed;
- one-task `max_tasks=1` enforcement;
- no auto-selection/auto-continuation;
- non-result-bearing Colab validation path;
- negative tests proving the scientific executor is not reached without authorization;
- dedicated P4B tests PASS;
- full repository tests PASS;
- CI PASS at exact head;
- `P4B_AUTHORIZED=false`;
- `RESULT_BEARING=false`;
- `SCIENTIFIC_EXECUTION=0`.

Merge requires separate exact-head authorization. Merge alone does not authorize the first scientific task.
