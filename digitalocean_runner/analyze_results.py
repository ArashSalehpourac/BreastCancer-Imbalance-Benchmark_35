#!/usr/bin/env python3
"""Aggregate a completed benchmark into tables, inference, figures, and paper text."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.config import MASTER_SEED, METHOD_CODES, MODEL_CODES
from src.metrics import calculate_metrics

INFERENCE_METRICS = ["malignant_recall", "malignant_precision", "pr_auc", "macro_f1",
                     "balanced_accuracy", "mcc", "specificity", "roc_auc"]

def holm(pvalues: list[float]) -> tuple[list[float], list[bool]]:
    p = np.asarray(pvalues, dtype=float); order = np.argsort(p); adjusted = np.empty(len(p)); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index])); adjusted[index] = running
    return adjusted.tolist(), (adjusted < 0.05).tolist()

def load_tasks(root: Path, allow_partial: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted((root / "tasks").glob("*.json"))
    tasks, predictions, identities = [], [], set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8")); ident = payload["identity"]
        key = (ident["repeat"], ident["fold"], ident["method"], ident["classifier"], ident["replicate"])
        if key in identities: raise ValueError(f"duplicate task identity: {key}")
        identities.add(key); tasks.append({**ident, **payload["metrics"]}); predictions.extend(payload["predictions"])
    expected = {(r, f, method, model, rep) for r in range(5) for f in range(10)
                for method in METHOD_CODES for model in MODEL_CODES for rep in range(3)}
    if not allow_partial and identities != expected:
        raise RuntimeError(f"final analysis requires all 3600 tasks; found {len(identities)}, missing={len(expected-identities)}, unexpected={len(identities-expected)}")
    if not tasks: raise RuntimeError("no valid task artifacts found")
    return pd.DataFrame(tasks), pd.DataFrame(predictions)

def subject_table(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["row_id", "method", "classifier"]
    truth_counts = predictions.groupby(keys)["y_true"].nunique()
    if (truth_counts != 1).any(): raise ValueError("inconsistent y_true for a subject-condition")
    result = predictions.groupby(keys, as_index=False).agg(y_true=("y_true", "first"),
        p_malignant=("p_malignant", "mean"), n_predictions=("p_malignant", "size"))
    result["y_pred"] = (result["p_malignant"] >= 0.5).astype(int)
    return result

def condition_metrics(subjects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, classifier), group in subjects.groupby(["method", "classifier"], sort=False):
        rows.append({"method": method, "classifier": classifier,
                     **calculate_metrics(group.y_true, group.p_malignant)})
    return pd.DataFrame(rows)

def bootstrap(subjects: pd.DataFrame, iterations: int = 10_000) -> pd.DataFrame:
    conditions = {(m, c): g.sort_values("row_id") for (m, c), g in subjects.groupby(["method", "classifier"])}
    reference = next(iter(conditions.values()))[["row_id", "y_true"]].reset_index(drop=True)
    for key, group in conditions.items():
        if not group[["row_id", "y_true"]].reset_index(drop=True).equals(reference): raise ValueError(f"subjects differ for {key}")
    y = reference.y_true.to_numpy(dtype=int); strata = [np.flatnonzero(y == label) for label in (0, 1)]
    rng = np.random.default_rng(MASTER_SEED)
    # Multinomial counts are exactly the subject frequencies produced by sampling
    # each class with replacement. One shared matrix makes the bootstrap paired.
    counts = np.zeros((iterations, len(y)), dtype=np.int16)
    for index in strata:
        counts[:, index] = rng.multinomial(len(index), np.full(len(index), 1 / len(index)), size=iterations)
    distributions = {}
    n_negative, n_positive = map(len, strata)
    safe_div = lambda numerator, denominator: np.divide(numerator, denominator,
        out=np.zeros_like(numerator, dtype=float), where=np.asarray(denominator) != 0)
    for key, group in conditions.items():
        probability = group.p_malignant.to_numpy(dtype=float); predicted = (probability >= 0.5).astype(int)
        tp = (counts @ ((y == 1) & (predicted == 1))).astype(float); fn = n_positive - tp
        tn = (counts @ ((y == 0) & (predicted == 0))).astype(float); fp = n_negative - tn
        recall = tp / n_positive; precision = safe_div(tp, tp + fp); specificity = tn / n_negative
        f1_positive = safe_div(2 * tp, 2 * tp + fp + fn); f1_negative = safe_div(2 * tn, 2 * tn + fp + fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = safe_div(tp * tn - fp * fn, denominator)
        # Weighted average precision from descending score ranks.
        descending = np.argsort(-probability, kind="stable"); ranked_counts = counts[:, descending]
        ranked_y = y[descending]; cumulative_total = np.cumsum(ranked_counts, axis=1)
        cumulative_positive = np.cumsum(ranked_counts * ranked_y, axis=1)
        ap = ((safe_div(cumulative_positive, cumulative_total) * ranked_counts * ranked_y).sum(axis=1) / n_positive)
        # Weighted Mann-Whitney form of ROC-AUC, including half credit for ties.
        auc_numerator = np.zeros(iterations); negatives_before = np.zeros(iterations)
        for score in np.unique(probability):
            tied = probability == score; positive_weight = counts[:, tied & (y == 1)].sum(axis=1)
            negative_weight = counts[:, tied & (y == 0)].sum(axis=1)
            auc_numerator += positive_weight * (negatives_before + 0.5 * negative_weight)
            negatives_before += negative_weight
        distributions[key] = {"malignant_recall": recall, "malignant_precision": precision,
            "pr_auc": ap, "macro_f1": (f1_positive + f1_negative) / 2,
            "balanced_accuracy": (recall + specificity) / 2, "mcc": mcc,
            "specificity": specificity, "roc_auc": auc_numerator / (n_positive * n_negative)}
    rows = []
    for (method, classifier), metrics in distributions.items():
        point = calculate_metrics(y, conditions[(method, classifier)].p_malignant)
        for metric, values in metrics.items():
            lower, upper = np.quantile(values, [0.025, 0.975])
            rows.append({"method": method, "classifier": classifier, "metric": metric,
                         "estimate": point[metric], "ci_lower": lower, "ci_upper": upper,
                         "bootstrap_iterations": iterations})
    return pd.DataFrame(rows)

def mcnemar_tests(subjects: pd.DataFrame) -> pd.DataFrame:
    malignant = subjects[subjects.y_true == 1]
    pivot = malignant.pivot(index="row_id", columns=["method", "classifier"], values="y_pred")
    rows = []
    available = set(pivot.columns)
    for model in MODEL_CODES:
        if ("baseline", model) not in available: continue
        baseline = pivot[("baseline", model)].to_numpy()
        for method in [m for m in METHOD_CODES if m != "baseline"]:
            if (method, model) not in available: continue
            candidate = pivot[(method, model)].to_numpy(); b = int(((baseline == 1) & (candidate == 0)).sum()); c = int(((baseline == 0) & (candidate == 1)).sum())
            p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
            rows.append({"method": method, "classifier": model, "baseline_only_correct": b,
                         "resampler_only_correct": c, "recall_difference": float((candidate-baseline).mean()), "p_value": p})
    adjusted, reject = holm([row["p_value"] for row in rows])
    for row, adj, sig in zip(rows, adjusted, reject): row.update(p_holm=adj, significant_holm=sig)
    return pd.DataFrame(rows)

def interaction_tests(subjects: pd.DataFrame, permutations: int = 100_000) -> pd.DataFrame:
    malignant = subjects[subjects.y_true == 1]
    pivot = malignant.pivot(index="row_id", columns=["method", "classifier"], values="y_pred")
    rng = np.random.default_rng(MASTER_SEED); rows = []
    available = set(pivot.columns)
    for method in [m for m in METHOD_CODES if m != "baseline"]:
        for model_a, model_b in combinations(MODEL_CODES, 2):
            needed = {(method, model_a), ("baseline", model_a), (method, model_b), ("baseline", model_b)}
            if not needed <= available: continue
            values = ((pivot[(method, model_a)] - pivot[("baseline", model_a)]) -
                      (pivot[(method, model_b)] - pivot[("baseline", model_b)])).to_numpy(dtype=float)
            observed = float(values.mean()); extreme = 0
            for start in range(0, permutations, 10_000):
                n = min(10_000, permutations - start)
                signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(values)))
                extreme += int((np.abs((signs * values).mean(axis=1)) >= abs(observed) - 1e-15).sum())
            p = (extreme + 1) / (permutations + 1)
            rows.append({"method": method, "classifier_a": model_a, "classifier_b": model_b,
                         "difference_in_differences": observed, "permutations": permutations, "p_value": p})
    adjusted, reject = holm([row["p_value"] for row in rows])
    for row, adj, sig in zip(rows, adjusted, reject): row.update(p_holm=adj, significant_holm=sig)
    return pd.DataFrame(rows)

def save_heatmap(summary: pd.DataFrame, metric: str, path: Path) -> None:
    matrix = summary.pivot(index="method", columns="classifier", values=metric).reindex(index=METHOD_CODES, columns=MODEL_CODES)
    fig, ax = plt.subplots(figsize=(7, 5)); image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns); ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)): ax.text(j, i, f"{matrix.iloc[i,j]:.3f}", ha="center", va="center", color="white")
    ax.set_title(metric.replace("_", " ").title()); fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)

def figures(summary: pd.DataFrame, intervals: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_heatmap(summary, "malignant_recall", directory / "malignant_recall_heatmap.png")
    save_heatmap(summary, "pr_auc", directory / "pr_auc_heatmap.png")
    ci = intervals[intervals.metric == "malignant_recall"].copy(); ci["condition"] = ci.method + " / " + ci.classifier; ci = ci.sort_values("estimate")
    fig, ax = plt.subplots(figsize=(8, 8)); y = np.arange(len(ci)); ax.errorbar(ci.estimate, y, xerr=[ci.estimate-ci.ci_lower, ci.ci_upper-ci.estimate], fmt="o", capsize=3)
    ax.set_yticks(y, ci.condition); ax.set_xlabel("Malignant recall (95% bootstrap CI)"); ax.set_xlim(0, 1.01); fig.tight_layout(); fig.savefig(directory / "malignant_recall_95ci.png", dpi=300); plt.close(fig)

def write_paper(summary: pd.DataFrame, intervals: pd.DataFrame, mcnemar: pd.DataFrame, interactions: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    best_recall = summary.loc[summary.malignant_recall.idxmax()]; best_pr = summary.loc[summary.pr_auc.idxmax()]
    ci = intervals[(intervals.method == best_recall.method) & (intervals.classifier == best_recall.classifier) & (intervals.metric == "malignant_recall")].iloc[0]
    text = (f"The highest subject-level malignant recall was {best_recall.malignant_recall:.4f} "
            f"(95% paired stratified bootstrap CI {ci.ci_lower:.4f}–{ci.ci_upper:.4f}) for {best_recall.method} + {best_recall.classifier}. "
            f"For this condition, PR-AUC was {best_recall.pr_auc:.4f}, specificity was {best_recall.specificity:.4f}, "
            f"balanced accuracy was {best_recall.balanced_accuracy:.4f}, and MCC was {best_recall.mcc:.4f}. "
            f"The highest PR-AUC was {best_pr.pr_auc:.4f} for {best_pr.method} + {best_pr.classifier}. "
            f"Holm correction identified {int(mcnemar.get('significant_holm', pd.Series(dtype=bool)).sum())} significant resampler-versus-baseline recall comparisons "
            f"and {int(interactions.get('significant_holm', pd.Series(dtype=bool)).sum())} significant classifier interaction tests.")
    (directory / "METHODS_RESULTS_DRAFT.md").write_text("# Methods and Results Draft\n\nSubject probabilities were averaged across five repeats and three stochastic replicates, then thresholded at 0.50. Confidence intervals used 10,000 paired stratified subject bootstraps. Primary comparisons used exact McNemar/binomial tests with Holm correction; interaction tests used 100,000 paired sign-flip permutations with Holm correction.\n\n## Results\n\n" + text + "\n", encoding="utf-8")
    (directory / "RESULTS_ONLY.txt").write_text(text + "\n", encoding="utf-8")

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--results", default="results"); p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args(argv); root = Path(args.results).resolve(); tables = root / "tables"; tables.mkdir(parents=True, exist_ok=True)
    tasks, predictions = load_tasks(root, args.allow_partial)
    tasks.to_csv(tables / "fold_task_metrics.csv", index=False); predictions.to_csv(tables / "predictions_all.csv", index=False)
    audits = [json.loads(path.read_text()) for path in sorted((root / "resampling_audits").glob("*.json"))]
    pd.DataFrame(audits).to_csv(tables / "resampling_audit.csv", index=False)
    subjects = subject_table(predictions); subjects.to_csv(tables / "subject_level_predictions.csv", index=False)
    summary = condition_metrics(subjects); summary.to_csv(tables / "condition_summary_subject_level.csv", index=False)
    aggregate = tasks.groupby(["method", "classifier"])[[c for c in tasks.columns if c not in {"repeat","fold","method","classifier","replicate"}]].agg(["mean", "std"])
    aggregate.columns = [f"{a}_{b}" for a, b in aggregate.columns]; aggregate.reset_index().to_csv(tables / "condition_summary_fold_replicate.csv", index=False)
    summary[["method", "classifier", *INFERENCE_METRICS]].to_csv(tables / "table_main_performance.csv", index=False)
    intervals = bootstrap(subjects); intervals.to_csv(tables / "bootstrap_intervals.csv", index=False)
    mc = mcnemar_tests(subjects); mc.to_csv(tables / "primary_mcnemar_holm.csv", index=False)
    interactions = interaction_tests(subjects); interactions.to_csv(tables / "interaction_permutation_holm.csv", index=False)
    figures(summary, intervals, root / "figures"); write_paper(summary, intervals, mc, interactions, root / "paper")
    print(f"analysis complete: tasks={len(tasks)} predictions={len(predictions)}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
