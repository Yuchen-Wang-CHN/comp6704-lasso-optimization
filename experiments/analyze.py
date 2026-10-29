"""Produce tables and verified numerical claims from the stored experiment record."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

METHODS = ["ISTA", "FISTA", "Restarted FISTA", "ADMM"]


def escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write_table(path, headers, rows, align=None):
    align = align or "l" + "r" * (len(headers) - 1)
    body = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    body.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    body.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def mean_sd(series, digits=2, factor=1):
    return f"{series.mean() * factor:.{digits}f} $\\pm$ {series.std(ddof=1) * factor:.{digits}f}"


def analyze(results, direct, destination):
    results, direct, out = Path(results), Path(direct), Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(results / "runs.csv")
    direct_frame = pd.read_csv(direct / "runs.csv")
    selected = pd.read_csv(results / "selected_test.csv")
    baseline = frame[frame.experiment == "convergence"]
    validation = {"solver_runs": len(frame), "direct_solver_runs": len(direct_frame),
                  "all_converged": bool(frame.converged.all() and direct_frame.converged.all()),
                  "largest_relative_dual_gap": float(max(frame.relative_gap.max(), direct_frame.relative_gap.max())),
                  "largest_objective_difference_to_reference": float(max(frame.reference_gap.abs().max(), direct_frame.reference_gap.abs().max())),
                  "largest_prediction_disagreement": float(max(frame.prediction_disagreement.max(), direct_frame.prediction_disagreement.max())),
                  "minimum_reference_gap": float(frame.reference_gap.min()),
                  "admm_all_primal_pass": bool((frame[frame.method == "ADMM"].primal_residual <= frame[frame.method == "ADMM"].eps_primal).all()),
                  "admm_all_dual_pass": bool((frame[frame.method == "ADMM"].dual_residual <= frame[frame.method == "ADMM"].eps_dual).all())}
    if not validation["all_converged"] or validation["largest_relative_dual_gap"] > 1.01e-6:
        raise RuntimeError(f"Incomplete solver validation: {validation}")
    (out / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    rows = []
    summary = []
    for dataset in ("synthetic", "news"):
        for method in METHODS:
            sub = baseline[(baseline.dataset == dataset) & (baseline.method == method)]
            rows.append(["Synthetic" if dataset == "synthetic" else "News", method,
                         mean_sd(sub.iterations, 0), mean_sd(sub.total_sec, 2, 1000),
                         mean_sd(sub.solver_setup_sec, 2, 1000), mean_sd(sub.nonzeros, 1)])
            summary.append({"dataset": dataset, "method": method, "iterations_mean": sub.iterations.mean(),
                            "iterations_sd": sub.iterations.std(), "total_ms_mean": 1000 * sub.total_sec.mean(),
                            "total_ms_sd": 1000 * sub.total_sec.std(), "setup_ms_mean": 1000 * sub.solver_setup_sec.mean(),
                            "objective_mean": sub.objective.mean(), "gap_max": sub.relative_gap.max()})
    pd.DataFrame(summary).to_csv(out / "baseline_comparison.csv", index=False)
    write_table(out / "baseline.tex", ["Data", "Method", "Iterations", "Total (ms)", "Setup (ms)", "Nonzeros"], rows, "llrrrr")
    rows = []
    for dataset in ("synthetic", "news"):
        for method in METHODS:
            g = baseline[(baseline.dataset == dataset) & (baseline.method == method)]
            d = direct_frame[(direct_frame.dataset == dataset) & (direct_frame.method == method)]
            rows.append(["Synthetic" if dataset == "synthetic" else "News", method,
                         mean_sd(g.total_sec, 2, 1000), mean_sd(d.total_sec, 2, 1000)])
    write_table(out / "backend.tex", ["Data", "Method", "Gram total (ms)", "Direct total (ms)"], rows, "llrr")
    rows = []
    for dataset in ("synthetic", "news"):
        for method in METHODS:
            sub = selected[(selected.dataset == dataset) & (selected.method == method)]
            rows.append(["Synthetic" if dataset == "synthetic" else "News", method,
                         mean_sd(sub.test_mse, 4), mean_sd(sub.test_r2, 4), mean_sd(sub.nonzeros, 1)])
    write_table(out / "prediction.tex", ["Data", "Method", "Test MSE", "Test $R^2$", "Nonzeros"], rows, "llrrr")
    rows = []
    robust = frame[frame.experiment == "robustness"]
    for corr in sorted(robust.correlation.unique()):
        for method in METHODS:
            sub = robust[(robust.correlation == corr) & (robust.method == method)]
            rows.append([f"{corr:.1f}", method, mean_sd(sub.iterations, 0), mean_sd(sub.total_sec, 2, 1000),
                         f"{sub.support_f1.mean():.3f}", f"{sub.coefficient_relative_error.mean():.3f}"])
    write_table(out / "robustness.tex", ["Corr.", "Method", "Iterations", "Total (ms)", "F1", "Coef. error"], rows, "llrrrr")
    paired = baseline.pivot(index=["dataset", "seed"], columns="method", values="total_sec")
    paired["FISTA_over_restart"] = paired["FISTA"] / paired["Restarted FISTA"]
    paired.reset_index().to_csv(out / "paired_restart.csv", index=False)
    # Support stability across different splits of the same real dataset, at fixed lambda ratio.
    support_rows = []
    for ratio, sub in frame[(frame.dataset == "news") & (frame.experiment == "sensitivity") &
                             (frame.method == "Restarted FISTA")].groupby("lambda_ratio"):
        for a, b in itertools.combinations(sub.itertuples(), 2):
            sa = np.abs(np.load(results / a.solution_file)) > 1e-6
            sb = np.abs(np.load(results / b.solution_file)) > 1e-6
            union = np.count_nonzero(sa | sb)
            support_rows.append({"lambda_ratio": ratio, "seed_a": a.seed, "seed_b": b.seed,
                                 "jaccard": np.count_nonzero(sa & sb) / union if union else 1.0})
    pd.DataFrame(support_rows).to_csv(out / "news_support_stability.csv", index=False)
    claims = {"validation": validation, "baseline": summary,
              "restart_paired_speedup": paired.groupby(level="dataset").FISTA_over_restart.agg(["mean", "std", "min", "max"]).to_dict(orient="index"),
              "rho_iterations": frame[frame.experiment == "rho"].groupby(["dataset", "rho"]).iterations.mean().reset_index().to_dict(orient="records"),
              "selected_lambdas": selected[selected.method == "Restarted FISTA"][["dataset", "seed", "lambda_ratio", "test_mse", "test_r2"]].to_dict(orient="records")}
    (out / "findings.json").write_text(json.dumps(claims, indent=2), encoding="utf-8")
    source_hashes = {str(p.as_posix()): hashlib.sha256(p.read_bytes()).hexdigest()
                     for directory in (Path("src"), Path("experiments"), Path("configs"))
                     for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    (out / "source_snapshot_sha256.json").write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/full")
    parser.add_argument("--direct", default="results/direct")
    parser.add_argument("--output", default="report/tables")
    args = parser.parse_args()
    analyze(args.results, args.direct, args.output)
