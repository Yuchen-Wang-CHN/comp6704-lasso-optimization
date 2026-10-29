"""Controlled experiment runner. Test outcomes are computed only after selection."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits, threadpool_info
from lassoopt import LassoProblem, ista, fista, admm
from lassoopt.data import synthetic, load_news
from lassoopt.preprocessing import split_and_standardize

METHODS = ["ISTA", "FISTA", "Restarted FISTA", "ADMM"]


def reference_solution(problem, lam, tol):
    started = perf_counter()
    model = Lasso(alpha=lam, fit_intercept=False, tol=tol, max_iter=200000,
                  selection="cyclic", precompute=problem.n * problem.gram if problem.gram is not None else False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(problem.X, problem.y)
    cert = problem.certificate(model.coef_, lam, direct=True)
    if cert["relative_gap"] > 1e-9 or any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError(f"Reference solution failed certification: {cert}")
    return model.coef_, cert, perf_counter() - started


class Runner:
    def __init__(self, config, output, use_news):
        self.cfg = config
        self.out = Path(output)
        for name in ("histories", "solutions", "splits"):
            (self.out / name).mkdir(parents=True, exist_ok=True)
        self.rows, self.selection = [], []
        self.news = load_news(Path("data/raw")) if use_news else None
        self.case_number = 0

    def dataset(self, dataset, seed, overrides=None):
        settings = {**self.cfg["base"], **(overrides or {})}
        if dataset == "news":
            X, y = self.news
            truth = None
        else:
            X, y, truth = synthetic(**settings, seed=seed)
        started = perf_counter()
        split = split_and_standardize(X, y, seed)
        preprocessing_sec = perf_counter() - started
        problem = LassoProblem.from_data(*split["train"], backend=self.cfg["backend"])
        return problem, split, truth, settings, len(y), preprocessing_sec

    def case(self, experiment, dataset, seed, ratio=0.03, overrides=None,
             prepared=None, methods=None, rho=None):
        prepared = prepared or self.dataset(dataset, seed, overrides)
        problem, split, truth, settings, n_total, preprocessing_sec = prepared
        self.case_number += 1
        case_id = f"case_{self.case_number:04d}"
        lam = ratio * problem.lambda_max
        reference, reference_cert, reference_sec = reference_solution(problem, lam, self.cfg["reference_tol"])
        np.savez_compressed(self.out / "splits" / f"{case_id}.npz", **split["indices"])
        np.save(self.out / "solutions" / f"{case_id}_reference.npy", reference)
        order = list(methods or METHODS)
        np.random.default_rng(seed + 1009 * self.case_number).shuffle(order)
        case_rows = []
        for order_index, method in enumerate(order):
            kwargs = {"tol": self.cfg["tol"], "max_iter": self.cfg["max_iter"],
                      "check_every": self.cfg["check_every"]}
            actual_rho = self.cfg["rho"] if rho is None else rho
            if method == "ADMM":
                result = admm(problem, lam, rho=actual_rho, abs_tol=self.cfg["abs_tol"],
                              rel_tol=self.cfg["rel_tol"], **kwargs)
            elif method == "ISTA":
                result = ista(problem, lam, **kwargs)
            else:
                result = fista(problem, lam, restart=method == "Restarted FISTA", **kwargs)
            final = problem.certificate(result.coefficients, lam, direct=True)
            if result.converged and final["relative_gap"] > self.cfg["tol"] * 1.01 + 1e-11:
                raise RuntimeError("Gram and direct final certificates disagree")
            history = pd.DataFrame(result.history)
            history["reference_gap"] = history["objective"] - reference_cert["objective"]
            slug = method.lower().replace(" ", "_")
            history_file = f"histories/{case_id}_{slug}.csv"
            history.to_csv(self.out / history_file, index=False)
            solution_file = f"solutions/{case_id}_{slug}.npy"
            np.save(self.out / solution_file, result.coefficients)
            Xv, yv = split["valid"]
            valid_mse = float(np.mean((Xv @ result.coefficients - yv) ** 2))
            row = {"case_id": case_id, "experiment": experiment, "dataset": dataset,
                   "seed": seed, "method": method, "order": order_index, "n_total": n_total,
                   "n_train": problem.n, "d": problem.d, "lambda_ratio": ratio, "lambda": lam,
                   "rho": actual_rho if method == "ADMM" else np.nan,
                   "noise": settings["noise"] if dataset == "synthetic" else np.nan,
                   "correlation": settings["correlation"] if dataset == "synthetic" else np.nan,
                   "sparsity": settings["sparsity"] if dataset == "synthetic" else np.nan,
                   "backend": problem.backend, "converged": result.converged,
                   "iterations": result.iterations, "restarts": result.restarts,
                   "preprocessing_sec": preprocessing_sec,
                   "common_setup_sec": result.common_setup_sec,
                   "solver_setup_sec": result.solver_setup_sec,
                   "iteration_sec": result.iteration_sec, "total_sec": result.total_sec,
                   "time_per_iteration_sec": result.iteration_sec / max(1, result.iterations),
                   "data_and_gram_mib": (problem.X.nbytes + problem.y.nbytes +
                                         (problem.gram.nbytes if problem.gram is not None else 0)) / 2**20,
                   "reference_sec": reference_sec, "reference_objective": reference_cert["objective"],
                   "reference_dual_gap": reference_cert["dual_gap"],
                   "reference_gap": final["objective"] - reference_cert["objective"],
                   "prediction_disagreement": float(np.linalg.norm(problem.X @ (result.coefficients - reference)) /
                                                     max(1.0, np.linalg.norm(problem.X @ reference))),
                   "nonzeros": int(np.count_nonzero(np.abs(result.coefficients) > 1e-6)),
                   "valid_mse": valid_mse, "history_file": history_file,
                   "solution_file": solution_file, **final}
            for key in ("primal_residual", "dual_residual", "eps_primal", "eps_dual"):
                row[key] = result.history[-1].get(key, np.nan)
            if truth is not None:
                original, _ = split["scaler"].original_parameters(result.coefficients)
                recovered, support = np.abs(original) > 1e-6, truth != 0
                tp = np.count_nonzero(recovered & support)
                fp = np.count_nonzero(recovered & ~support)
                fn = np.count_nonzero(~recovered & support)
                row["support_f1"] = float(2 * tp / max(1, 2 * tp + fp + fn))
                row["coefficient_relative_error"] = float(np.linalg.norm(original - truth) / np.linalg.norm(truth))
            self.rows.append(row)
            case_rows.append(row)
        # Checkpoint after every case. Never silently reuse an incomplete run.
        pd.DataFrame(self.rows).to_csv(self.out / "runs.csv", index=False)
        print(f"{case_id} {experiment:14s} {dataset:9s} seed={seed} n={n_total} d={problem.d} "
              f"lambda/max={ratio:g}: " + ", ".join(f"{r['method']} {r['iterations']} "
              f"({'OK' if r['converged'] else 'LIMIT'})" for r in case_rows), flush=True)
        return case_rows

    def run(self, experiments):
        datasets = ["synthetic"] + (["news"] if self.news is not None else [])
        for seed in self.cfg["seeds"]:
            if "convergence" in experiments:
                for dataset in datasets:
                    self.case("convergence", dataset, seed)
            if "scalability" in experiments:
                for n in self.cfg["sample_sizes"]:
                    self.case("scale_samples", "synthetic", seed, overrides={"n": n})
                for d in self.cfg["feature_sizes"]:
                    self.case("scale_features", "synthetic", seed, overrides={"d": d})
            if "sensitivity" in experiments:
                for dataset in datasets:
                    prepared = self.dataset(dataset, seed)
                    candidates = []
                    for ratio in self.cfg["lambda_ratios"]:
                        candidates.extend(self.case("sensitivity", dataset, seed, ratio=ratio, prepared=prepared))
                    # Select one shared lambda using reference-validation MSE only.
                    # No test-set outcomes exist in runs.csv or influence selection.
                    problem, split, _, _, _, _ = prepared
                    validation = []
                    Xv, yv = split["valid"]
                    for case_id in dict.fromkeys(r["case_id"] for r in candidates):
                        ref = np.load(self.out / "solutions" / f"{case_id}_reference.npy")
                        validation.append((float(np.mean((Xv @ ref - yv) ** 2)), case_id))
                    best_mse, best_id = min(validation)
                    Xt, yt = split["test"]
                    for row in candidates:
                        if row["case_id"] == best_id:
                            w = np.load(self.out / row["solution_file"])
                            test_mse = float(np.mean((Xt @ w - yt) ** 2))
                            baseline_mse = float(np.mean(yt**2))
                            self.selection.append({"dataset": dataset, "seed": seed,
                                                   "lambda_ratio": row["lambda_ratio"],
                                                   "method": row["method"], "valid_reference_mse": best_mse,
                                                   "test_mse": test_mse, "baseline_test_mse": baseline_mse,
                                                   "test_r2": 1 - test_mse / float(np.var(yt)),
                                                   "nonzeros": row["nonzeros"], "case_id": best_id})
            if "robustness" in experiments:
                for noise in self.cfg["noise_levels"]:
                    for corr in self.cfg["correlations"]:
                        for sparsity in self.cfg["sparsities"]:
                            self.case("robustness", "synthetic", seed,
                                      overrides={"noise": noise, "correlation": corr, "sparsity": sparsity})
            if "rho" in experiments:
                for dataset in datasets:
                    prepared = self.dataset(dataset, seed)
                    for rho in self.cfg["rho_values"]:
                        self.case("rho", dataset, seed, prepared=prepared, methods=["ADMM"], rho=rho)
        pd.DataFrame(self.selection).to_csv(self.out / "selected_test.csv", index=False)
        frame = pd.DataFrame(self.rows)
        keys = ["experiment", "dataset", "method", "n_total", "d", "lambda_ratio", "noise", "correlation", "sparsity", "rho"]
        metrics = ["total_sec", "iteration_sec", "iterations", "relative_gap", "nonzeros", "valid_mse", "support_f1"]
        metrics = [m for m in metrics if m in frame.columns]
        grouped = frame.groupby(keys, dropna=False)[metrics].agg(["mean", "std", "count"])
        grouped.columns = ["_".join(c) for c in grouped.columns]
        grouped.reset_index().to_csv(self.out / "summary.csv", index=False)
        return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--experiments", nargs="+", default=["convergence", "scalability", "sensitivity", "robustness", "rho"],
                        choices=["convergence", "scalability", "sensitivity", "robustness", "rho"])
    parser.add_argument("--skip-news", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(args.output or f"results/{cfg['name']}")
    if (output / "runs.csv").exists():
        raise FileExistsError(f"{output} already contains results; choose a fresh --output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with threadpool_limits(limits=1):
        # Warm up common numerical kernels before starting method timers.
        warm = LassoProblem.from_data(np.eye(5), np.arange(5.0))
        for method in (ista, fista, admm):
            method(warm, 0.1, max_iter=10)
        metadata = {"started_utc": datetime.now(timezone.utc).isoformat(),
                    "python": platform.python_version(), "platform": platform.platform(),
                    "processor": platform.processor(), "blas": threadpool_info(),
                    "packages": {p: version(p) for p in ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "threadpoolctl")},
                    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    "config": cfg, "experiments": args.experiments, "news": not args.skip_news,
                    "thread_limit": 1, "timer": "time.perf_counter",
                    "timing_scope": "common sufficient statistics + method setup + iterations including diagnostics; excludes split/standardization, reference, I/O, prediction"}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        runner = Runner(cfg, output, not args.skip_news)
        frame = runner.run(args.experiments)
    metadata.update({"finished_utc": datetime.now(timezone.utc).isoformat(),
                     "wall_sec": perf_counter() - started, "solver_runs": len(frame),
                     "converged_runs": int(frame.converged.sum())})
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Completed {len(frame)} solver runs; {int(frame.converged.sum())} converged. Output: {output}")


if __name__ == "__main__":
    main()
