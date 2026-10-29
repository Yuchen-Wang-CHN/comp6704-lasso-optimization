"""Generate report figures from measured CSV files, preserving source-data tables."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_panel_alignment import require_matplotlib_panel_alignment

METHODS = ["ISTA", "FISTA", "Restarted FISTA", "ADMM"]
COLORS = {"ISTA": "#64748B", "FISTA": "#4079A8", "Restarted FISTA": "#B36349", "ADMM": "#74669C"}
STYLES = {"ISTA": "--", "FISTA": "-.", "Restarted FISTA": "-", "ADMM": ":"}
DISPLAY = {"synthetic": "Synthetic", "news": "Online News Popularity"}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.7, "pdf.fonttype": 42, "svg.fonttype": "none"})


class Figures:
    def __init__(self, results, output):
        self.results, self.output = Path(results), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "source_data").mkdir(exist_ok=True)
        (self.output / "qa").mkdir(exist_ok=True)
        self.runs = pd.read_csv(self.results / "runs.csv")
        self.contracts = []

    def canvas(self, rows=1, cols=2, height=3.15):
        fig, axes = plt.subplots(rows, cols, figsize=(7.2, height), squeeze=False)
        # Fixed physical plot rectangles ensure consistent panel geometry.
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.21 if rows == 1 else 0.14,
                            top=0.81 if rows == 1 else 0.88, wspace=0.38, hspace=0.75)
        for i, ax in enumerate(axes.flat):
            ax.annotate(chr(97 + i), xy=(0, 1), xycoords="axes fraction", xytext=(-29, 14),
                        textcoords="offset points", fontsize=11, fontweight="bold", annotation_clip=False)
            ax.grid(axis="y", color="#E5E7EB", linewidth=0.5)
            ax.set_axisbelow(True)
        return fig, axes

    def legend(self, fig, axes):
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.995),
                   ncol=min(4, len(labels)), frameon=False, handlelength=2.1, columnspacing=1.3)

    def save(self, fig, stem, source, claim, roles, statistics):
        self.contracts.append({"figure": stem, "claim": claim, "panel_roles": roles,
                               "archetype": "quantitative grid", "backend": "Python / Matplotlib",
                               "statistics": statistics, "width_inches": 7.2,
                               "minimum_glyph_pt": 5, "source_data": f"source_data/{stem}.csv"})
        source.to_csv(self.output / "source_data" / f"{stem}.csv", index=False)
        (self.output / "figure_contracts.json").write_text(json.dumps(self.contracts, indent=2), encoding="utf-8")
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=str(self.output / "qa" / f"{stem}.alignment.json"),
                                           tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
        fig.savefig(self.output / f"{stem}.pdf")
        fig.savefig(self.output / f"{stem}.svg")
        fig.savefig(self.output / f"{stem}.png", dpi=300)
        plt.close(fig)
        root = Path(__file__).resolve().parents[1]
        audits = [(["audit_pdf_text.py", str(self.output / f"{stem}.pdf"), "--min-pt", "5", "--json"], "text"),
                  (["audit_figure_collisions.py", str(self.output / f"{stem}.pdf"),
                    "--json-out", str(self.output / "qa" / f"{stem}.collision.json")], "collision")]
        for args, name in audits:
            result = subprocess.run([sys.executable, str(root / "tools" / args[0]), *args[1:]],
                                    capture_output=True, text=True)
            (self.output / "qa" / f"{stem}.{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError(f"{stem}: {name} audit failed; inspect figures/qa")

    def traces(self, objective=False):
        selected = self.runs[(self.runs.experiment == "convergence") & (self.runs.seed == 0)]
        datasets = [s for s in ("synthetic", "news") if s in set(selected.dataset)]
        fig, axes = self.canvas(cols=len(datasets))
        source = []
        for ax, dataset in zip(axes.flat, datasets):
            for method in METHODS:
                row = selected[(selected.dataset == dataset) & (selected.method == method)].iloc[0]
                hist = pd.read_csv(self.results / row.history_file)
                if objective:
                    x, y = hist.iteration, hist.objective
                    ax.set_xscale("symlog", linthresh=5)
                    ax.set_xlabel("Iterations (symmetric log scale)")
                    ax.set_ylabel("LASSO objective")
                    ticks = [v for v in (0, 5, 50, 500, 5000) if v <= selected[selected.dataset == dataset].iterations.max() * 1.1]
                    ax.set_xticks(ticks)
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                    ax.xaxis.set_minor_formatter(NullFormatter())
                else:
                    x = hist.elapsed_sec * 1000
                    raw = hist.reference_gap / max(1, abs(row.reference_objective))
                    # Nonpositive reference differences are omitted on log axes.
                    # They are retained unchanged in source_data, never floored.
                    y = raw.where(raw > 0)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlabel("Elapsed time incl. setup (ms)")
                    ax.set_ylabel("Relative objective gap")
                    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                    ax.xaxis.set_minor_formatter(NullFormatter())
                ax.plot(x, y, color=COLORS[method], ls=STYLES[method], lw=1.5, label=method)
                hist = hist.assign(dataset=dataset, method=method, reference_objective=row.reference_objective)
                source.append(hist)
            ax.set_title(DISPLAY[dataset], pad=9)
        self.legend(fig, axes)
        stem = "01_objective_iterations" if objective else "02_gap_time"
        self.save(fig, stem, pd.concat(source, ignore_index=True),
                  "At seed 0, acceleration and splitting produce different objective trajectories under the same LASSO objective.",
                  ["Controlled synthetic convergence", "Convergence on real data"],
                  "Individual seed 0 trajectories; checkpoints every five iterations; no across-seed uncertainty claimed.")

    def scalability(self):
        fig, axes = self.canvas()
        selected = self.runs[self.runs.experiment.isin(["scale_samples", "scale_features"])]
        for ax, experiment, xcolumn, xlabel in zip(axes.flat, ["scale_samples", "scale_features"],
                                                  ["n_total", "d"], ["Total generated samples", "Features"]):
            for method in METHODS:
                summary = selected[(selected.experiment == experiment) & (selected.method == method)].groupby(xcolumn).total_sec.agg(["mean", "std"])
                ax.errorbar(summary.index, summary["mean"] * 1000, yerr=summary["std"].fillna(0) * 1000,
                            color=COLORS[method], ls=STYLES[method], marker="o", ms=3, lw=1.3, capsize=2, label=method)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_xticks(sorted(selected[selected.experiment == experiment][xcolumn].unique()))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_ylabel("Total optimization time (ms)")
            ax.set_title("Sample scaling" if experiment == "scale_samples" else "Feature scaling", pad=9)
        self.legend(fig, axes)
        self.save(fig, "03_scalability", selected,
                  "Runtime scaling depends on common Gram construction and the method's setup and iteration work.",
                  ["Effect of sample count at fixed feature count", "Effect of feature count at fixed sample count"],
                  "Mean and sample SD over five seeded datasets. Setup included; split/standardization excluded.")

    def sensitivity(self):
        selected = self.runs[(self.runs.experiment == "sensitivity") & (self.runs.method == "Restarted FISTA")]
        datasets = [s for s in ("synthetic", "news") if s in set(selected.dataset)]
        fig, axes = self.canvas(rows=2, cols=len(datasets), height=5.2)
        for col, dataset in enumerate(datasets):
            subset = selected[selected.dataset == dataset]
            for row, metric, ylabel in [(0, "nonzeros", "Nonzero coefficients"), (1, "valid_mse", "Validation MSE")]:
                ax = axes[row, col]
                summary = subset.groupby("lambda_ratio")[metric].agg(["mean", "std"])
                ax.errorbar(summary.index, summary["mean"], yerr=summary["std"].fillna(0), color=COLORS["Restarted FISTA"],
                            lw=1.4, marker="o", ms=3, capsize=2)
                ax.set_xscale("log")
                if row == 1 and dataset == "synthetic":
                    ax.set_yscale("log")
                ax.set_xlabel("Regularization ratio")
                ax.set_ylabel(ylabel)
                if row == 0:
                    ax.set_title(DISPLAY[dataset], pad=9)
        self.save(fig, "04_regularization", selected,
                  "The penalty controls sparsity and validation error; minimizing optimization error alone does not choose the predictive model.",
                  ["Synthetic sparsity", "Real-data sparsity", "Synthetic prediction tradeoff", "Real-data prediction tradeoff"],
                  "Mean and sample SD over five seeds; one solver displayed because converged solvers solve the same objective. Test set unused.")

    def restart(self):
        selected = self.runs[self.runs.experiment == "convergence"]
        paired = selected.pivot(index=["dataset", "seed"], columns="method", values="total_sec").reset_index()
        paired["speedup"] = paired["FISTA"] / paired["Restarted FISTA"]
        fig, axes = self.canvas()
        for ax, dataset in zip(axes.flat, ("synthetic", "news")):
            subset = paired[paired.dataset == dataset]
            ax.axhline(1, color="#9CA3AF", lw=0.8, ls="--")
            ax.plot(subset.seed, subset.speedup, marker="o", color=COLORS["Restarted FISTA"], lw=1.4)
            ax.set_xticks(subset.seed)
            ax.set_xlabel("Matched random seed")
            ax.set_ylabel("FISTA time / restarted FISTA time")
            ax.set_title(DISPLAY[dataset], pad=9)
        self.save(fig, "05_restart_paired", paired,
                  "Adaptive restart improves measured runtime relative to FISTA in each of the matched convergence seeds.",
                  ["Paired synthetic runtime benefit", "Paired real-data runtime benefit"],
                  "Raw paired ratios for seeds 0-4. Dashed line is equal runtime; points are not repeated timing measurements.")

    def rho(self):
        selected = self.runs[self.runs.experiment == "rho"]
        fig, axes = self.canvas()
        for ax, dataset in zip(axes.flat, ("synthetic", "news")):
            summary = selected[selected.dataset == dataset].groupby("rho").iterations.agg(["mean", "std"])
            ax.errorbar(summary.index, summary["mean"], yerr=summary["std"].fillna(0), marker="o", color=COLORS["ADMM"],
                        capsize=3, ms=4, lw=1.3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("ADMM penalty rho")
            ax.set_ylabel("Iterations to joint stopping rule")
            ax.set_title(DISPLAY[dataset], pad=9)
        self.save(fig, "06_admm_rho", selected,
                  "ADMM's convergence depends strongly on rho, and the most efficient tested value can differ by dataset.",
                  ["Synthetic penalty sensitivity", "Real-data penalty sensitivity"],
                  "Mean and sample SD over five seeded datasets/splits. Same dual-gap and residual thresholds at every rho.")

    def residuals(self):
        selected = self.runs[(self.runs.experiment == "convergence") & (self.runs.seed == 0) & (self.runs.method == "ADMM")]
        fig, axes = self.canvas()
        sources = []
        for ax, dataset in zip(axes.flat, ("synthetic", "news")):
            row = selected[selected.dataset == dataset].iloc[0]
            hist = pd.read_csv(self.results / row.history_file)
            hist = hist.assign(dataset=dataset)
            for residual, tolerance, color, label in [("primal_residual", "eps_primal", "#4079A8", "Primal"),
                                                      ("dual_residual", "eps_dual", "#B36349", "Dual")]:
                ratio = hist[residual] / hist[tolerance]
                ax.plot(hist.iteration, ratio.where(ratio > 0), color=color, lw=1.3, label=label)
            ax.axhline(1, color="#9CA3AF", ls="--", lw=0.8)
            ax.set_yscale("log")
            ax.set_xlabel("ADMM iterations")
            ax.set_ylabel("Residual / stopping threshold")
            ax.set_title(DISPLAY[dataset], pad=9)
            sources.append(hist)
        self.legend(fig, axes)
        self.save(fig, "07_admm_residuals", pd.concat(sources, ignore_index=True),
                  "The reported ADMM endpoint satisfies both residual checks in addition to the common dual-gap test.",
                  ["Synthetic feasibility and stationarity diagnostics", "Real-data feasibility and stationarity diagnostics"],
                  "Individual seed 0 trajectories. Zero initial residuals are omitted on logarithmic axes and preserved in source CSV.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/full")
    parser.add_argument("--output", default="figures")
    args = parser.parse_args()
    plots = Figures(args.results, args.output)
    plots.traces(objective=True)
    plots.traces()
    plots.scalability()
    plots.sensitivity()
    plots.restart()
    plots.rho()
    plots.residuals()


if __name__ == "__main__":
    main()
