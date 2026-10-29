"""Reproduce the real-data validation/outlier inspection without deleting data."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lassoopt.data import load_news
from lassoopt.preprocessing import split_and_standardize


def inspect(results, output):
    results, output = Path(results), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(results / "runs.csv")
    rows = frame[(frame.dataset == "news") & (frame.experiment == "sensitivity") &
                 (frame.method == "Restarted FISTA")]
    X, y = load_news("data/raw")
    names = json.loads(Path("data/raw/manifest.json").read_text(encoding="utf-8"))["features"]
    diagnostics = []
    for seed, subset in rows.groupby("seed"):
        split = split_and_standardize(X, y, int(seed))
        Xv, yv = split["valid"]
        location = np.unravel_index(np.abs(Xv).argmax(), Xv.shape)
        for row in subset.itertuples():
            coefficients = np.load(results / row.solution_file)
            errors = (Xv @ coefficients - yv)**2
            largest = int(errors.argmax())
            diagnostics.append({"seed": int(seed), "lambda_ratio": row.lambda_ratio,
                                "validation_mse": float(errors.mean()),
                                "largest_squared_error": float(errors[largest]),
                                "largest_error_fraction": float(errors[largest] / errors.sum()),
                                "largest_error_original_row": int(split["indices"]["valid"][largest]),
                                "max_abs_validation_z": float(abs(Xv[location])),
                                "extreme_feature": names[location[1]],
                                "extreme_feature_original_row": int(split["indices"]["valid"][location[0]]),
                                "all_rows_retained": True})
    pd.DataFrame(diagnostics).to_csv(output / "validation_outliers.csv", index=False)
    print(f"Saved {len(diagnostics)} validation diagnostics without modifying the dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/full")
    parser.add_argument("--output", default="report/tables")
    args = parser.parse_args()
    inspect(args.results, args.output)
