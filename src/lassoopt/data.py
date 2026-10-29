"""Seeded synthetic data and official UCI download with a SHA-256 manifest."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen, Request
from zipfile import ZipFile
import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip"


def synthetic(n=5000, d=200, sparsity=0.1, noise=1.0, correlation=0.5, seed=0):
    if not (0 <= correlation < 1 and 0 < sparsity <= 1 and noise >= 0):
        raise ValueError("invalid synthetic configuration")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    # AR(1) columns give population covariance corr^|i-j| in O(nd).
    for j in range(1, d):
        X[:, j] = correlation * X[:, j - 1] + np.sqrt(1 - correlation**2) * X[:, j]
    w = np.zeros(d)
    support = rng.choice(d, max(1, round(d * sparsity)), replace=False)
    w[support] = rng.choice([-1.0, 1.0], len(support)) * rng.uniform(0.5, 2.0, len(support))
    y = 2.0 + X @ w + rng.normal(scale=noise, size=n)
    return X, y, w


def load_news(data_dir):
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "online_news_popularity.zip"
    csv = directory / "OnlineNewsPopularity.csv"
    if not archive.exists():
        request = Request(UCI_URL, headers={"User-Agent": "COMP6704-reproducible-research/0.1"})
        temporary = archive.with_suffix(".part")
        with urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        temporary.replace(archive)
    with ZipFile(archive) as bundle:
        name = next(n for n in bundle.namelist() if n.endswith("OnlineNewsPopularity.csv"))
        raw_csv = bundle.read(name)
    csv.write_bytes(raw_csv)
    frame = pd.read_csv(csv)
    frame.columns = frame.columns.str.strip()
    target = np.log1p(frame["shares"].to_numpy(dtype=float))
    features = frame.drop(columns=["url", "timedelta", "shares"])
    X = features.to_numpy(dtype=float)
    if X.shape != (39644, 58) or not np.isfinite(X).all() or not np.isfinite(target).all():
        raise ValueError("Unexpected UCI schema or nonfinite values; inspect before proceeding")
    manifest = {"source": UCI_URL, "landing_page": "https://archive.ics.uci.edu/dataset/332/online+news+popularity",
                "doi": "10.24432/C5NS3V", "license": "CC BY 4.0",
                "creators": "Fernandes, Vinagre, Cortez, and Sernadela (2015)",
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "csv_sha256": hashlib.sha256(raw_csv).hexdigest(),
                "rows": len(frame), "features": list(features.columns),
                "removed_columns": ["url", "timedelta"], "rows_removed": 0,
                "target": "log1p(shares)", "missing_value_policy": "fail, never silently drop rows"}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["archive_sha256"] != manifest["archive_sha256"]:
            raise ValueError("Cached dataset checksum changed")
        manifest["retrieved_utc"] = previous["retrieved_utc"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return X, target
