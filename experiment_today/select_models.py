"""Select pruning thresholds for the frozen current-33 H=40/60/100 models."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pruning"))
from threshold_tools import load_model, predict_rgb8, prune, run


def prepare(hidden_sizes):
    here = Path(__file__).resolve().parent
    validation = ROOT / "pruning/validation.csv"
    with (ROOT / "retraining/data/current33.csv").open(newline="") as f:
        records = list(csv.DictReader(f))
    x = np.array([[r[f"input_{c}"] for c in "rgb"] for r in records], dtype=np.float32)
    target = np.array([[r[f"target_{c}8"] for c in "rgb"] for r in records], dtype=np.float64)
    for h in hidden_sizes:
        source = ROOT / f"retraining/results/h{h}/model_parameters.h"
        directory = here / f"selection/h{h}"
        saved = directory / "summary.json"
        if saved.exists():
            report = json.loads(saved.read_text())
            if (report["model_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest() or
                    report["validation_csv_sha256"] != hashlib.sha256(validation.read_bytes()).hexdigest()):
                raise ValueError("Frozen selection inputs changed; use a separate result directory.")
            layers = load_model(source)
        else:
            layers, _, report = run(source, directory, validation, expected_reads=2)
        if report["selected"] is None:
            raise ValueError(f"H={h}: no threshold meets the recorded capacity/MSE criteria.")
        sparse = prune(layers, report["selected"]["threshold"])
        metrics = {"hidden_size": h, "selection_status": report["selection_status"],
                   "threshold": report["selected"]["threshold"],
                   "dense_bytes": report["selected"]["dense_bytes"],
                   "csr_bytes": report["selected"]["csr_bytes"],
                   "nonzero_weights": report["selected"]["nonzero"],
                   "validation_dense_mse": report["dense_validation_mse"],
                   "validation_csr_mse": report["selected"]["validation_mse"],
                   "validation_change_percent": report["selected"]["mse_increase_pct"],
                   "note": "Saved-input PC results; validation selected the threshold; no independent final evaluation or hardware timing yet."}
        for model_name, model in [("dense", layers), ("csr", sparse)]:
            prediction = predict_rgb8(model, x).astype(np.float64)
            for chart, mask in [("3x3", slice(0, 9)), ("6x4", slice(9, 33)), ("all33", slice(None))]:
                metrics[f"learning_{model_name}_mse_{chart}"] = float(np.mean((prediction[mask] - target[mask]) ** 2))
        (directory / "comparison.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, nargs="+", default=[40, 60, 100])
    prepare(parser.parse_args().hidden)
