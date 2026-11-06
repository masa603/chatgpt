"""Build new candidate firmware without changing original measurement artifacts."""

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiment_today"
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE), str(ROOT / "pruning")]
from export_experiments import export_one
from build_fresh_scanner import build
from threshold_tools import load_model, predict_rgb8, prune


def main():
    with (ROOT / "retraining/data/current33.csv").open(newline="") as f:
        records = list(csv.DictReader(f))
    x = np.array([[r[f"input_{c}"] for c in "rgb"] for r in records], np.float32)
    y = np.array([[r[f"target_{c}8"] for c in "rgb"] for r in records], float)
    for h in (40, 60, 100):
        selection = json.loads((HERE / f"selection/h{h}/summary.json").read_text())
        dense = load_model(ROOT / f"retraining/results/h{h}/model_parameters.h")
        sparse = prune(dense, selection["selected"]["threshold"])
        comparison = {"hidden_size": h, "selection_status": selection["selection_status"],
                      "threshold": selection["selected"]["threshold"],
                      "dense_bytes": selection["selected"]["dense_bytes"], "csr_bytes": selection["selected"]["csr_bytes"],
                      "validation_dense_mse": selection["dense_validation_mse"],
                      "validation_csr_mse": selection["selected"]["validation_mse"],
                      "role": "Exploratory precision follow-up. Not original hardware results."}
        for name, model in [("dense", dense), ("csr", sparse)]:
            prediction = predict_rgb8(model, x).astype(float)
            for chart, mask in [("3x3", slice(0,9)), ("6x4", slice(9,33)), ("all33", slice(None))]:
                comparison[f"learning_{name}_mse_{chart}"] = float(np.mean((prediction[mask] - y[mask]) ** 2))
        (HERE / f"selection/h{h}/comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        target = HERE / f"compare_h{h}"
        export_one(h, selection_root=HERE / "selection", output_dir=target, stage_label="EXPLORATORY_PRECISION_FOLLOW_UP")
        (target / "parameters.h").rename(target / "parameters_precision.h")
        ino = target / f"compare_h{h}.ino"
        ino.write_text(ino.read_text().replace('#include "parameters.h"', '#include "parameters_precision.h"'))
    target = HERE / "fresh_scan_all"
    build(selection_root=HERE / "selection", target=target, stage_label="EXPLORATORY_PRECISION_FOLLOW_UP")
    (target / "models_all.h").rename(target / "models_precision.h")
    ino = target / "fresh_scan_all.ino"
    ino.write_text(ino.read_text().replace('#include "models_all.h"', '#include "models_precision.h"'))


if __name__ == "__main__":
    main()
