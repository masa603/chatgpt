"""Exploratory pruning selection preserving the frozen Dense mapping."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiment_today"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pruning"))
from export_csr_arduino import dense_layer_references
from threshold_tools import load_model, predict_rgb8, prune, save_csr_npz


def prepare():
    policy_hash = hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest()
    data = ROOT / "retraining/data/current33.csv"
    with data.open(newline="") as f:
        rows = list(csv.DictReader(f))
    training_x = np.array([[r[f"input_{c}"] for c in "rgb"] for r in rows], np.float32)
    training_y = np.array([[r[f"target_{c}8"] for c in "rgb"] for r in rows], np.float64)
    axis = np.linspace(0, 1, 9, dtype=np.float32)
    grid = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T.copy()
    for h in (40, 60, 100):
        start = time.monotonic()
        output = HERE / f"selection/h{h}"
        output.mkdir(parents=True, exist_ok=True)
        if (output / "summary.json").exists():
            raise FileExistsError("Follow-up selection is already frozen; do not overwrite it.")
        source = ROOT / f"retraining/results/h{h}/model_parameters.h"
        original = json.loads((BASE / f"selection/h{h}/summary.json").read_text())
        assert hashlib.sha256(source.read_bytes()).hexdigest() == original["model_sha256"]
        assert hashlib.sha256((ROOT / "pruning/validation.csv").read_bytes()).hexdigest() == original["validation_csv_sha256"]
        dense = load_model(source)
        dense_training_mse = float(np.mean((predict_rgb8(dense, training_x).astype(float) - training_y) ** 2))
        dense_grid = np.clip(dense_layer_references(dense, grid)[-1], 0, 1)
        with (BASE / f"selection/h{h}/candidates.csv").open(newline="") as f:
            candidates = list(csv.DictReader(f))
        candidates.sort(key=lambda r: (int(r["csr_bytes"]), float(r["validation_mse"]), float(r["threshold"])))
        checked = []
        selected = None
        for index, candidate in enumerate(candidates):
            record = {"threshold": float(candidate["threshold"]), "nonzero": int(candidate["nonzero"]),
                      "csr_bytes": int(candidate["csr_bytes"]), "dense_bytes": int(candidate["dense_bytes"]),
                      "validation_mse": float(candidate["validation_mse"]),
                      "training_mse": None, "grid_max_abs_clipped_difference": None, "result": ""}
            checked.append(record)
            if record["csr_bytes"] >= record["dense_bytes"]:
                record["result"] = "capacity_not_smaller"
                continue
            per_read = [float(candidate[f"read_{i}_mse"]) for i in (1, 2)]
            if any(new > old * 1.05 for new, old in zip(per_read, original["dense_mse_by_read"])):
                record["result"] = "validation_read_limit"
                continue
            sparse = prune(dense, record["threshold"])
            training_mse = float(np.mean((predict_rgb8(sparse, training_x).astype(float) - training_y) ** 2))
            record["training_mse"] = training_mse
            if training_mse > dense_training_mse * 1.05:
                record["result"] = "training_fidelity_limit"
                continue
            sparse_grid = np.clip(dense_layer_references(sparse, grid)[-1], 0, 1)
            max_difference = float(np.max(np.abs(sparse_grid - dense_grid)))
            record["grid_max_abs_clipped_difference"] = max_difference
            if max_difference > 0.5 / 255:
                record["result"] = "grid_fidelity_limit"
                continue
            record["result"] = "selected"
            selected = {key: (value == "True" if value in ("True", "False") else None if value == "" else float(value))
                        for key, value in candidate.items()}
            for key in ("csr_bytes", "dense_bytes", "nonzero"):
                selected[key] = int(selected[key])
            selected.update(training_mse=training_mse, grid_max_abs_clipped_difference=max_difference)
            assert save_csr_npz(sparse, output / "selected_csr.npz") == selected["csr_bytes"]
            break
        report = {**original, "selected": selected, "follow_up_stage": "EXPLORATORY_PRECISION_FOLLOW_UP",
                  "selection_status": "no_eligible_candidate" if selected is None else "provisional_input_collisions",
                  "selection_policy_sha256": policy_hash,
                  "learning_data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                  "selection_inputs": "Original 33 learning inputs; original 2 validation reads; fixed unlabeled 729-point grid. No fresh captures used for filtering/ranking.",
                  "dense_learning_all33_mse": dense_training_mse, "grid_points": len(grid),
                  "grid_limit_raw_normalized": 0.5 / 255,
                  "candidates_checked_in_storage_ascending_order": len(checked),
                  "selection_explanation": "First passing candidate in increasing byte order. Smaller candidates were checked and rejected. Does not assume error is monotonic.",
                  "wall_seconds": time.monotonic() - start,
                  "note": "Exploratory method change after observing original results. Existing fresh captures are not an independent final test. No new hardware timing or accuracy result yet."}
        with (output / "checked_candidates.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(checked[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(checked)
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"h": h, "selected": selected, "checked": len(checked), "seconds": report["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    prepare()
