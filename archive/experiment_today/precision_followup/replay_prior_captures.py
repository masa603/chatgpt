"""Post-hoc PC replay after all follow-up thresholds are frozen; not a new measurement."""

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiment_today"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pruning"))
from threshold_tools import load_model, prune
from export_csr_arduino import dense_layer_references


def main():
    reports = {h: json.loads((HERE / f"selection/h{h}/summary.json").read_text()) for h in (40,60,100)}
    assert all(r["selected"] for r in reports.values())
    rows = []
    for key in ("capture_01", "capture_03"):
        directory = BASE / "fresh_readings" / key
        summary = json.loads((directory / "summary.json").read_text())
        with (directory / "inputs.csv").open(newline="") as f:
            data = list(csv.DictReader(f))
        raw = np.array([[r[f"raw_{c}"] for c in "rgb"] for r in data], np.float32)
        black, white = np.array(summary["black_rgb"], np.float32), np.array(summary["white_rgb"], np.float32)
        inputs = np.clip((raw - black) / (white - black), 0, 1)
        targets = np.array([[r[f"target_{c}8"] for c in "rgb"] for r in data], float)
        for h in (40,60,100):
            dense = load_model(ROOT / f"retraining/results/h{h}/model_parameters.h")
            sparse = prune(dense, reports[h]["selected"]["threshold"])
            old = next(r["mse_rgb8"] for r in summary["results"] if r["hidden_size"] == h and r["format"] == "CSR")
            baseline = next(r["mse_rgb8"] for r in summary["results"] if r["hidden_size"] == h and r["format"] == "DENSE")
            raw_output = dense_layer_references(sparse, inputs)[-1]
            prediction = np.floor(np.clip(raw_output.astype(float), 0, 1) * 255 + .5)
            mse = float(np.mean((prediction - targets) ** 2))
            rows.append({"capture_key": key, "hidden_size": h, "dense_device_mse": baseline,
                         "original_csr_device_mse": old, "followup_csr_pc_replay_mse": mse,
                         "followup_change_vs_dense_percent": (mse / baseline - 1) * 100,
                         "role": "POST_HOC_PC_REPLAY_NOT_NEW_DEVICE_MEASUREMENT"})
    with (HERE / "prior_capture_diagnostic.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print("Post-hoc replay saved; no new device measurement or independent final evaluation.")


if __name__ == "__main__":
    main()
