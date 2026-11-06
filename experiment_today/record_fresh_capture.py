"""Verify fresh physical readings, archive six PPMs, and summarize actual captures."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pruning"))
from export_csr_arduino import dense_layer_references
from threshold_tools import load_model, prune


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def record(path, received_date, aggregate=True):
    path = Path(path).resolve()
    directory = path.parent
    text = path.read_text()
    lines = text.splitlines()
    assert text.count("========== FRESH CAPTURE START ==========") == 1
    assert text.count("=========== FRESH CAPTURE END ===========") == 1
    assert "INPUT_SOURCE=NEW_PHYSICAL_SENSOR_READINGS" in lines
    assert "CHART_ROLE=LEARNING_3X3_REPEAT; NOT_INDEPENDENT_FINAL_EVALUATION" in lines
    assert "MODEL_GENERATION=CURRENT33_SEED42_EPOCH50000" in lines
    assert "# size,3,3" in lines
    with (ROOT / "retraining/data/current33.csv").open(newline="") as f:
        learning = list(csv.DictReader(f))[:9]
    targets = np.array([[r[f"target_{c}8"] for c in "rgb"] for r in learning], np.int32)
    ids = [f"r{r}c{c}" for r in range(1, 4) for c in range(1, 4)]
    start = lines.index("color_id,input_r,input_g,input_b")
    normalized = list(csv.DictReader(io.StringIO("\n".join(lines[start:start + 10]))))
    start = lines.index("color_id,sample_completed_ms,raw_r,raw_g,raw_b")
    raw = list(csv.DictReader(io.StringIO("\n".join(lines[start:start + 10]))))
    assert [r["color_id"] for r in raw] == ids == [r["color_id"] for r in normalized]
    raw_values = np.array([[r[f"raw_{c}"] for c in "rgb"] for r in raw], np.float32)
    logged_x = np.array([[r[f"input_{c}"] for c in "rgb"] for r in normalized], np.float32)
    black = np.array(next(line for line in lines if line.startswith("# black_rgb,")).split(",")[1:], np.float32)
    white = np.array(next(line for line in lines if line.startswith("# white_rgb,")).split(",")[1:], np.float32)
    assert np.all(np.isfinite(raw_values)) and np.all(np.isfinite(logged_x)) and np.all(white > black)
    assert np.all((logged_x >= 0) & (logged_x <= 1))
    reconstructed = np.clip((raw_values - black) / (white - black), 0, 1)
    assert np.allclose(reconstructed, logged_x, atol=1e-7, rtol=1e-6)
    timestamps = [int(r["sample_completed_ms"]) for r in raw]
    assert timestamps == sorted(timestamps)
    pattern = r"H=(\d+) MODE=(DENSE|CSR)\nMODEL_SOURCE_SHA256=([^\n]+)\nTHRESHOLD=([^\n]+)\n---------- PPM START ----------\n(.*?)----------- PPM END -----------"
    images = re.findall(pattern, text, re.S)
    assert [(int(h), mode) for h, mode, *_ in images] == [(h, mode) for h in (40, 60, 100) for mode in ("DENSE", "CSR")]
    start = lines.index("fresh_mse_dataset,H,format,colors,mse")
    logged_mses = list(csv.DictReader(io.StringIO("\n".join(lines[start:start + 7]))))
    results, patch_errors = [], []
    for (h_text, mode, model_hash, threshold_text, ppm), reported in zip(images, logged_mses):
        h = int(h_text)
        selection = json.loads((HERE / f"selection/h{h}/summary.json").read_text())
        assert model_hash == selection["model_sha256"]
        expected_threshold = 0 if mode == "DENSE" else selection["selected"]["threshold"]
        assert float(threshold_text) == expected_threshold
        tokens = ppm.split()
        assert tokens[:4] == ["P3", "3", "3", "255"] and len(tokens) == 31
        pixels = np.array(tokens[4:], np.int32).reshape(9, 3)
        assert np.all((pixels >= 0) & (pixels <= 255))
        mse = float(np.mean((pixels - targets) ** 2))
        assert (int(reported["H"]), reported["format"], int(reported["colors"])) == (h, mode, 9)
        assert abs(float(reported["mse"]) - mse) < 1e-6
        dense = load_model(ROOT / f"retraining/results/h{h}/model_parameters.h")
        model = dense if mode == "DENSE" else prune(dense, expected_threshold)
        expected_raw = dense_layer_references(model, reconstructed)[-1]
        expected_pixels = np.floor(np.clip(expected_raw.astype(np.float64), 0, 1) * 255 + .5).astype(np.int32)
        max_difference = int(np.max(np.abs(pixels - expected_pixels)))
        assert max_difference <= 1, f"H={h} {mode}: unexpected inference mismatch {max_difference}"
        image_name = f"h{h}_{mode.lower()}.ppm"
        (directory / image_name).write_text(ppm)
        results.append({"capture_key": directory.name, "hidden_size": h, "format": mode, "colors": 9,
                        "mse_rgb8": mse, "model_sha256": model_hash, "threshold": expected_threshold,
                        "pc_replay_max_rgb8_difference": max_difference, "ppm": image_name})
        for i, (prediction, target) in enumerate(zip(pixels, targets)):
            patch_errors.append({"hidden_size": h, "format": mode, "color_id": ids[i],
                                 **{f"predicted_{c}8": int(prediction[j]) for j, c in enumerate("rgb")},
                                 **{f"target_{c}8": int(target[j]) for j, c in enumerate("rgb")},
                                 "mse_rgb8": float(np.mean((prediction - target) ** 2))})
    combined = [{**r, **{k: v for k, v in x.items() if k != "color_id"}} for r, x in zip(raw, normalized)]
    for i, row in enumerate(combined):
        row.update({f"target_{c}8": int(targets[i, j]) for j, c in enumerate("rgb")})
    write_csv(directory / "inputs.csv", combined)
    write_csv(directory / "mse.csv", results)
    write_csv(directory / "patch_errors.csv", patch_errors)
    comparison = []
    for h in (40, 60, 100):
        dense, csr = [r["mse_rgb8"] for r in results if r["hidden_size"] == h]
        comparison.append({"hidden_size": h, "dense_mse": dense, "csr_mse": csr,
                           "csr_minus_dense_mse": csr - dense, "csr_change_percent": (csr / dense - 1) * 100 if dense else None})
    summary = {"received_date_japan": received_date, "execution_timestamp": None,
               "capture_counter": int(next(line for line in lines if line.startswith("CAPTURE=")).split("=")[1]),
               "capture_key": directory.name, "input_source": "NEW_PHYSICAL_SENSOR_READINGS",
               "chart_role": "Learning 3x3 chart re-read; same six models per physical capture.",
               "independent_final_evaluation": "NOT_PERFORMED", "startup_check_log": "NOT_SUPPLIED",
               "black_rgb": black.tolist(), "white_rgb": white.tolist(), "colors": 9,
               "mse_from_ppm_check": "PASS for all six images", "raw_to_normalized_input_check": "PASS",
               "pc_model_replay_check": "RGB8 agrees within at most one quantization unit for all six models; see per-model differences.",
               "normalization_max_abs_difference": float(np.max(np.abs(reconstructed - logged_x))),
               "results": results, "paired_comparisons": comparison,
               "target_source": "Recovered original Notebook display; original PNG not verified.",
               "reference_sha256": hashlib.sha256((ROOT / "retraining/data/current33.csv").read_bytes()).hexdigest(),
               "raw_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if aggregate:
        update_totals(directory.parent)
    print("PASS: all six PPM MSEs, model hashes/thresholds, raw/calibration normalization and fresh-input model replay.")
    print(json.dumps(comparison))


def update_totals(parent):
    summaries = [json.loads(path.read_text()) for path in sorted(parent.glob("capture_*/summary.json"))]
    all_results = [row for summary in summaries for row in summary["results"]]
    write_csv(parent / "mse_results.csv", all_results)
    aggregate = []
    for h in (40, 60, 100):
        for mode in ("DENSE", "CSR"):
            values = [r["mse_rgb8"] for r in all_results if r["hidden_size"] == h and r["format"] == mode]
            aggregate.append({"hidden_size": h, "format": mode, "physical_captures": len(values),
                              "chart_role": "learning_3x3_repeat", "mean_mse": statistics.mean(values),
                              "sample_stddev_mse": statistics.stdev(values) if len(values) > 1 else "",
                              "min_mse": min(values), "max_mse": max(values)})
    write_csv(parent / "mse_summary.csv", aggregate)
    pairs = [{"capture_key": s["capture_key"], **row} for s in summaries for row in s["paired_comparisons"]]
    write_csv(parent / "paired_comparisons.csv", pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--received-date", required=True)
    parser.add_argument("--no-aggregate", action="store_true", help="Archive an out-of-schedule log separately from planned repetitions.")
    args = parser.parse_args()
    record(args.log, args.received_date, aggregate=not args.no_aggregate)
