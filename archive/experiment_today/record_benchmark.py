"""Check a supplied Arduino log, recompute statistics, and append actual results."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent


def record(path, received_date):
    path = Path(path).resolve()
    text = path.read_text()
    lines = text.splitlines()

    def value(key):
        return next(line[len(key) + 1:] for line in lines if line.startswith(key + "="))

    h = int(value("H"))
    selection = json.loads((HERE / f"selection/h{h}/summary.json").read_text())
    comparison = json.loads((HERE / f"selection/h{h}/comparison.json").read_text())
    assert value("MODEL_SOURCE_SHA256") == selection["model_sha256"]
    assert value("INPUT_SHA256") == selection["validation_csv_sha256"]
    assert float(value("THRESHOLD")) == selection["selected"]["threshold"]
    assert (int(value("INPUT_COUNT")), int(value("WARMUP_CALLS")), int(value("CALLS_PER_GROUP")), int(value("GROUPS"))) == (128, 10, 100, 5)
    assert value("MSE_SOURCE") == "SAVED_INPUTS_REPLAYED_ON_DEVICE"
    assert value("INDEPENDENT_FINAL_EVALUATION") == "NOT_PERFORMED"
    assert all(s in lines for s in ["CSR INDEX CHECK: PASS", "FIXED INPUT CHECK: Dense=PASS CSR=PASS", "TIMING VALID: YES"])
    start = next(i for i, line in enumerate(lines) if line.startswith("group,first_model,"))
    timings = list(csv.DictReader(io.StringIO("\n".join(lines[start:start + 6]))))
    stats = {}
    for i, row in enumerate(timings, 1):
        assert int(row["group"]) == i and row["first_model"] == ("Dense" if i % 2 else "CSR")
        for mode in ("dense", "csr"):
            assert int(row[f"{mode}_total_us"]) > 0
            assert abs(int(row[f"{mode}_total_us"]) / 100 - float(row[f"{mode}_us_per_inference"])) < .000051
            assert math.isfinite(float(row[f"{mode}_checksum"]))
    for mode, label in [("dense", "DENSE"), ("csr", "CSR")]:
        values = [int(row[f"{mode}_total_us"]) / 100 for row in timings]
        mean, sd = statistics.mean(values), statistics.stdev(values)
        match = re.search(rf"^{label} mean_us=([\d.]+) sample_stddev_us=([\d.]+)$", text, re.M)
        assert abs(float(match[1]) - mean) < .000051 and abs(float(match[2]) - sd) < .000051
        stats[mode] = {"mean_us": mean, "sample_stddev_us": sd}
    change = (stats["csr"]["mean_us"] / stats["dense"]["mean_us"] - 1) * 100
    assert abs(change - float(value("CSR time_change_percent"))) < .000051
    start = lines.index("mse_dataset,read_id,colors,dense_mse,csr_mse")
    mses = list(csv.DictReader(io.StringIO("\n".join(lines[start:start + 7]))))
    for mode in ("dense", "csr"):
        for row in mses:
            dataset = row["mse_dataset"]
            key = f"learning_{mode}_mse_" + dataset.removeprefix("learning_") if dataset.startswith("learning_") else f"validation_{mode}_mse" if dataset == "validation_mean" else None
            if key:
                assert abs(float(row[f"{mode}_mse"]) - comparison[key]) < 1e-6
            else:
                expected = selection["dense_mse_by_read"][int(row["read_id"]) - 1] if mode == "dense" else selection["selected"][f"read_{row['read_id']}_mse"]
                assert abs(float(row[f"{mode}_mse"]) - expected) < 1e-6
    sizes = re.search(r"^MODEL BYTES: Dense=(\d+) CSR=(\d+)$", text, re.M)
    dense_bytes, csr_bytes = map(int, sizes.groups())
    assert (dense_bytes, csr_bytes) == (comparison["dense_bytes"], comparison["csr_bytes"])
    report = {"received_date_japan": received_date, "execution_timestamp": None,
              "source": "User-supplied serial output from Arduino UNO R4 WiFi", "hidden_size": h,
              "status": "DEVICE_CHECK_AND_TIMING_PASS", "model_sha256": value("MODEL_SOURCE_SHA256"),
              "input_sha256": value("INPUT_SHA256"), "threshold": float(value("THRESHOLD")),
              "compiler": value("COMPILER"), "f_cpu_hz": int(value("F_CPU_HZ")), "input_count": 128,
              "warmup_calls": 10, "calls_per_group": 100, "groups": 5,
              "dense_model_bytes": dense_bytes, "csr_model_bytes": csr_bytes,
              "model_bytes_reduction_percent": (1 - csr_bytes / dense_bytes) * 100, "timing": stats,
              "csr_time_change_percent": change, "dense_over_csr_speed_ratio": stats["dense"]["mean_us"] / stats["csr"]["mean_us"],
              "timing_groups": timings, "mse_rows": mses, "pc_mse_agreement": "All learning/validation rows agree within 1e-6",
              "paired_firmware_flash_bytes": None, "paired_firmware_static_ram_bytes": None,
              "independent_final_evaluation": "NOT_PERFORMED", "new_sensor_capture": False,
              "limitation": "Learning MSE worsens after pruning; threshold selection used validation mean MSE. Timing variation describes only five timer-quantized groups.",
              "raw_log": str(path.relative_to(HERE)), "raw_log_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    result_path = HERE / "device_results.csv"
    with result_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    row = next(r for r in rows if int(r["hidden_size"]) == h)
    row.update(status=report["status"], compiler=report["compiler"], source_log=report["raw_log"])
    for mode in ("dense", "csr"):
        row[f"{mode}_mean_us"] = f"{stats[mode]['mean_us']:.8f}"
        row[f"{mode}_sample_stddev_us"] = f"{stats[mode]['sample_stddev_us']:.8f}"
    with result_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated and recorded H={h}: bytes -{report['model_bytes_reduction_percent']:.4f}%, speed {report['dense_over_csr_speed_ratio']:.4f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--received-date", required=True)
    args = parser.parse_args()
    record(args.log, args.received_date)
