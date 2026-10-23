"""Magnitude-pruning candidates and validation for the experiment plan."""

from pathlib import Path
import csv
import hashlib
import json
import re

import numpy as np


VALIDATION_COLUMNS = [
    "read_id", "color_id", "input_r", "input_g", "input_b",
    "target_r8", "target_g8", "target_b8",
]


def load_model(path):
    """Read the original row-major FP32 Arduino arrays without retraining."""
    source = Path(path).read_text(encoding="utf-8")
    pattern = r"(?:const\s+)?float\s+((?:weight|bias)_[123])\s*\[\s*\d*\s*\]\s*=\s*\{([^}]+)\}"
    arrays = {}
    for name, values in re.findall(pattern, source):
        if name in arrays:
            raise ValueError(f"Duplicate array: {name}")
        tokens = [v.strip() for v in values.split(",") if v.strip()]
        arrays[name] = np.array(
            [float(v.rstrip("fF")) for v in tokens], dtype=np.float32
        )
    expected = {f"{kind}_{i}" for kind in ("weight", "bias") for i in (1, 2, 3)}
    if set(arrays) != expected:
        raise ValueError("weight_1..3 and bias_1..3 are required.")
    h = len(arrays["bias_1"])
    if not 1 <= h <= 200:
        raise ValueError("The CSR index types support H=1..200 in this tool.")
    shapes = [(h, 3), (h, h), (3, h)]
    layers = []
    for i, shape in enumerate(shapes, start=1):
        w, b = arrays[f"weight_{i}"], arrays[f"bias_{i}"]
        if w.size != shape[0] * shape[1] or b.size != shape[0]:
            raise ValueError(f"Unexpected layer {i} shape; expected {shape}.")
        if not np.isfinite(w).all() or not np.isfinite(b).all():
            raise ValueError("Model values must be finite.")
        layers.append((w.reshape(shape), b.copy()))
    return layers


def thresholds_from_weights(layers):
    """Zero baseline plus a midpoint of every distinct adjacent magnitude."""
    magnitudes = np.unique(np.concatenate([
        np.abs(w).ravel().astype(np.float64) for w, _ in layers
    ]))
    # Use float64 midpoints so adjacent float32 weights remain distinguishable.
    midpoints = magnitudes[:-1] + np.diff(magnitudes) / 2.0
    return np.concatenate(([0.0], midpoints[midpoints > 0]))


def prune(layers, threshold):
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative.")
    return [
        (np.where(np.abs(w).astype(np.float64) < threshold, 0, w).astype(np.float32), b.copy())
        for w, b in layers
    ]


def storage(layers):
    """CSR: float32 values/bias, uint8 columns, uint16 row pointers."""
    nonzero = sum(int(np.count_nonzero(w)) for w, _ in layers)
    total = sum(w.size for w, _ in layers)
    dense = sum(w.nbytes + b.nbytes for w, b in layers)
    csr = sum(5 * int(np.count_nonzero(w)) + 2 * (w.shape[0] + 1) + b.nbytes
              for w, b in layers)
    return {
        "nonzero": nonzero,
        "sparsity_pct": (1 - nonzero / total) * 100,
        "csr_bytes": int(csr),
        "dense_bytes": int(dense),
        "saving_pct": (1 - csr / dense) * 100,
    }


def predict_rgb8(layers, inputs):
    """Bias first, sequential FP32 sums, hidden ReLU, Arduino half-up output."""
    x = np.asarray(inputs, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("Inputs must be a finite (N, 3) array.")
    for i, (w, b) in enumerate(layers):
        output = np.broadcast_to(b, (len(x), len(b))).copy()
        for column in range(w.shape[1]):
            output += x[:, column:column + 1] * w[:, column][None, :]
        x = np.maximum(output, np.float32(0)) if i < 2 else output
    return np.floor(np.clip(x, 0, 1) * np.float32(255) + np.float32(0.5)).astype(np.uint8)


def load_validation(path):
    """Three independent readings of the same validation chart, with RGB8 truth."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != VALIDATION_COLUMNS:
            raise ValueError(f"CSV columns must be: {','.join(VALIDATION_COLUMNS)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Validation CSV has a header but no measured rows.")
    keys = [(r["read_id"].strip(), r["color_id"].strip()) for r in rows]
    if any(not read or not color for read, color in keys) or len(set(keys)) != len(keys):
        raise ValueError("Each reading/color pair must be present exactly once.")
    groups = sorted({read for read, _ in keys})
    if len(groups) != 3:
        raise ValueError("The plan requires three separately measured validation readings.")
    colors = [{color for read, color in keys if read == group} for group in groups]
    if any(c != colors[0] for c in colors[1:]):
        raise ValueError("All three readings must contain the same validation colors.")
    x = np.array([[r[f"input_{c}"] for c in "rgb"] for r in rows], dtype=np.float32)
    y = np.array([[r[f"target_{c}8"] for c in "rgb"] for r in rows], dtype=np.float64)
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("NN input RGB must already be normalized to 0..1.")
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 255)) or np.any(y != np.floor(y)):
        raise ValueError("Target RGB must be integer values from the reference image, 0..255.")
    targets = {}
    for (_, color), target in zip(keys, y):
        if color in targets and not np.array_equal(targets[color], target):
            raise ValueError("Reference RGB for a color must be identical across readings.")
        targets[color] = target
    masks = [np.array([read == group for read, _ in keys]) for group in groups]
    # Record collisions without claiming that matching numbers prove reused readings.
    collisions = []
    for group, mask in zip(groups, masks):
        indices = np.flatnonzero(mask)
        for offset, a in enumerate(indices):
            for b in indices[offset + 1:]:
                if np.array_equal(x[a], x[b]) and not np.array_equal(y[a], y[b]):
                    collisions.append({"read_id": group, "color_ids": [keys[a][1], keys[b][1]]})
    return {"inputs": x, "targets": y, "masks": masks, "read_ids": groups,
            "colors_per_read": len(colors[0]), "input_collisions": collisions}


def validation_mses(layers, validation):
    prediction = predict_rgb8(layers, validation["inputs"]).astype(np.float64)
    squared = (prediction - validation["targets"]) ** 2
    return [float(np.mean(squared[mask])) for mask in validation["masks"]]


def choose_threshold(records):
    eligible = [r for r in records if r.get("eligible") is True]
    return min(eligible, key=lambda r: (r["csr_bytes"], r["validation_mse"], r["threshold"])) if eligible else None


def analyze(layers, validation=None):
    baseline = validation_mses(layers, validation) if validation is not None else None
    dense_mse = float(np.mean(baseline)) if baseline is not None else None
    records = []
    for threshold in thresholds_from_weights(layers):
        sparse = prune(layers, float(threshold))
        row = {"threshold": float(threshold), **storage(sparse)}
        row["memory_ok"] = row["csr_bytes"] < row["dense_bytes"]
        row.update(validation_mse=None, mse_difference=None, mse_increase_pct=None,
                   quality_ok=None, eligible=None)
        if validation is not None:
            mses = validation_mses(sparse, validation)
            mse = float(np.mean(mses))
            quality_ok = mse == 0 if dense_mse == 0 else mse <= dense_mse * 1.05
            row.update(validation_mse=mse, mse_difference=mse - dense_mse,
                       mse_increase_pct=None if dense_mse == 0 else (mse / dense_mse - 1) * 100,
                       quality_ok=quality_ok, eligible=quality_ok and row["memory_ok"])
            for i, value in enumerate(mses, start=1):
                row[f"read_{i}_mse"] = value
        records.append(row)
    return records, {"dense_validation_mse": dense_mse, "dense_mse_by_read": baseline}


def save_csr_npz(layers, path):
    arrays = {}
    for i, (w, b) in enumerate(layers, start=1):
        rows, columns = np.nonzero(w)
        row_ptr = np.concatenate(([0], np.cumsum(np.count_nonzero(w, axis=1))))
        if len(rows) > np.iinfo(np.uint16).max or w.shape[1] > 256:
            raise ValueError("CSR index overflow.")
        arrays[f"values_{i}"] = w[rows, columns].astype(np.float32)
        arrays[f"col_idx_{i}"] = columns.astype(np.uint8)
        arrays[f"row_ptr_{i}"] = row_ptr.astype(np.uint16)
        arrays[f"bias_{i}"] = b.copy()
        restored = np.zeros_like(w)
        for row in range(len(b)):
            start, end = row_ptr[row:row + 2]
            restored[row, columns[start:end]] = w[rows[start:end], columns[start:end]]
        if not np.array_equal(restored, w):
            raise AssertionError("CSR reconstruction failed.")
    np.savez(path, **arrays)
    return sum(v.nbytes for v in arrays.values())


def run(header_path, output_dir, validation_path=None):
    header_path, output_dir = Path(header_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = load_model(header_path)
    validation = load_validation(validation_path) if validation_path is not None else None
    records, baseline = analyze(layers, validation)
    with (output_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    selected = choose_threshold(records)
    # A later run without data must not leave an older selection looking current.
    selected_path = output_dir / "selected_csr.npz"
    if selected is not None:
        byte_count = save_csr_npz(prune(layers, selected["threshold"]), selected_path)
        if byte_count != selected["csr_bytes"]:
            raise AssertionError("CSR byte count does not match exported array payloads.")
    elif selected_path.exists():
        selected_path.unlink()
    report = {
        "model_sha256": hashlib.sha256(header_path.read_bytes()).hexdigest(),
        "hidden_size": len(layers[0][1]),
        "weight_count": sum(w.size for w, _ in layers),
        "candidate_count_including_zero_baseline": len(records),
        "candidate_rule": "Zero baseline plus all midpoints between distinct adjacent absolute weights; one global threshold; biases unchanged.",
        "capacity_scope": "FP32 values/bias + uint8 columns + uint16 row pointers; excludes whole-build alignment, buffers, NPZ container overhead.",
        "validation_status": "not_supplied" if validation is None else "evaluated",
        "validation_csv_sha256": None if validation is None else hashlib.sha256(Path(validation_path).read_bytes()).hexdigest(),
        "validation_read_ids": None if validation is None else validation["read_ids"],
        "validation_colors_per_read": None if validation is None else validation["colors_per_read"],
        "input_collisions": None if validation is None else validation["input_collisions"],
        "mse_scale": "RGB 0..255 after clipping and Arduino half-up rounding; equal mean of three reading MSEs.",
        **baseline,
        "selected": selected,
        "diagnostic_only": None if validation is None or selected is not None else min(records, key=lambda r: (r["csr_bytes"], r["validation_mse"], r["threshold"])),
        "note": "No threshold is selected without independent validation. Training/final-evaluation data must not be supplied as validation. Training data and random seed are not encoded in the header; record model provenance separately. Re-run for each frozen trained model.",
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return layers, records, report


def plot_results(layers, records, report, output_dir):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    absolute = np.concatenate([np.abs(w).ravel().astype(np.float64) for w, _ in layers])
    positive = absolute[absolute > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    if positive.size:
        bins = 10 ** np.histogram_bin_edges(np.log10(positive), bins="auto")
        axes[0].hist(positive, bins=bins)
        axes[0].set_xscale("log")
    axes[0].set(xlabel="Absolute weight", ylabel="Count", title="Three weight matrices (bias excluded)")
    sorted_weights = np.sort(absolute)
    axes[1].step(sorted_weights, np.arange(1, len(absolute) + 1) / len(absolute), where="post")
    if positive.size:
        axes[1].set_xscale("log")
    axes[1].set(xlabel="Absolute weight", ylabel="Cumulative fraction", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(output_dir / "weight_distribution.png", dpi=160)

    nonzero_thresholds = [r for r in records if r["threshold"] > 0]
    fig_capacity, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot([r["threshold"] for r in nonzero_thresholds], [r["csr_bytes"] for r in nonzero_thresholds], label="CSR arrays")
    ax.axhline(records[0]["dense_bytes"], color="gray", linestyle="--", label="Dense FP32 arrays")
    if nonzero_thresholds:
        ax.set_xscale("log")
    ax.set(xlabel="Pruning threshold", ylabel="Model array bytes", title="Capacity only; quality requires validation")
    ax.legend()
    fig_capacity.tight_layout()
    fig_capacity.savefig(output_dir / "capacity.png", dpi=160)

    figures = [fig, fig_capacity]
    tradeoff_path = output_dir / "validation_tradeoff.png"
    if report["validation_status"] == "evaluated":
        fig_tradeoff, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.scatter([r["csr_bytes"] for r in records], [r["validation_mse"] for r in records], s=8, label="Evaluated thresholds")
        ax.axhline(report["dense_validation_mse"] * 1.05, color="gray", linestyle="--", label="Dense MSE x 1.05")
        if report["selected"] is not None:
            chosen = report["selected"]
            ax.scatter([chosen["csr_bytes"]], [chosen["validation_mse"]], color="red", s=65, label="Selected")
        ax.set(xlabel="CSR model array bytes", ylabel="Mean validation MSE (RGB8)")
        ax.legend()
        fig_tradeoff.tight_layout()
        fig_tradeoff.savefig(tradeoff_path, dpi=160)
        figures.append(fig_tradeoff)
    elif tradeoff_path.exists():
        tradeoff_path.unlink()
    return figures
