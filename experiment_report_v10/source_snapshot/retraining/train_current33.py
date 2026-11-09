"""Re-train Dense FP32 models on the newly measured 33 learning inputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import torch
from torch import nn


class ColorNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(3, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                   nn.Linear(hidden_size, 3))

    def forward(self, inputs):
        return self.model(inputs)


def load_training(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    if len(records) != 33 or [int(r["sample_id"]) for r in records] != list(range(1, 34)):
        raise ValueError("The current training dataset must contain ordered sample IDs 1..33.")
    x = np.array([[r[f"input_{c}"] for c in "rgb"] for r in records], dtype=np.float32)
    y8 = np.array([[r[f"target_{c}8"] for c in "rgb"] for r in records], dtype=np.float64)
    if not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("NN input RGB must already be normalized to 0..1.")
    if not np.isfinite(y8).all() or np.any((y8 < 0) | (y8 > 255)) or np.any(y8 != np.floor(y8)):
        raise ValueError("Reference RGB must be integer 0..255 values.")
    charts = [r["chart_id"] for r in records]
    if charts != ["train_3x3"] * 9 + ["train_6x4"] * 24:
        raise ValueError("Expected 9 row-major 3x3 samples then 24 row-major 6x4 samples.")
    return x, y8


def model_arrays(model):
    return [(layer.weight.detach().cpu().numpy().copy(), layer.bias.detach().cpu().numpy().copy())
            for layer in model.model if isinstance(layer, nn.Linear)]


def forward_fp32(layers, inputs):
    """Match bias-first, sequential FP32 accumulation in the Arduino sketch."""
    x = np.asarray(inputs, dtype=np.float32)
    for i, (weight, bias) in enumerate(layers):
        output = np.broadcast_to(bias, (len(x), len(bias))).copy()
        for column in range(weight.shape[1]):
            output += x[:, column:column + 1] * weight[:, column][None, :]
        x = np.maximum(output, np.float32(0)) if i < 2 else output
    return x


def rgb8(raw_output):
    return np.floor(np.clip(raw_output, 0, 1) * np.float32(255) + np.float32(.5)).astype(np.uint8)


def mse_rgb8(prediction, target):
    return float(np.mean((np.asarray(prediction, dtype=np.float64) - target) ** 2))


def float_literal(value):
    value = np.float32(value)
    text = format(float(value), ".9g")
    if "." not in text and "e" not in text.lower():
        text += ".0"
    if not np.isfinite(value) or np.float32(float(text)) != value:
        raise ValueError("Invalid FP32 parameter or failed literal round trip.")
    return text + "f"


def export_header(layers, hidden_size, dataset_hash, seed, epochs, path):
    lines = ["// Dense FP32, re-trained on the current 33 saved learning inputs.", "#pragma once",
             'static_assert(sizeof(float) == 4, "FP32 required");',
             f"constexpr int MODEL_HIDDEN_SIZE = {hidden_size};",
             f"constexpr int MODEL_TRAINING_SEED = {seed};",
             f"constexpr int MODEL_TRAINING_EPOCHS = {epochs};",
             f'const char MODEL_TRAINING_DATA_SHA256[] = "{dataset_hash}";']
    for layer, (weight, bias) in enumerate(layers, 1):
        for name, values in [(f"weight_{layer}", weight.ravel()), (f"bias_{layer}", bias)]:
            lines.append(f"const float {name}[] = {{")
            for start in range(0, len(values), 8):
                lines.append("  " + ", ".join(float_literal(v) for v in values[start:start + 8]) + ",")
            lines.append("};")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def input_collision_bounds(x, y8):
    groups = {}
    for i, values in enumerate(x):
        groups.setdefault(tuple(values.tolist()), []).append(i)
    continuous = 0.0
    integer = 0.0
    collisions = []
    for indices in groups.values():
        if len(indices) > 1:
            targets = y8[indices]
            mean = targets.mean(axis=0)
            continuous += float(np.sum((targets - mean) ** 2))
            integer += float(np.sum((targets - np.floor(mean + .5)) ** 2))
            collisions.append({"sample_ids": [i + 1 for i in indices], "input_rgb": x[indices[0]].tolist(),
                               "target_rgb": targets.astype(int).tolist()})
    return {"groups": collisions, "minimum_continuous_rgb255_mse": continuous / y8.size,
            "minimum_rgb8_mse": integer / y8.size}


def evaluate_validation(layers, path):
    if path is None or not Path(path).exists():
        return None
    with Path(path).open(newline="") as f:
        records = list(csv.DictReader(f))
    x = np.array([[r[f"input_{c}"] for c in "rgb"] for r in records], dtype=np.float32)
    target = np.array([[r[f"target_{c}8"] for c in "rgb"] for r in records], dtype=np.float64)
    predicted = rgb8(forward_fp32(layers, x))
    readings = sorted({r["read_id"] for r in records})
    mses = [mse_rgb8(predicted[[r["read_id"] == read for r in records]],
                     target[[r["read_id"] == read for r in records]]) for read in readings]
    return {"role": "validation only; not used in training or checkpoint selection", "read_ids": readings,
            "mse_rgb8_by_read": mses, "mean_mse_rgb8": float(np.mean(mses)),
            "data_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def train_one(x, y8, hidden_size, output_dir, dataset_hash, epochs=50000, seed=42,
              learning_rate=.01, lambda_l1=1e-6, log_every=5000, validation_path=None):
    if hidden_size < 1 or epochs < 1:
        raise ValueError("Hidden size and epochs must be positive.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "model_parameters.h").exists():
        raise FileExistsError(f"Frozen weights already exist in {output_dir}; use a new output directory.")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = ColorNet(hidden_size).cpu().float()
    inputs = torch.from_numpy(np.asarray(x, dtype=np.float32))
    targets = torch.from_numpy((y8 / 255).astype(np.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    curve = []
    start = time.monotonic()

    def record(epoch):
        with torch.no_grad():
            mse = torch.mean((model(inputs) - targets) ** 2).item()
            l1 = sum(p.abs().sum().item() for name, p in model.named_parameters() if "weight" in name)
        curve.append({"epoch": epoch, "mse_normalized": mse, "l1_sum": l1, "loss": mse + lambda_l1 * l1})
        print(f"H={hidden_size} epoch={epoch}/{epochs} MSE(0..1)={mse:.8f}", flush=True)

    record(0)
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(inputs)
        mse = torch.mean((outputs - targets) ** 2)
        l1 = sum(p.abs().sum() for name, p in model.named_parameters() if "weight" in name)
        loss = mse + lambda_l1 * l1
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite loss; no model exported.")
        loss.backward()
        optimizer.step()
        if epoch % log_every == 0 or epoch == epochs:
            record(epoch)
    model.eval()
    layers = model_arrays(model)
    raw = forward_fp32(layers, x)
    predicted = rgb8(raw)
    with torch.no_grad():
        torch_output = model(inputs).numpy()
    report = {
        "hidden_size": hidden_size, "epochs": epochs, "seed_numpy_and_torch": seed,
        "learning_rate": learning_rate, "lambda_l1_weights_only": lambda_l1, "optimizer": "Adam",
        "checkpoint_rule": "Final epoch; no validation-driven early stopping or checkpoint selection.",
        "device": "cpu", "threads": 1, "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "python_version": platform.python_version(), "training_data_sha256": dataset_hash,
        "training_rows": 33, "all_rows_retained": True,
        "pc_training_mse_normalized_torch": curve[-1]["mse_normalized"],
        "pc_training_mse_rgb255_raw": float(np.mean((raw.astype(np.float64) * 255 - y8) ** 2)),
        "pc_training_mse_rgb8_all33": mse_rgb8(predicted, y8),
        "pc_training_mse_rgb8_3x3": mse_rgb8(predicted[:9], y8[:9]),
        "pc_training_mse_rgb8_6x4": mse_rgb8(predicted[9:], y8[9:]),
        "torch_vs_sequential_fp32_max_abs_difference": float(np.max(np.abs(raw - torch_output))),
        "dense_model_array_bytes": int(sum(w.nbytes + b.nbytes for w, b in layers)),
        "input_collision_bounds": input_collision_bounds(x, y8),
        "saved_validation_evaluation": evaluate_validation(layers, validation_path),
        "wall_seconds": time.monotonic() - start,
        "interpretation": "PC fit on saved learning inputs, not newly measured device MSE. Hand-held variability and identical-input conflicting labels remain. Teacher RGB recovered from supplied Notebook reference displays; original PNG files not independently compared.",
        "experiment_role": "additional exploratory H=100 comparison" if hidden_size == 100 else "planned hidden size",
    }
    export_header(layers, hidden_size, dataset_hash, seed, epochs, output_dir / "model_parameters.h")
    report["model_header_sha256"] = hashlib.sha256((output_dir / "model_parameters.h").read_bytes()).hexdigest()
    torch.save({"hidden_size": hidden_size, "epoch": epochs, "seed": seed,
                "training_data_sha256": dataset_hash, "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, output_dir / "checkpoint.pt")
    arrays = {f"{kind}_{i}": value for i, (w, b) in enumerate(layers, 1)
              for kind, value in [("weight", w), ("bias", b)]}
    np.savez(output_dir / "model_arrays.npz", **arrays)
    with (output_dir / "learning_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(curve)
    with (output_dir / "training_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["sample_id", "predicted_r8", "predicted_g8", "predicted_b8", "target_r8", "target_g8", "target_b8"])
        for i, (prediction, truth) in enumerate(zip(predicted, y8), 1):
            writer.writerow([i, *prediction.tolist(), *truth.astype(int).tolist()])
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"DONE H={hidden_size}: learning RGB8 MSE 3x3={report['pc_training_mse_rgb8_3x3']:.4f}, "
          f"6x4={report['pc_training_mse_rgb8_6x4']:.4f}, all33={report['pc_training_mse_rgb8_all33']:.4f}", flush=True)
    return report


def train_models(data_path, output_dir, hidden_sizes=(40, 50, 60, 100), epochs=50000, seed=42, validation_path=None):
    x, y8 = load_training(data_path)
    data_hash = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    reports = [train_one(x, y8, h, Path(output_dir) / f"h{h}", data_hash, epochs=epochs,
                         seed=seed, validation_path=validation_path) for h in hidden_sizes]
    columns = ["hidden_size", "epochs", "seed_numpy_and_torch", "pc_training_mse_rgb8_3x3",
               "pc_training_mse_rgb8_6x4", "pc_training_mse_rgb8_all33", "dense_model_array_bytes"]
    with (Path(output_dir) / "models_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)
    return reports


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=here / "data/current33.csv")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--hidden", type=int, nargs="+", default=[40, 50, 60, 100])
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation", type=Path, default=here.parent / "pruning/validation.csv")
    args = parser.parse_args()
    if args.out is None:
        (here / "runs").mkdir(exist_ok=True)
        args.out = Path(tempfile.mkdtemp(prefix="run_", dir=here / "runs"))
    print("Output:", args.out, flush=True)
    train_models(args.data, args.out, args.hidden, args.epochs, args.seed, args.validation)
