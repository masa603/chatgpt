"""Generate Dense scanner headers with fixed-input checks for the new models."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from train_current33 import float_literal, forward_fp32, load_training


def matrix_definition(name, values):
    rows, columns = values.shape
    return (f"const float {name}[{rows}][{columns}] = {{\n" +
            ",\n".join("  {" + ", ".join(float_literal(v) for v in row) + "}" for row in values) + "\n};\n")


def build_header(root, hidden_size, output):
    model_dir = root / "retraining/results" / f"h{hidden_size}"
    summary = json.loads((model_dir / "summary.json").read_text())
    base = model_dir / "model_parameters.h"
    if hashlib.sha256(base.read_bytes()).hexdigest() != summary["model_header_sha256"]:
        raise ValueError("Frozen model header changed.")
    data = root / "retraining/data/current33.csv"
    if hashlib.sha256(data.read_bytes()).hexdigest() != summary["training_data_sha256"]:
        raise ValueError("Training data changed.")
    x, _ = load_training(data)
    with np.load(model_dir / "model_arrays.npz", allow_pickle=False) as arrays:
        layers = [(arrays[f"weight_{i}"].copy(), arrays[f"bias_{i}"].copy()) for i in (1, 2, 3)]
    cases = np.concatenate([np.array(list(itertools.product((0., 1.), repeat=3)), dtype=np.float32), x[[0, 1, 6, 9]]])
    expected = forward_fp32(layers, cases)
    text = base.read_text()
    text += f'\nconst char MODEL_FROZEN_HEADER_SHA256[] = "{summary["model_header_sha256"]}";\n'
    text += f"constexpr unsigned long MODEL_EXPECTED_BYTES = {summary['dense_model_array_bytes']};\n"
    text += f"constexpr double PC_TRAIN_MSE_3X3 = {summary['pc_training_mse_rgb8_3x3']:.17g};\n"
    text += f"constexpr double PC_TRAIN_MSE_6X4 = {summary['pc_training_mse_rgb8_6x4']:.17g};\n"
    text += f"constexpr double PC_TRAIN_MSE_ALL33 = {summary['pc_training_mse_rgb8_all33']:.17g};\n"
    text += f"constexpr int DENSE_CHECK_CASES = {len(cases)};\n"
    text += matrix_definition("DENSE_CHECK_INPUTS", cases)
    text += matrix_definition("DENSE_CHECK_OUTPUTS", expected)
    output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    target = root / "hdjsk_retrained"
    target.mkdir(exist_ok=True)
    for h in (60, 100):
        build_header(root, h, target / f"model_h{h}_current33.h")
        print("Generated", target / f"model_h{h}_current33.h")
