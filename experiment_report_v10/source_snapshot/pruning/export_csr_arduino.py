"""Export the selected CSR model and pruned-dense references for Arduino."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from threshold_tools import load_model, prune, storage


def float_literal(value):
    value = np.float32(value)
    if not np.isfinite(value):
        raise ValueError("Only finite FP32 values can be exported.")
    text = format(float(value), ".9g")
    if "." not in text and "e" not in text.lower():
        text += ".0"
    if np.float32(float(text)) != value:
        raise AssertionError("FP32 literal did not round trip.")
    return text + "f"


def array_definition(name, values, ctype):
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError("An empty CSR layer needs C++ padding and revised capacity accounting before export.")
    dimensions = "".join(f"[{n}]" for n in values.shape)
    literal = float_literal if ctype == "float" else lambda value: str(int(value))
    if values.ndim == 1:
        lines = [", ".join(literal(value) for value in values[i:i + 8])
                 for i in range(0, len(values), 8)]
    elif values.ndim == 2:
        lines = ["{" + ", ".join(literal(value) for value in row) + "}" for row in values]
    else:
        raise ValueError("Only vectors and matrices are exported.")
    return f"const {ctype} {name}{dimensions} = {{\n  " + ",\n  ".join(lines) + "\n};\n"


def dense_layer_references(layers, inputs):
    """Independent dense FP32 calculation, including zeroed matrix entries."""
    x = np.asarray(inputs, dtype=np.float32)
    references = []
    for i, (weight, bias) in enumerate(layers):
        output = np.broadcast_to(bias, (len(x), len(bias))).copy()
        for column in range(weight.shape[1]):
            output += x[:, column:column + 1] * weight[:, column][None, :]
        x = np.maximum(output, np.float32(0)) if i < 2 else output
        references.append(x.copy())
    return references


def export(header_path, selection_path, archive_path, validation_path, output_path):
    header_path, selection_path = Path(header_path), Path(selection_path)
    archive_path, validation_path, output_path = map(Path, (archive_path, validation_path, output_path))
    report = json.loads(selection_path.read_text())
    selected = report.get("selected")
    if selected is None:
        raise ValueError("A selected or explicitly provisional candidate is required.")
    model_hash = hashlib.sha256(header_path.read_bytes()).hexdigest()
    if model_hash != report["model_sha256"]:
        raise ValueError("The source model changed; re-run threshold selection first.")
    if hashlib.sha256(validation_path.read_bytes()).hexdigest() != report["validation_csv_sha256"]:
        raise ValueError("Validation data changed; re-run threshold selection first.")
    dense = load_model(header_path)
    pruned = prune(dense, selected["threshold"])
    h = len(dense[0][1])
    definitions = []
    actual_bytes = 0
    with np.load(archive_path, allow_pickle=False) as packed:
        expected_names = {f"{kind}_{i}" for i in (1, 2, 3)
                          for kind in ("values", "col_idx", "row_ptr", "bias")}
        if set(packed.files) != expected_names:
            raise ValueError("Unexpected CSR archive arrays.")
        for i, (weight, bias) in enumerate(pruned, start=1):
            values, columns, pointers, saved_bias = [packed[f"{kind}_{i}"]
                                                     for kind in ("values", "col_idx", "row_ptr", "bias")]
            if (values.dtype != np.float32 or columns.dtype != np.uint8 or
                    pointers.dtype != np.uint16 or saved_bias.dtype != np.float32):
                raise ValueError("CSR array dtypes must be float32/uint8/uint16/float32.")
            if (values.ndim != 1 or columns.shape != values.shape or pointers.shape != (weight.shape[0] + 1,)
                    or saved_bias.shape != bias.shape or pointers[0] != 0 or pointers[-1] != len(values)
                    or np.any(np.diff(pointers.astype(np.int64)) < 0) or np.any(columns >= weight.shape[1])):
                raise ValueError("Invalid CSR shapes or indices.")
            restored = np.zeros_like(weight)
            for row in range(len(bias)):
                start, end = map(int, pointers[row:row + 2])
                if np.any(np.diff(columns[start:end].astype(int)) <= 0):
                    raise ValueError("Columns must be strictly increasing within each row.")
                restored[row, columns[start:end]] = values[start:end]
            if not np.array_equal(restored, weight) or not np.array_equal(saved_bias, bias):
                raise ValueError("CSR archive does not match the selected pruned-dense model.")
            for name, array, ctype in [
                (f"weight_{i}_values", values, "float"),
                (f"weight_{i}_col_idx", columns, "uint8_t"),
                (f"weight_{i}_row_ptr", pointers, "uint16_t"),
                (f"bias_{i}", saved_bias, "float"),
            ]:
                definitions.append(array_definition(name, array, ctype))
                actual_bytes += array.nbytes
    if actual_bytes != selected["csr_bytes"] or actual_bytes != storage(pruned)["csr_bytes"]:
        raise ValueError("CSR byte count differs from the threshold-selection report.")

    # All eight normalized-input corners plus four measured interior cases.
    with validation_path.open(newline="") as f:
        readings = list(csv.DictReader(f))
    first = {r["color_id"]: r for r in readings if r["read_id"] == report["validation_read_ids"][0]}
    interiors = [[first[color][f"input_{channel}"] for channel in "rgb"]
                 for color in ("r1c2", "r2c3", "r3c4", "r6c4")]
    corners = list(itertools.product((0.0, 1.0), repeat=3))
    inputs = np.concatenate((np.asarray(corners, dtype=np.float32), np.asarray(interiors, dtype=np.float32)))
    references = dense_layer_references(pruned, inputs)
    provisional = report["selection_status"] == "provisional_input_collisions"
    text = "// Generated by pruning/export_csr_arduino.py. Do not edit arrays by hand.\n#pragma once\n"
    text += "#include <stddef.h>\n#include <stdint.h>\n\n"
    text += "static_assert(sizeof(float) == 4, \"FP32 float is required\");\n"
    text += f"constexpr int CSR_HIDDEN_SIZE = {h};\n"
    text += f"constexpr size_t CSR_DENSE_REFERENCE_BYTES = {selected['dense_bytes']};\n"
    text += f"constexpr size_t CSR_EXPECTED_MODEL_BYTES = {actual_bytes};\n"
    text += f"constexpr size_t CSR_NONZERO_COUNT = {selected['nonzero']};\n"
    text += f"constexpr bool CSR_PROVISIONAL = {'true' if provisional else 'false'};\n"
    text += f'const char CSR_THRESHOLD_TEXT[] = "{selected["threshold"]}";\n'
    text += f'const char CSR_SOURCE_MODEL_SHA256[] = "{model_hash}";\n\n'
    text += "\n".join(definitions)
    text += "\n#if ENABLE_CSR_SELF_TEST\n"
    text += "// Expected hidden/output values are from pruned-dense, before RGB8 conversion.\n"
    text += f"constexpr int CSR_SELF_TEST_CASES = {len(inputs)};\n"
    text += array_definition("CSR_SELF_TEST_INPUTS", inputs, "float")
    for name, values in zip(("LAYER1", "LAYER2", "OUTPUT"), references):
        text += array_definition(f"CSR_SELF_TEST_{name}", values, "float")
    text += "#endif\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {"model_bytes": int(actual_bytes), "self_test_cases": len(inputs),
            "self_test_reference_bytes": sum(array.nbytes for array in [inputs, *references]),
            "provisional": provisional, "output": str(output_path)}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, default=root / "hdjsk/model_parameters.h")
    parser.add_argument("--selection", type=Path, default=root / "pruning/results/h40/summary.json")
    parser.add_argument("--archive", type=Path, default=root / "pruning/results/h40/selected_csr.npz")
    parser.add_argument("--validation", type=Path, default=root / "pruning/validation.csv")
    parser.add_argument("--output", type=Path, default=root / "hdjsk_csr/model_parameters_csr.h")
    args = parser.parse_args()
    print(json.dumps(export(args.header, args.selection, args.archive, args.validation, args.output), indent=2))
