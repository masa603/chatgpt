"""Build one saved-input Dense/CSR comparison sketch per frozen hidden size."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pruning"))
from export_csr_arduino import array_definition, dense_layer_references, export
from threshold_tools import load_model, load_validation, prune


def export_one(h):
    selection = HERE / f"selection/h{h}/summary.json"
    report = json.loads(selection.read_text())
    source = ROOT / f"retraining/results/h{h}/model_parameters.h"
    validation_csv = ROOT / "pruning/validation.csv"
    output = HERE / f"compare_h{h}"
    output.mkdir(exist_ok=True)
    dense = load_model(source)
    sparse = prune(dense, report["selected"]["threshold"])
    validation = load_validation(validation_csv, expected_reads=2)
    with tempfile.TemporaryDirectory() as temporary:
        csr_header = Path(temporary) / "csr.h"
        export(source, selection, selection.parent / "selected_csr.npz", validation_csv, csr_header)
        text = csr_header.read_text().split("\n#if ENABLE_CSR_SELF_TEST\n")[0]
    text = "// Frozen current-33 model; saved-input comparison, not a new sensor capture.\n" + text
    for i, (weight, _) in enumerate(dense, 1):
        text += array_definition(f"dense_weight_{i}", weight.ravel(), "float")
    inputs = validation["inputs"]
    text += f"\nconstexpr uint16_t BENCHMARK_INPUT_COUNT = {len(inputs)};\n"
    text += f'const char BENCHMARK_INPUT_SHA256[] = "{hashlib.sha256(validation_csv.read_bytes()).hexdigest()}";\n'
    text += array_definition("BENCHMARK_INPUTS", inputs, "float")
    text += array_definition("DENSE_EXPECTED_OUTPUTS", dense_layer_references(dense, inputs)[-1], "float")
    text += array_definition("CSR_EXPECTED_OUTPUTS", dense_layer_references(sparse, inputs)[-1], "float")
    text += array_definition("VALIDATION_TARGETS", validation["targets"].astype(np.uint8), "uint8_t")
    with (ROOT / "retraining/data/current33.csv").open(newline="") as f:
        training = list(csv.DictReader(f))
    x = np.array([[r[f"input_{c}"] for c in "rgb"] for r in training], np.float32)
    target = np.array([[r[f"target_{c}8"] for c in "rgb"] for r in training], np.uint8)
    text += "constexpr uint16_t TRAINING_INPUT_COUNT = 33;\n"
    text += array_definition("TRAINING_INPUTS", x, "float")
    text += array_definition("TRAINING_TARGETS", target, "uint8_t")
    text += "constexpr uint16_t VALIDATION_COLORS_PER_READ = 64;\n"
    text += "constexpr uint16_t VALIDATION_READ_COUNT = 2;\n"
    (output / "parameters.h").write_text(text, encoding="utf-8")

    sketch = (ROOT / "hdjsk_benchmark/hdjsk_benchmark.ino").read_text()
    sketch = sketch.replace('#include "benchmark_parameters.h"', '#include "parameters.h"')
    helpers = (HERE / "metrics_and_checks.h").read_text()
    sketch = sketch.replace("void printMetadata() {", helpers + "\nvoid printMetadata() {")
    sketch = sketch.replace('  Serial.println("========== BENCHMARK START ==========");',
                            '  Serial.println("========== BENCHMARK START ==========");\n'
                            '  Serial.println("MODEL_GENERATION=CURRENT33_SEED42_EPOCH50000");\n'
                            '  Serial.println("BUILD_MODE=PAIRED_DENSE_AND_CSR");')
    sketch = sketch.replace('  Serial.print(" CSR="); Serial.println(csrModelBytes());',
                            '  Serial.print(" CSR="); Serial.println(csrModelBytes());\n'
                            '  printMSESummary();')
    sketch = sketch.replace("void runComparison() {\n", "void runComparison() {\n"
                            "  if (!validCSRModel()) {\n"
                            '    Serial.println("CSR INDEX CHECK: FAIL; no inference or timing run.");\n'
                            "    return;\n  }\n")
    sketch = sketch.replace('  Serial.println("FIXED INPUT CHECK: Dense=PASS CSR=PASS");',
                            '  Serial.println("CSR INDEX CHECK: PASS");\n'
                            '  Serial.println("FIXED INPUT CHECK: Dense=PASS CSR=PASS");')
    sketch = sketch.replace('  Serial.println("Dense/CSR speed comparison: send t to start.");',
                            f'  Serial.println("Current-33 H={h}: send t for MSE, model bytes and device speed.");')
    sketch = sketch.replace('SCOPE: inference on saved inputs; provisional model; color accuracy assessed separately.',
                            'SCOPE: saved learning/validation inputs; no new sensor capture; independent final evaluation not performed.')
    (output / f"compare_h{h}.ino").write_text(sketch, encoding="utf-8")
    print("Exported", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, nargs="+", default=[40, 60, 100])
    for hidden_size in parser.parse_args().hidden:
        export_one(hidden_size)
