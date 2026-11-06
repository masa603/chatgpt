"""Create a scanner that applies all six frozen models to each fresh sensor input."""

import csv
import itertools
import json
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pruning"))
from export_csr_arduino import array_definition, dense_layer_references, export
from threshold_tools import load_model, prune


def replace_function(text, name, replacement):
    start = re.search(rf"^(?:void|float|int|bool) {name}\(", text, re.M).start()
    opening = text.index("{", start)
    end, depth = opening + 1, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[:start] + replacement + text[end:]


def build():
    target = HERE / "fresh_scan_all"
    target.mkdir(exist_ok=True)
    with (ROOT / "retraining/data/current33.csv").open(newline="") as f:
        training = list(csv.DictReader(f))
    x = np.array([[r[f"input_{c}"] for c in "rgb"] for r in training], np.float32)
    y = np.array([[r[f"target_{c}8"] for c in "rgb"] for r in training[:9]], np.uint8)
    cases = np.concatenate([np.array(list(itertools.product((0., 1.), repeat=3)), np.float32), x[[0, 1, 6, 9]]])
    header = "// Generated from frozen current-33 models; no threshold is chosen from fresh scans.\n#pragma once\n#include <stddef.h>\n#include <stdint.h>\n#include <cmath>\n"
    models = []
    for h in (40, 60, 100):
        source = ROOT / f"retraining/results/h{h}/model_parameters.h"
        selection = HERE / f"selection/h{h}/summary.json"
        report = json.loads(selection.read_text())
        dense = load_model(source)
        sparse = prune(dense, report["selected"]["threshold"])
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary) / "csr.h"
            export(source, selection, selection.parent / "selected_csr.npz", ROOT / "pruning/validation.csv", generated)
            csr = generated.read_text().split("\n#if ENABLE_CSR_SELF_TEST\n")[0]
        csr = "\n".join(line for line in csr.splitlines() if not line.startswith(("#include", "#pragma")))
        namespace = f"h{h}"
        header += f"\nnamespace {namespace} {{\n" + csr + "\n"
        for i, (weight, _) in enumerate(dense, 1):
            header += array_definition(f"dense_weight_{i}", weight.ravel(), "float")
        header += array_definition("DENSE_CHECK", dense_layer_references(dense, cases)[-1], "float")
        header += array_definition("CSR_CHECK", dense_layer_references(sparse, cases)[-1], "float")
        dense_size = " + ".join(f"sizeof(dense_weight_{i}) + sizeof(bias_{i})" for i in (1, 2, 3))
        csr_size = " + ".join(f"sizeof(weight_{i}_values) + sizeof(weight_{i}_col_idx) + sizeof(weight_{i}_row_ptr) + sizeof(bias_{i})" for i in (1, 2, 3))
        header += f'static_assert({dense_size} == CSR_DENSE_REFERENCE_BYTES, "Dense size mismatch");\n'
        header += f'static_assert({csr_size} == CSR_EXPECTED_MODEL_BYTES, "CSR size mismatch");\n'
        header += "}\n"
        layers = []
        for i, (w, _) in enumerate(dense, 1):
            prefix = f"{namespace}::"
            layers.append("{" + ", ".join(prefix + name for name in [f"dense_weight_{i}", f"bias_{i}", f"weight_{i}_values", f"weight_{i}_col_idx", f"weight_{i}_row_ptr"]) +
                          f", {w.shape[0]}, {w.shape[1]}, sizeof({prefix}weight_{i}_values)/sizeof(float)" + "}")
        models.append("{" + f"{h}, {namespace}::CSR_THRESHOLD_TEXT, {namespace}::CSR_SOURCE_MODEL_SHA256, {namespace}::CSR_DENSE_REFERENCE_BYTES, {namespace}::CSR_EXPECTED_MODEL_BYTES, {namespace}::DENSE_CHECK, {namespace}::CSR_CHECK, " + "{" + ", ".join(layers) + "}}")
    definitions = "const Model models[3] = {\n" + ",\n".join(models) + "\n};\n"
    definitions += array_definition("CHECK_INPUTS", cases, "float")
    definitions += array_definition("REFERENCE_RGB", y, "uint8_t")
    kernels = (HERE / "fresh_scan_kernels.h").read_text().replace("// INSERT_MODEL_DEFINITIONS", definitions)
    (target / "models_all.h").write_text(header + kernels)

    sketch = (ROOT / "hdjsk/hdjsk.ino").read_text()
    sketch = sketch.replace('#include "model_parameters.h"', '#include "models_all.h"\nvoid outputNNInputs();')
    sketch = sketch.replace("#define HIDDEN_SIZE1 40\n#define HIDDEN_SIZE2 40\n", "")
    sketch = sketch.replace("#define MAX_IMAGE_SIZE 8", "#define MAX_IMAGE_SIZE 3")
    sketch = sketch.replace("int imageSize = 0;", "int imageSize = 3;\nuint8_t freshImages[6][3][3][3];\nfloat rawHistory[3][3][3];\nunsigned long sampleCompletedMs[3][3];\nunsigned int captureNumber = 1;")
    sketch = replace_function(sketch, "predict", '''void predict(float input[3], float output[3]) {
  float allOutputs[6][3];
  fresh_models::inferAll(input, allOutputs);
  for (int model = 0; model < 6; model++) {
    for (int c = 0; c < 3; c++) {
      float value = allOutputs[model][c];
      if (value < 0.0f) value = 0.0f;
      if (value > 1.0f) value = 1.0f;
      freshImages[model][currentRow][currentColumn][c] = static_cast<int>(value * 255.0 + 0.5);
    }
  }
  for (int c = 0; c < 3; c++) output[c] = allOutputs[1][c];
}''')
    sketch = replace_function(sketch, "outputPPM", (HERE / "fresh_scan_output.cpp").read_text())
    sketch = sketch.replace("    outputPPM();\n    outputNNInputs();", "    outputPPM();")
    sketch = sketch.replace("inputHistory[currentRow][currentColumn][channel] = RGBInput[channel];", "inputHistory[currentRow][currentColumn][channel] = RGBInput[channel];\n    rawHistory[currentRow][currentColumn][channel] = rawRGB[channel];")
    read_start = sketch.index("void handleRead()")
    sketch = sketch[:read_start] + sketch[read_start:].replace("  readRGB();", "  readRGB();\n  sampleCompletedMs[currentRow][currentColumn] = millis();", 1)
    sketch = sketch.replace("    case 'b':", "    case 'n':\n    case 'N':\n      calibrationStep = 0; currentRow = 0; currentColumn = 0; imageFinished = false; captureNumber++;\n      Serial.println(\"新しい測定：黒を当てて b、白を当てて w、各色で r を送信してください。\");\n      break;\n\n    case 'b':")
    sketch = sketch.replace("  inputImageSize();", "  if (!fresh_models::selfCheck()) {\n    Serial.println(\"Model check failed; save this log before reading colors.\");\n    while (true) delay(1000);\n  }")
    sketch = sketch.replace("ニューラルネット版準備完了", "全6条件の3x3スキャナ準備完了（画像サイズ入力不要）")
    sketch = sketch.replace("--- NN補正後 ---", "--- CSR40補正後（全6条件は9色後に出力）---")
    sketch = replace_function(sketch, "inputImageSize", "")
    (target / "fresh_scan_all.ino").write_text(sketch)
    print("Generated", target)


if __name__ == "__main__":
    build()
