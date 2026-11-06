"""Generate a single-model build probe with six compile-time choices."""

import csv
import json
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pruning"))
from export_csr_arduino import array_definition, export
from threshold_tools import load_model


def function_source(text, name):
    start = re.search(rf"^(?:NOINLINE )?(?:void|size_t) {name}\([^;{{]*\)\s*\{{", text, re.M).start()
    opening = text.index("{", start)
    end, depth = opening + 1, 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def build():
    output = HERE / "memory_probe"
    output.mkdir(exist_ok=True)
    header = '''// Frozen model arrays; exactly one H/format survives preprocessing.
#pragma once
#include <stddef.h>
#include <stdint.h>
static_assert(sizeof(float) == 4, "FP32 float required");
#if PROBE_CSR != 0 && PROBE_CSR != 1
#error "Set PROBE_CSR to 0 (Dense) or 1 (CSR)."
#endif
'''
    expected = []
    for n, h in enumerate((40, 60, 100)):
        source = ROOT / f"retraining/results/h{h}/model_parameters.h"
        selection = HERE / f"selection/h{h}/summary.json"
        report = json.loads(selection.read_text())
        layers = load_model(source)
        header += ("#if" if n == 0 else "#elif") + f" PROBE_H == {h}\n#if PROBE_CSR == 0\n"
        header += f"constexpr int CSR_HIDDEN_SIZE = {h};\n"
        header += f'const char CSR_SOURCE_MODEL_SHA256[] = "{report["model_sha256"]}";\n'
        for i, (w, b) in enumerate(layers, 1):
            header += array_definition(f"dense_weight_{i}", w.ravel(), "float")
            header += array_definition(f"bias_{i}", b, "float")
        dense_size = " + ".join(f"sizeof(dense_weight_{i}) + sizeof(bias_{i})" for i in (1, 2, 3))
        header += f"constexpr size_t PROBE_MODEL_ARRAY_BYTES = {dense_size};\n"
        header += f'static_assert(PROBE_MODEL_ARRAY_BYTES == {report["selected"]["dense_bytes"]}, "Dense array size mismatch");\n'
        header += "#else\n"
        with tempfile.TemporaryDirectory() as temporary:
            csr = Path(temporary) / "csr.h"
            export(source, selection, selection.parent / "selected_csr.npz", ROOT / "pruning/validation.csv", csr)
            csr_text = csr.read_text().split("\n#if ENABLE_CSR_SELF_TEST\n")[0]
        header += "\n".join(line for line in csr_text.splitlines() if not line.startswith(("#include", "#pragma"))) + "\n"
        csr_size = " + ".join(f"sizeof(weight_{i}_values) + sizeof(weight_{i}_col_idx) + sizeof(weight_{i}_row_ptr) + sizeof(bias_{i})" for i in (1, 2, 3))
        header += f"constexpr size_t PROBE_MODEL_ARRAY_BYTES = {csr_size};\n"
        header += 'static_assert(PROBE_MODEL_ARRAY_BYTES == CSR_EXPECTED_MODEL_BYTES, "CSR array size mismatch");\n#endif\n'
        for mode, key in [("DENSE", "dense_bytes"), ("CSR", "csr_bytes")]:
            expected.append({"hidden_size": h, "format": mode, "model_array_bytes": report["selected"][key],
                             "status": "ARDUINO_COMPILE_NOT_RECORDED", "flash_used_bytes": "", "flash_max_bytes": "",
                             "static_ram_used_bytes": "", "ram_max_bytes": "", "source_log": ""})
    header += '#else\n#error "Set PROBE_H to 40, 60 or 100."\n#endif\n'
    (output / "model_data.h").write_text(header)
    benchmark = (ROOT / "hdjsk_benchmark/hdjsk_benchmark.ino").read_text()
    dense_kernel = function_source(benchmark, "densePredict").replace("densePredict(", "modelPredict(", 1)
    csr_kernel = function_source(benchmark, "sparseLayer") + "\n" + function_source(benchmark, "csrPredict").replace("csrPredict(", "modelPredict(", 1)
    sketch = '''// Change these two settings, then click Verify for each row in the README.
#ifndef PROBE_H
#define PROBE_H 40
#endif
#ifndef PROBE_CSR
#define PROBE_CSR 0  // 0 = Dense FP32, 1 = pruned CSR FP32
#endif

#include <Arduino.h>
#include "model_data.h"
#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif
void modelPredict(const float *input, float *output);

// Volatile inputs prevent the model calculation being folded into constants.
// Volatile outputs keep inference results observable even without serial traffic.
volatile float probeInput[3] = {0.2f, 0.4f, 0.6f};
volatile float probeOutput[3] = {0.0f, 0.0f, 0.0f};
#if PROBE_CSR == 0
''' + dense_kernel + "\n#else\n" + csr_kernel + '''
#endif

void setup() {
  Serial.begin(9600);
  delay(1000);
  Serial.println("SINGLE_MODEL_MEMORY_PROBE");
  Serial.print("H="); Serial.println(PROBE_H);
  Serial.print("FORMAT_ID="); Serial.println(PROBE_CSR);
  Serial.print("MODEL_SOURCE_SHA256="); Serial.println(CSR_SOURCE_MODEL_SHA256);
  Serial.print("MODEL_ARRAY_BYTES="); Serial.println(PROBE_MODEL_ARRAY_BYTES);
}

void loop() {
  float input[3] = {probeInput[0], probeInput[1], probeInput[2]};
  float output[3];
  modelPredict(input, output);
  for (int c = 0; c < 3; c++) probeOutput[c] = output[c];
  delay(100);
}
'''
    (output / "memory_probe.ino").write_text(sketch)
    template = HERE / "single_model_memory_template.csv"
    with template.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(expected[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(expected)
    print("Generated memory_probe: one model and one format per build; no validation/training image arrays.")


if __name__ == "__main__":
    build()
