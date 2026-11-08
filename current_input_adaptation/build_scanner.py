"""Package adapted models and the exact provided reference RGB into V5."""

import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'pruning'), str(ROOT / 'experiment_today')]
from threshold_tools import load_model
from export_csr_arduino import array_definition, dense_layer_references
from build_fresh_scanner import replace_function


def build():
    output = HERE / 'color_retrain_v5'
    output.mkdir(exist_ok=True)
    reference = ROOT / 'learning_followup/reference_audit/provided/result.ppm'
    reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    tokens = reference.read_text().split()
    assert tokens[:4] == ['P3', '3', '3', '255'] and len(tokens) == 31
    targets = np.array(tokens[4:], np.uint8).reshape(9, 3)
    data = HERE / 'training60.csv'
    data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    with data.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 60
    for i, row in enumerate(rows):
        assert [int(row[f'target_{c}8']) for c in 'rgb'] == targets[[f'r{rr}c{cc}' for rr in range(1,4) for cc in range(1,4)].index(row['color_id'])].tolist()
    inputs = np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows], np.float32)
    cases = np.concatenate([np.array(list(itertools.product((0., 1.), repeat=3)), np.float32), inputs[[47, 52, 58, 59]]])
    header = '#pragma once\n#include <Arduino.h>\n#include <cmath>\n'
    descriptions = []
    for h in [40, 60, 100]:
        directory = HERE / f'models/h{h}'
        summary = json.loads((directory / 'summary.json').read_text())
        source = directory / 'model_parameters.h'
        assert summary['training_data_sha256'] == data_hash
        assert summary['reference_ppm_sha256'] == reference_hash
        assert summary['model_header_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
        text = '\n'.join(line for line in source.read_text().splitlines() if not line.startswith(('#include', '#pragma')))
        namespace = f'h{h}'
        header += f'namespace {namespace} {{\n' + text + '\n'
        layers = load_model(source)
        header += array_definition('EXPECTED', dense_layer_references(layers, cases)[-1], 'float')
        header += f'const char MODEL_SHA256[] = "{summary["model_header_sha256"]}";\n}}\n'
        descriptions.append('{' + str(h) + ', ' + ', '.join(f'{namespace}::{name}' for name in ['weight_1', 'bias_1', 'weight_2', 'bias_2', 'weight_3', 'bias_3', 'MODEL_SHA256', 'EXPECTED']) + '}')
    definitions = 'const Model models[3] = {\n' + ',\n'.join(descriptions) + '\n};\n'
    definitions += f'const char DATA_SHA256[] = "{data_hash}";\nconstexpr int TRAINING_ROWS=60;\n'
    definitions += f'const char REFERENCE_SHA256[] = "{reference_hash}";\n'
    definitions += array_definition('CHECK_INPUTS', cases, 'float') + array_definition('REFERENCE_RGB', targets, 'uint8_t')
    kernels = (ROOT / 'learning_followup/dense_kernels.h').read_text()
    kernels = kernels.replace('  Serial.println("FORMAT=DENSE_FP32_ONLY");', '  Serial.println("FORMAT=DENSE_FP32_ONLY");\n  Serial.println("REFERENCE_SOURCE=USER_PROVIDED_result.ppm");\n  Serial.print("REFERENCE_FILE_SHA256=");Serial.println(REFERENCE_SHA256);')
    header += kernels.replace('// INSERT_DEFINITIONS', definitions)
    (output / 'trained_models_v5.h').write_text(header)
    sketch = (ROOT / 'learning_followup/color_retrain_v3/color_retrain_v3.ino').read_text()
    sketch = sketch.replace('trained_models_v3.h', 'trained_models_v5.h').replace('COLOR_RETRAIN_V3', 'COLOR_RETRAIN_V5')
    sketch = replace_function(sketch, 'outputPPM', (HERE / 'capture_output.cpp').read_text())
    sketch = sketch.replace('再学習Dense版準備完了', '低入力域への適応版準備完了')
    (output / 'color_retrain_v5.ino').write_text(sketch)
    print('Generated COLOR_RETRAIN_V5 with trained and embedded targets both matching the provided result.ppm.')


if __name__ == '__main__':
    build()
