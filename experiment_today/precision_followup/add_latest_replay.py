"""Embed the latest archived input for new-coefficient device replay, without tuning."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiment_today"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pruning"))
from export_csr_arduino import array_definition, dense_layer_references
from threshold_tools import load_model, prune


def add_replay(target):
    source = BASE / "version_mismatch/old_firmware_retry_01"
    if not (source / "summary.json").exists():
        return
    summary = json.loads((source / "summary.json").read_text())
    with (source / "inputs.csv").open(newline="") as f:
        data = list(csv.DictReader(f))
    raw = np.array([[r[f"raw_{c}"] for c in "rgb"] for r in data], np.float32)
    black, white = np.array(summary["black_rgb"], np.float32), np.array(summary["white_rgb"], np.float32)
    inputs = np.clip((raw - black) / (white - black), 0, 1)
    targets = np.array([[r[f"target_{c}8"] for c in "rgb"] for r in data], np.uint8)
    expected, diagnostics = [], []
    for h in (40,60,100):
        report = json.loads((HERE / f"selection/h{h}/summary.json").read_text())
        dense = load_model(ROOT / f"retraining/results/h{h}/model_parameters.h")
        sparse = prune(dense, report["selected"]["threshold"])
        for mode, model in [("DENSE",dense),("CSR",sparse)]:
            raw_output = dense_layer_references(model, inputs)[-1]
            expected.append(raw_output)
            pixels = np.floor(np.clip(raw_output.astype(float), 0, 1) * 255 + .5)
            diagnostics.append({"hidden_size":h,"format":mode,"mse_pc_replay":float(np.mean((pixels-targets.astype(float))**2)),"role":"PC_REPLAY_ONLY; NEW_DEVICE_REPLAY_NOT_YET_RECEIVED"})
    raw_hash = hashlib.sha256((source / "raw_log.txt").read_bytes()).hexdigest()
    text = "\n#define PRECISION_LATEST_REPLAY_ENABLED 1\nnamespace precision_replay {\n"
    text += f'const char SOURCE_LOG_SHA256[] = "{raw_hash}";\n'
    text += array_definition("INPUTS", inputs, "float")
    text += array_definition("TARGETS", targets, "uint8_t")
    text += array_definition("EXPECTED", np.concatenate(expected), "float")
    text += "}\n"
    header = target / "precision_models.h"
    header.write_text(header.read_text() + text)
    ino = target / "precision_scan.ino"
    sketch = ino.read_text()
    sketch = sketch.replace("void handleSerialCommand() {", (HERE / "latest_replay.cpp").read_text() + "\nvoid handleSerialCommand() {")
    sketch = sketch.replace("    case 'n':", "    case 'v':\n    case 'V':\n      printLatestReplay();\n      break;\n\n    case 'n':", 1)
    marker = '  Serial.println("全6条件の3x3スキャナ準備完了（画像サイズ入力不要）");'
    sketch = sketch.replace(marker, marker + '\n  Serial.println("Send v to replay the last measured chart with the new candidates.");')
    ino.write_text(sketch)
    manifest = {"source_log":str((source / "raw_log.txt").relative_to(BASE)),"source_log_sha256":raw_hash,
                "scope":"Physical-input log was received after follow-up thresholds were frozen; exact acquisition time is unconfirmed. It was not used for selection. Reuse input for new-coefficient inference; not a new sensor capture at replay time and not independent final evaluation.",
                "source_counter":summary["capture_counter"],"input_reconstruction":"FP32 normalize from archived ADC means and calibration, then clip 0..1.",
                "input_array_sha256_float32_le":hashlib.sha256(inputs.astype('<f4').tobytes()).hexdigest(),"pc_results":diagnostics,"device_replay":"PENDING"}
    (HERE / "latest_replay_manifest.json").write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == "__main__":
    add_replay(HERE / "precision_scan")
