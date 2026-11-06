"""Test all six build selections without treating host sizes as Arduino measurements."""

from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / "pruning")]
from verify_host import ARDUINO_STUB
from export_csr_arduino import dense_layer_references
from threshold_tools import load_model, prune

RUNNER = r'''
#include "Arduino.h"
#include <cassert>
#include <cmath>
#include <iostream>
MockSerial Serial;
void delay(unsigned long) {}
#include "memory_probe.ino"
int main(int argc,char** argv) {
  assert(argc==2 && CSR_HIDDEN_SIZE==PROBE_H);
  if(std::string(argv[1])=="predict") {
    float input[3],output[3]; std::cout<<std::setprecision(9);
    while(std::cin>>input[0]>>input[1]>>input[2]) {
      modelPredict(input,output);
      for(int c=0;c<3;c++) std::cout<<output[c]<<(c==2?'\n':' ');
    }
  } else {
    setup(); loop();
    for(int c=0;c<3;c++) assert(std::isfinite(probeOutput[c]));
    std::cout<<PROBE_MODEL_ARRAY_BYTES<<'\n'<<Serial.output.str();
  }
}
'''


def main():
    records = []
    for path in (ROOT / "retraining/data/current33.csv", ROOT / "pruning/validation.csv"):
        with path.open(newline="") as f:
            records.extend(csv.DictReader(f))
    cases = np.concatenate([np.array([[r[f"input_{c}"] for c in "rgb"] for r in records], np.float32),
                            np.random.default_rng(42).random((256, 3), dtype=np.float32)])
    input_text = "\n".join(" ".join(format(float(v), ".9g") for v in row) for row in cases) + "\n"
    report = {"scope": "Host tests only; no host executable sizes recorded as Arduino Flash/RAM.",
              "arduino_single_model_build_measurements": "NOT_RECEIVED", "conditions": []}
    with tempfile.TemporaryDirectory(prefix="single-model-probe-") as name:
        directory = Path(name)
        (directory / "Arduino.h").write_text(ARDUINO_STUB)
        (directory / "main.cpp").write_text(RUNNER)

        def verify(condition):
            h, csr = condition
            defines = [f"-DPROBE_H={h}", f"-DPROBE_CSR={csr}"]
            includes = [f"-I{directory}", f"-I{HERE / 'memory_probe'}"]
            exe = directory / f"h{h}_{csr}"
            subprocess.run(["g++", "-std=c++11", "-O2", "-ffp-contract=off", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                            *defines, *includes, str(directory / "main.cpp"), "-o", str(exe)], check=True, text=True)
            preprocessed = subprocess.run(["g++", "-std=c++11", "-E", "-P", *defines, *includes, str(directory / "main.cpp")],
                                          check=True, capture_output=True, text=True).stdout
            assert ("dense_weight_1" in preprocessed) == (csr == 0)
            assert ("weight_1_values" in preprocessed) == (csr == 1)
            assert "BENCHMARK_INPUTS" not in preprocessed and "TRAINING_INPUTS" not in preprocessed
            selection = json.loads((HERE / f"selection/h{h}/summary.json").read_text())
            dense = load_model(ROOT / f"retraining/results/h{h}/model_parameters.h")
            model = prune(dense, selection["selected"]["threshold"]) if csr else dense
            predictions = subprocess.run([str(exe), "predict"], input=input_text, check=True, capture_output=True, text=True).stdout
            actual = np.loadtxt(io.StringIO(predictions))
            expected = dense_layer_references(model, cases)[-1]
            assert np.allclose(actual, expected, rtol=1e-6, atol=1e-6)
            smoke = subprocess.run([str(exe), "smoke"], check=True, capture_output=True, text=True).stdout
            expected_bytes = selection["selected"]["csr_bytes" if csr else "dense_bytes"]
            assert int(smoke.splitlines()[0]) == expected_bytes
            assert f"H={h}\n" in smoke and f"FORMAT_ID={csr}\n" in smoke and selection["model_sha256"] in smoke
            return {"hidden_size": h, "format": "CSR" if csr else "DENSE", "cpp11_asan_ubsan": "PASS",
                    "exactly_one_format_in_preprocessed_program": True, "model_array_bytes": expected_bytes,
                    "model_header_sha256": selection["model_sha256"], "differential_inputs": len(cases),
                    "max_abs_fp32_output_difference": float(np.max(np.abs(actual - expected))),
                    "volatile_io_wrapper": "PASS", "arduino_flash_bytes": None, "arduino_static_ram_bytes": None}

        conditions = [(h, csr) for h in (40, 60, 100) for csr in (0, 1)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            report["conditions"] = list(pool.map(verify, conditions))
    (HERE / "memory_probe_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: all six compile-time selections, single-format preprocessing, 417 FP32 predictions each and volatile wrapper; ASan/UBSan.")


if __name__ == "__main__":
    main()
