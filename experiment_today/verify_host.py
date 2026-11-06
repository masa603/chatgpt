"""Check generated firmware on the PC; never record PC timings as Arduino results."""

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
sys.path[:0] = [str(ROOT / "pruning"), str(ROOT / "retraining")]
from export_csr_arduino import dense_layer_references
from threshold_tools import load_model, predict_rgb8, prune


ARDUINO_STUB = r'''
#pragma once
#include <cstdint>
#include <deque>
#include <iomanip>
#include <sstream>
#include <string>
unsigned long micros();
void delay(unsigned long);
struct MockSerial {
  std::deque<char> input;
  std::ostringstream output;
  void begin(unsigned long) {}
  template<class T> void print(T value, int=10) { output << value; }
  void print(float value, int precision=2) { output << std::fixed << std::setprecision(precision) << value; }
  void print(double value, int precision=2) { output << std::fixed << std::setprecision(precision) << value; }
  template<class T> void println(T value, int precision=2) { print(value,precision); output << '\n'; }
  int available() { return static_cast<int>(input.size()); }
  int read() { int v=input.front(); input.pop_front(); return v; }
};
extern MockSerial Serial;
'''

HOST_RUNNER = r'''
#include "Arduino.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
MockSerial Serial;
unsigned long micros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}
void delay(unsigned long) {}
#include "firmware.ino"
int main(int argc, char** argv) {
  assert(argc==2);
  assert(validCSRModel());
  assert(denseModelBytes()==CSR_DENSE_REFERENCE_BYTES);
  assert(csrModelBytes()==CSR_EXPECTED_MODEL_BYTES);
  bool dp,cp; float de,ce;
  fixedInputCheck(dp,cp,de,ce);
  assert(dp && cp);
  uint32_t totals[5]={100,200,300,400,500}; double mean,sd;
  timingStatistics(totals,mean,sd);
  assert(std::fabs(mean-3.0)<1e-12 && std::fabs(sd-std::sqrt(2.5))<1e-12);
  float v[2]={1,2}, b[1]={0}; uint8_t c[2]={0,1}; uint16_t p[2]={0,2};
  assert(validCSRLayer(v,c,p,b,1,3,2));
  c[1]=3; assert(!validCSRLayer(v,c,p,b,1,3,2)); c[1]=1;
  p[1]=3; assert(!validCSRLayer(v,c,p,b,1,3,2)); p[1]=2;
  v[0]=std::numeric_limits<float>::quiet_NaN(); assert(!validCSRLayer(v,c,p,b,1,3,2));
  if(std::string(argv[1])=="predict") {
    std::cout << std::setprecision(9);
    float input[3],d[3],s[3];
    while(std::cin>>input[0]>>input[1]>>input[2]) {
      densePredict(input,d); csrPredict(input,s);
      for(int i=0;i<3;i++) std::cout<<d[i]<<' ';
      for(int i=0;i<3;i++) std::cout<<s[i]<<' ';
      for(int i=0;i<3;i++) std::cout<<outputToRGB8(d[i])<<' ';
      for(int i=0;i<3;i++) std::cout<<outputToRGB8(s[i])<<(i==2?'\n':' ');
    }
  } else {
    setup(); Serial.input.push_back('t'); loop();
    std::string output=Serial.output.str();
    assert(output.find("FIXED INPUT CHECK: Dense=PASS CSR=PASS")!=std::string::npos);
    assert(output.find("TIMING VALID: YES")!=std::string::npos);
    assert(output.find("INDEPENDENT_FINAL_EVALUATION=NOT_PERFORMED")!=std::string::npos);
    std::cout << output;
  }
}
'''


def main():
    with (ROOT / "retraining/data/current33.csv").open(newline="") as f:
        learning = list(csv.DictReader(f))
    with (ROOT / "pruning/validation.csv").open(newline="") as f:
        validation = list(csv.DictReader(f))
    cases = np.concatenate([np.array([[r[f"input_{c}"] for c in "rgb"] for r in learning + validation], np.float32),
                            np.random.default_rng(42).random((256, 3), dtype=np.float32)])
    report = {"scope": "Host C++11 tests only. All timings produced by these tests are excluded from experiment results.",
              "arduino_hardware_results": "NOT_RECORDED", "models": {}}
    with tempfile.TemporaryDirectory(prefix="today-host-check-") as temporary:
        directory = Path(temporary)
        (directory / "Arduino.h").write_text(ARDUINO_STUB)
        (directory / "main.cpp").write_text(HOST_RUNNER)
        for h in (40, 60, 100):
            source = HERE / f"compare_h{h}"
            (directory / "firmware.ino").write_text((source / f"compare_h{h}.ino").read_text())
            exe = directory / "check"
            subprocess.run(["g++", "-std=c++11", "-O2", "-ffp-contract=off", "-fsanitize=address,undefined",
                            "-fno-omit-frame-pointer", "-Wall", "-Wextra", f"-I{source}", f"-I{directory}",
                            str(directory / "main.cpp"), "-o", str(exe)], check=True, capture_output=True, text=True)
            output = subprocess.run([str(exe), "predict"], input="\n".join(" ".join(format(float(v), ".9g") for v in row) for row in cases) + "\n",
                                    check=True, capture_output=True, text=True).stdout
            result = np.loadtxt(io.StringIO(output))
            selection = json.loads((HERE / f"selection/h{h}/summary.json").read_text())
            comparison = json.loads((HERE / f"selection/h{h}/comparison.json").read_text())
            dense = load_model(ROOT / f"retraining/results/h{h}/model_parameters.h")
            sparse = prune(dense, selection["selected"]["threshold"])
            max_error = 0.0
            for offset, layers in [(0, dense), (3, sparse)]:
                expected = dense_layer_references(layers, cases)[-1]
                max_error = max(max_error, float(np.max(np.abs(result[:, offset:offset + 3] - expected))))
                assert np.allclose(result[:, offset:offset + 3], expected, atol=1e-6, rtol=1e-6)
                assert np.array_equal(result[:, offset + 6:offset + 9].astype(np.uint8), predict_rgb8(layers, cases))
            serial = subprocess.run([str(exe), "serial"], check=True, capture_output=True, text=True).stdout
            rows = {r.split(",")[0]: r.split(",") for r in serial.splitlines()
                    if r.startswith(("learning_", "validation_mean,"))}
            for chart in ("3x3", "6x4", "all33"):
                row = rows[f"learning_{chart}"]
                for offset, mode in [(3, "dense"), (4, "csr")]:
                    assert abs(float(row[offset]) - comparison[f"learning_{mode}_mse_{chart}"]) < 1e-6
            for offset, mode in [(3, "dense"), (4, "csr")]:
                assert abs(float(rows["validation_mean"][offset]) - comparison[f"validation_{mode}_mse"]) < 1e-6
            assert selection["model_sha256"] in serial and f"H={h}\n" in serial
            report["models"][str(h)] = {"cpp_compile": "PASS", "sanitizers": "ASan+UBSan",
                "runtime_saved_input_check": "PASS (128 inputs)", "differential_inputs": len(cases),
                "max_abs_fp32_difference": max_error, "rgb8_match": "EXACT, Dense and CSR",
                "serial_t_and_mse_rows": "PASS", "csr_index_rejection": "PASS", "timing_statistics": "PASS",
                "model_header_sha256": selection["model_sha256"]}
            print(f"PASS: H={h}, 417 Dense/CSR input pairs, MSE rows, serial t, ASan/UBSan", flush=True)
    (HERE / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
