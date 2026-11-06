"""Test six-model fresh acquisition on a host mock; no physical readings are invented."""

import csv
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pruning"))
from export_csr_arduino import dense_layer_references
from threshold_tools import load_model, prune

STUB = r'''
#pragma once
#include <cstdint>
#include <deque>
#include <iomanip>
#include <sstream>
#include <string>
#define A0 14
#define LOW 0
#define HIGH 1
#define OUTPUT 1
#define INPUT_PULLUP 2
unsigned long millis();
void delay(unsigned long);
void pinMode(uint8_t,int);
int digitalRead(uint8_t);
int analogRead(uint8_t);
void analogWrite(uint8_t,int);
struct MockSerial {
  std::deque<char> input;
  std::ostringstream output;
  void begin(unsigned long) {}
  template<class T> void print(T v,int=10) { output<<v; }
  void print(uint8_t v,int=10) { output<<static_cast<unsigned>(v); }
  void print(float v,int p=2) { output<<std::fixed<<std::setprecision(p)<<v; }
  void print(double v,int p=2) { output<<std::fixed<<std::setprecision(p)<<v; }
  template<class T> void println(T v,int p=2) { print(v,p); output<<'\n'; }
  void println() { output<<'\n'; }
  int available() { return input.size(); }
  int read() { if(input.empty()) return -1; int v=input.front();input.pop_front();return v; }
  void feed(const std::string& s) { for(char c:s) input.push_back(c); }
};
extern MockSerial Serial;
'''

RUNNER = r'''
#include "Arduino.h"
#include <cassert>
#include <iostream>
MockSerial Serial;
unsigned long clockMs=0;
int leds[3]={0}, raw[3]={10,20,30}, sensorReads=0, buttonReads=0;
unsigned long millis() { return clockMs; }
void delay(unsigned long ms) { clockMs+=ms; }
void pinMode(uint8_t,int) {}
int digitalRead(uint8_t) { buttonReads++; return LOW; }
void analogWrite(uint8_t pin,int value) {
  int ch=pin==8?0:pin==6?1:2; assert(pin==8 || pin==6 || pin==7); leds[ch]=value;
}
int analogRead(uint8_t pin) {
  assert(pin==A0); int channel=-1;
  for(int c=0;c<3;c++) if(leds[c]) { assert(channel==-1 && leds[c]==255); channel=c; }
  assert(channel>=0); sensorReads++; return raw[channel];
}
#include "fresh_scan_all.ino"
void send(const std::string& s) { Serial.feed(s); while(Serial.available()) loop(); }
int main(int argc,char** argv) {
  assert(argc==2);
  if(std::string(argv[1])=="predict") {
    std::cout<<std::setprecision(9); float input[3],output[6][3];
    while(std::cin>>input[0]>>input[1]>>input[2]) {
      fresh_models::inferAll(input,output);
      for(int m=0;m<6;m++) for(int c=0;c<3;c++) std::cout<<output[m][c]<<' ';
      for(int m=0;m<6;m++) for(int c=0;c<3;c++) std::cout<<outputTo255(output[m][c])<<(m==5 && c==2?'\n':' ');
    }
    return 0;
  }
  setup(); assert(imageSize==3 && sensorReads==0 && buttonReads==0);
  assert(Serial.output.str().find("ALL MODELS CHECK: PASS")!=std::string::npos);
  assert(Serial.output.str().find("画像サイズを入力してください")==std::string::npos);
  send("r\r\n"); assert(sensorReads==0);
  int patches[9][3]={{184,55,62},{98,89,69},{27,39,74},{230,240,150},{66,138,151},{74,52,76},{0,0,0},{110,120,130},{400,400,400}};
  for(int repeat=0;repeat<2;repeat++) {
    if(repeat) { send("n\r\n"); assert(captureNumber==2 && calibrationStep==0 && currentRow==0 && currentColumn==0 && !imageFinished); }
    raw[0]=10;raw[1]=20;raw[2]=30; send("b\r\n"); assert(calibrationStep==1);
    raw[0]=210;raw[1]=220;raw[2]=230; send("w\r\n"); assert(calibrationStep==2);
    for(int i=0;i<9;i++) {
      int index=repeat?8-i:i;
      for(int c=0;c<3;c++) raw[c]=patches[index][c];
      send("r\r\n");
    }
    assert(imageFinished && sensorReads==(repeat+1)*330);
    send("r\r\n"); assert(sensorReads==(repeat+1)*330);
  }
  std::cout<<Serial.output.str();
}
'''


def function_body(text, name):
    start = re.search(rf"^(?:void|float|int|bool) {name}\(", text, re.M).start()
    first = text.index("{", start)
    last, depth = first + 1, 1
    while depth:
        depth += (text[last] == "{") - (text[last] == "}")
        last += 1
    return text[first:last]


def main(base=None):
    base = HERE if base is None else Path(base)
    source = base / "fresh_scan_all"
    original = (ROOT / "hdjsk/hdjsk.ino").read_text()
    scanner = (source / "fresh_scan_all.ino").read_text()
    for name in ("ledOff", "readAverage", "readRGB", "normalizeInput", "outputTo255", "handleCalibration"):
        assert function_body(original, name) == function_body(scanner, name), name
    assert "#define SAMPLE_COUNT 10" in scanner and "#define RGB_FLASH_DELAY 10" in scanner
    layers = []
    for h in (40, 60, 100):
        dense = load_model(ROOT / f"retraining/results/h{h}/model_parameters.h")
        selection = json.loads((base / f"selection/h{h}/summary.json").read_text())
        layers.extend([dense, prune(dense, selection["selected"]["threshold"])])
    records = []
    for path in [ROOT / "retraining/data/current33.csv", ROOT / "pruning/validation.csv"]:
        with path.open(newline="") as f:
            records.extend(csv.DictReader(f))
    cases = np.concatenate([np.array([[r[f"input_{c}"] for c in "rgb"] for r in records], np.float32),
                            np.random.default_rng(42).random((256, 3), dtype=np.float32)])
    reference = np.array([[r[f"target_{c}8"] for c in "rgb"] for r in records[:9]], float)
    with tempfile.TemporaryDirectory(prefix="fresh-scanner-check-") as name:
        temporary = Path(name)
        (temporary / "Arduino.h").write_text(STUB)
        (temporary / "main.cpp").write_text(RUNNER)
        exe = temporary / "check"
        subprocess.run(["g++", "-std=c++11", "-O2", "-ffp-contract=off", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                        f"-I{source}", f"-I{temporary}", str(temporary / "main.cpp"), "-o", str(exe)], check=True, capture_output=True, text=True)
        result = subprocess.run([str(exe), "predict"], input="\n".join(" ".join(format(float(v), ".9g") for v in row) for row in cases) + "\n",
                                check=True, capture_output=True, text=True)
        predicted = np.loadtxt(io.StringIO(result.stdout))
        max_error = 0.0
        for m, model in enumerate(layers):
            raw = dense_layer_references(model, cases)[-1]
            max_error = max(max_error, float(np.max(np.abs(predicted[:, m * 3:m * 3 + 3] - raw))))
            assert np.allclose(predicted[:, m * 3:m * 3 + 3], raw, atol=1e-6, rtol=1e-6)
            expected = np.floor(np.clip(raw.astype(np.float64), 0, 1) * 255 + .5)
            assert np.array_equal(predicted[:, 18 + m * 3:21 + m * 3], expected)
        serial = subprocess.run([str(exe), "scan"], check=True, capture_output=True, text=True).stdout
    frames = re.findall(r"========== FRESH CAPTURE START ==========\n(.*?)=========== FRESH CAPTURE END ===========", serial, re.S)
    assert len(frames) == 2
    patches = np.array([[184,55,62],[98,89,69],[27,39,74],[230,240,150],[66,138,151],[74,52,76],[0,0,0],[110,120,130],[400,400,400]], np.float32)
    for repeat, frame in enumerate(frames, 1):
        assert f"CAPTURE={repeat}\n" in frame
        assert "INPUT_SOURCE=NEW_PHYSICAL_SENSOR_READINGS" in frame
        raw_patches = patches if repeat == 1 else patches[::-1]
        inputs = np.clip((raw_patches - np.array([10,20,30], np.float32)) / np.float32(200), 0, 1)
        images = re.findall(r"---------- PPM START ----------\n(.*?)----------- PPM END -----------", frame, re.S)
        assert len(images) == 6
        lines = frame.splitlines()
        start = lines.index("fresh_mse_dataset,H,format,colors,mse")
        mses = list(csv.DictReader(io.StringIO("\n".join(lines[start:start + 7]))))
        for m, (image_text, model) in enumerate(zip(images, layers)):
            tokens = image_text.split()
            assert tokens[:4] == ["P3", "3", "3", "255"]
            pixels = np.array(tokens[4:], int).reshape(9, 3)
            raw = dense_layer_references(model, inputs)[-1]
            expected = np.floor(np.clip(raw.astype(np.float64), 0, 1) * 255 + .5)
            assert np.array_equal(pixels, expected)
            assert int(mses[m]["H"]) == (40,40,60,60,100,100)[m]
            assert mses[m]["format"] == ("DENSE" if m % 2 == 0 else "CSR")
            assert abs(float(mses[m]["mse"]) - np.mean((pixels - reference) ** 2)) < 1e-6
        summary = re.search(r"color_id,input_r,input_g,input_b\n(.*?)----------- NN INPUT END", frame, re.S)[1]
        values = np.array([[float(v) for v in row.split(",")[1:]] for row in summary.strip().splitlines()])
        assert np.allclose(values, inputs, atol=1e-8, rtol=1e-7)
    report = {"scope": "Host C++11 ASan/UBSan tests only; no actual sensor result or physical MSE fabricated.",
              "source_pipeline": "Acquisition, normalization, calibration and RGB conversion unchanged from original scanner.",
              "models": [40,60,100], "formats": ["Dense", "CSR"], "differential_input_count": len(cases),
              "max_abs_output_difference": max_error, "rgb8_predictions": "EXACT for all six models",
              "startup_checks": "PASS for all three frozen model pairs",
              "mock_capture": "Two complete 3x3 acquisitions, black/white calibration, raw channel values, all six PPMs, MSE rows, NN input summary, n restart and no extra reads after completion PASS.",
              "arduino_hardware_check": "PENDING", "physical_repeat_readings_received": 0}
    (base / "fresh_scan_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: six models x 417 inputs; two fresh-scanner mock captures and all PPM/MSE/NN rows; ASan/UBSan.")


if __name__ == "__main__":
    main()
