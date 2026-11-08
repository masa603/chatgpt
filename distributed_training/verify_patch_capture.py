"""Host checks for labelled acquisition. Mock samples are not physical measurements."""

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
sys.path.insert(0, str(ROOT))
from experiment_today.verify_fresh_scanner import STUB, function_body

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'patch_learning_capture'
RUNNER = r'''
#include "Arduino.h"
#include <cassert>
#include <iostream>
MockSerial Serial;
unsigned long clockMs=0;
int leds[3]={0},raw[3]={10,20,30},sensorReads=0;
unsigned long millis() { return clockMs; }
void delay(unsigned long ms) { clockMs+=ms; }
void pinMode(uint8_t,int) {}
int digitalRead(uint8_t) { assert(false);return 0; }
void analogWrite(uint8_t pin,int value) {
  assert(pin==8 || pin==6 || pin==7);
  leds[pin==8?0:pin==6?1:2]=value;
}
int analogRead(uint8_t pin) {
  assert(pin==A0);int channel=-1;
  for(int c=0;c<3;c++) if(leds[c]) { assert(channel==-1 && leds[c]==255);channel=c; }
  assert(channel>=0);++sensorReads;return raw[channel];
}
#include "patch_learning_capture.ino"
void send(const std::string& text) { Serial.feed(text);while(Serial.available()) loop(); }
void setRaw(int r,int g,int b) { raw[0]=r;raw[1]=g;raw[2]=b; }
int main() {
  setup();assert(sensorReads==0);
  send("p\r\nw\r\n");assert(sensorReads==0 && calibrationStep==0);
  send("b\n");assert(calibrationStep==1);
  send("w\n");assert(calibrationStep==0 && sensorReads==60);
  send("b\n");setRaw(210,220,230);send("w\n");
  assert(calibrationStep==2 && calibrationNumber==1 && sensorReads==120);
  send("w\nr\n");assert(sensorReads==120);
  setRaw(110,70,230);send("p\r\n");
  setRaw(5,10,20);send("k\n");
  setRaw(50,70,80);send("g\n");
  setRaw(310,420,530);send("V\n");
  assert(sensorReads==480 && batchNumber==4);
  for(int value:leds) assert(value==0);
  send("n\np\n");assert(sensorReads==480 && calibrationStep==0);
  setRaw(10,20,30);send("b\n");setRaw(210,220,230);send("w\n");
  setRaw(110,120,130);send("g\n");
  assert(calibrationNumber==2 && batchNumber==5 && sensorReads==630);
  std::cout<<Serial.output.str();
}
'''


def main():
    original = (HERE / 'color_retrain_v7/color_retrain_v7.ino').read_text()
    sketch = (SOURCE / 'patch_learning_capture.ino').read_text()
    for name in ('ledOff', 'readAverage', 'readRGB', 'normalizeInput'):
        assert function_body(original, name) == function_body(sketch, name)
    with tempfile.TemporaryDirectory(prefix='patch-collector-host-check-') as name:
        directory = Path(name)
        (directory / 'Arduino.h').write_text(STUB)
        (directory / 'main.cpp').write_text(RUNNER)
        exe = directory / 'check'
        subprocess.run(['g++', '-std=c++11', '-O2', '-ffp-contract=off', '-fsanitize=address,undefined',
                        '-fno-omit-frame-pointer', f'-I{directory}', f'-I{SOURCE}',
                        str(directory / 'main.cpp'), '-o', str(exe)], check=True, capture_output=True, text=True)
        output = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    frames = re.findall(r'========== PATCH LEARNING START ==========\n(.*?)=========== PATCH LEARNING END ===========', output, re.S)
    assert len(frames) == 5
    ids = ['r2c3','r3c1','r3c2','r3c3','r3c2']
    expected_raw = [[110,70,230],[5,10,20],[50,70,80],[310,420,530],[110,120,130]]
    header = 'repeat,sample_completed_ms,raw_r,raw_g,raw_b,input_r,input_g,input_b,unclipped_r,unclipped_g,unclipped_b'
    for i, frame in enumerate(frames):
        assert f'PATCH_ID={ids[i]}\n' in frame
        assert f'CALIBRATION={1 if i<4 else 2}\n' in frame
        assert f'BATCH={i+1}\n' in frame
        lines = frame.splitlines()
        start = lines.index(header)
        records = list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
        assert len(records) == 3
        timestamps = []
        for n, record in enumerate(records,1):
            assert int(record['repeat']) == n
            raw = np.array([float(record[f'raw_{c}']) for c in 'rgb'])
            clipped = np.array([float(record[f'input_{c}']) for c in 'rgb'])
            unclipped = np.array([float(record[f'unclipped_{c}']) for c in 'rgb'])
            assert np.array_equal(raw, expected_raw[i])
            expected = (raw-[10,20,30])/200
            assert np.allclose(unclipped, expected, rtol=0, atol=1e-7)
            assert np.allclose(clipped, np.clip(expected,0,1), rtol=0, atol=1e-7)
            timestamps.append(int(record['sample_completed_ms']))
        assert all(b>a for a,b in zip(timestamps,timestamps[1:]))
    report = {'scope':'Host mock acquisition only; no physical samples generated.',
              'sensor_and_clipped_normalization_bodies_identical_to_v7': True,
              'cpp11_asan_ubsan':'PASS','invalid_calibration_and_state_guards':'PASS',
              'label_mapping_purple_black_gray_white':'PASS','three_samples_per_command':'PASS',
              'raw_and_unclipped_and_clipped_inputs':'PASS','negative_and_over_one_inputs':'PASS',
              'serial_newlines_uppercase_n_restart':'PASS','led_off_after_batch':'PASS',
              'models_changed_or_trained':False,'new_physical_patch_samples_received':False}
    (SOURCE / 'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: unchanged acquisition, state guards, labels, 3 samples/command, raw/clipped/unclipped values, n restart; C++11 ASan/UBSan. Mock only.')


if __name__ == '__main__':
    main()
