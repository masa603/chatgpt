"""Host verification of the eight single-model builds; never measures Arduino memory."""

from concurrent.futures import ThreadPoolExecutor
import csv
import io
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / 'pruning')]
from experiment_today.verify_host import ARDUINO_STUB
from export_csr_arduino import dense_layer_references
from threshold_tools import load_model, prune

RUNNER = r'''
#include "Arduino.h"
#include <cassert>
#include <cmath>
#include <iostream>
MockSerial Serial;
void delay(unsigned long) {}
#include "memory_probe_v10.ino"
int main(int argc,char **argv) {
  assert(argc==2 && memory_model::HIDDEN_SIZE==PROBE_H);
  if(std::string(argv[1])=="predict") {
    float input[3],output[3];std::cout<<std::setprecision(9);
    while(std::cin>>input[0]>>input[1]>>input[2]) {
      modelPredict(input,output);
      for(int c=0;c<3;c++)std::cout<<output[c]<<' ';
      for(int c=0;c<3;c++) {
        double v=output[c];if(v<0)v=0;if(v>1)v=1;
        std::cout<<static_cast<int>(v*255+.5)<<(c==2?'\n':' ');
      }
    }
  } else {
    setup();loop();
    for(int c=0;c<3;c++)assert(std::isfinite(probeOutput[c]));
    std::cout<<Serial.output.str();
  }
}
'''


def run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f'{command}\n{result.stderr}')
    return result.stdout


def verify():
    folder = HERE / 'memory_probe_v10'
    metadata = json.loads((folder / 'conditions.json').read_text())
    records = []
    paths = [HERE / 'training135.csv', HERE / 'heldout9.csv', ROOT / 'pruning/validation.csv']
    paths += sorted((HERE / 'fresh_readings').glob('capture_*/inputs.csv'))
    for path in paths:
        with path.open(newline='') as f:
            records.extend(csv.DictReader(f))
    cases = np.concatenate([
        np.array([[r[f'input_{c}'] for c in 'rgb'] for r in records], np.float32),
        np.random.default_rng(42).random((256, 3), dtype=np.float32),
        np.array(list(itertools.product([0, 1], repeat=3)), np.float32)])
    assert len(cases) == 626
    input_text = '\n'.join(' '.join(format(float(v), '.9g') for v in x) for x in cases) + '\n'
    report = {
        'scope': 'Host tests only. No host executable size is used as Arduino Flash or RAM.',
        'arduino_single_model_measurements': 'PENDING_MAC_ARDUINO_BUILD',
        'declared_prediction_work_array_bytes': 800,
        'runtime_stack_peak': 'NOT_MEASURED',
        'conditions': []}
    with tempfile.TemporaryDirectory(prefix='v10-single-memory-') as name:
        temp = Path(name)
        (temp / 'Arduino.h').write_text(ARDUINO_STUB)
        (temp / 'main.cpp').write_text(RUNNER)
        includes = [f'-I{temp}', f'-I{folder}']

        def condition_check(condition):
            h, mode = condition['H'], condition['PROBE_CSR']
            defines = [f'-DPROBE_H={h}', f'-DPROBE_CSR={mode}']
            source = str(temp / 'main.cpp')
            exe = temp / f'h{h}_{mode}'
            run(['g++', '-std=c++11', '-O2', '-ffp-contract=off', '-fsanitize=address,undefined',
                 '-fno-omit-frame-pointer', *defines, *includes, source, '-o', str(exe)])
            preprocessed = run(['g++', '-std=c++11', '-E', '-P', *defines, *includes, source])
            assert ('const float weight_1[' in preprocessed) == (mode == 0)
            assert ('const float values_1[' in preprocessed) == (mode == 1)
            for other in metadata['conditions'][::2]:
                assert (other['model_source_sha256'] in preprocessed) == (other['H'] == h)
            for absent in ['BENCHMARK_INPUTS', 'CHECK_INPUTS', 'TRAINING_INPUTS', 'REFERENCE_RGB']:
                assert absent not in preprocessed
            model = load_model(HERE / f'models/h{h}/model_parameters.h')
            if mode:
                model = prune(model, condition['threshold'])
            actual = np.loadtxt(io.StringIO(run([str(exe), 'predict'], input=input_text)))
            expected = dense_layer_references(model, cases)[-1]
            assert actual.shape == (626, 6)
            assert np.allclose(actual[:, :3], expected, rtol=1e-6, atol=1e-6)
            rgb8 = np.floor(np.clip(expected.astype(float), 0, 1) * 255 + .5).astype(int)
            assert np.array_equal(actual[:, 3:], rgb8)
            smoke = run([str(exe), 'smoke'])
            for text in [f'H={h}\n', f'FORMAT_ID={mode}\n', condition['model_source_sha256'],
                         f"MODEL_ARRAY_BYTES={condition['model_array_bytes']}\n"]:
                assert text in smoke
            # Check retention under link-time optimization without treating host sizes as device data.
            optimized = temp / f'optimized_h{h}_{mode}'
            run(['g++', '-std=c++11', '-Os', '-flto', '-ffp-contract=off', *defines, *includes,
                 source, '-o', str(optimized)])
            run([str(optimized), 'smoke'])
            symbols = run(['nm', '-C', str(optimized)])
            for symbol in ['modelPredict(', 'probeInput', 'probeOutput', 'memory_model::bias_1']:
                assert symbol in symbols
            assert ('memory_model::weight_1' in symbols) == (mode == 0)
            assert ('memory_model::values_1' in symbols) == (mode == 1)
            return dict(condition, cpp11_asan_ubsan='PASS', optimized_volatile_retention='PASS',
                        exactly_one_model_and_format='PASS', differential_inputs=len(cases),
                        rgb8_exact_match=True,
                        max_abs_fp32_error=float(np.max(np.abs(actual[:, :3] - expected))),
                        arduino_flash_bytes=None, arduino_static_ram_bytes=None)

        with ThreadPoolExecutor(max_workers=3) as pool:
            report['conditions'] = list(pool.map(condition_check, metadata['conditions']))
        for h, mode in [(50, 0), (40, 2)]:
            invalid = subprocess.run(['g++', '-std=c++11', '-E', f'-DPROBE_H={h}',
                f'-DPROBE_CSR={mode}', *includes, str(temp / 'main.cpp')], capture_output=True)
            assert invalid.returncode != 0
        report['invalid_selections_rejected'] = True
    (HERE / 'memory_probe_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS:8 single-model selections,626 FP32/RGB8 cases each,ASan/UBSan,optimized retention; Arduino memory remains unmeasured.')


if __name__ == '__main__':
    verify()
