"""Check V4 weights, sensor path, reference RGB and the actual teacher calculator."""

import argparse
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'pruning'), str(ROOT / 'retraining')]
from experiment_today.verify_fresh_scanner import STUB, function_body
from learning_followup.verify_scanner import RUNNER
from threshold_tools import load_model
from train_current33 import ColorNet, model_arrays, forward_fp32


def verify(calculator):
    reference = ROOT / 'learning_followup/reference_audit/provided/result.ppm'
    reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    tokens = reference.read_text().split()
    assert tokens[:4] == ['P3', '3', '3', '255'] and len(tokens) == 31
    target = np.array(tokens[4:], int).reshape(9, 3)
    data = HERE / 'training45.csv'
    with data.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 45 and len({r['group'] for r in rows}) == 5
    for i, row in enumerate(rows):
        assert [int(row[f'target_{c}8']) for c in 'rgb'] == target[i % 9].tolist()
    x = np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows], np.float32)
    y = np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in rows], float)
    original = (ROOT / 'hdjsk/hdjsk.ino').read_text()
    scanner = (HERE / 'color_retrain_v4/color_retrain_v4.ino').read_text()
    header = (HERE / 'color_retrain_v4/trained_models_v4.h').read_text()
    match = re.search(r'REFERENCE_RGB\[9\]\[3\]\s*=\s*\{([\s\S]*?)\};', header)
    assert match
    assert np.array_equal(np.array(re.findall(r'\d+', match[1]), int).reshape(9, 3), target)
    assert reference_hash in header
    for name in ['ledOff', 'readRGB', 'readAverage', 'normalizeInput', 'outputTo255', 'handleCalibration']:
        assert function_body(original, name) == function_body(scanner, name)
    models = []
    learning_mses = []
    for h in [40, 60, 100]:
        directory = HERE / f'models/h{h}'
        summary = json.loads((directory / 'summary.json').read_text())
        source = directory / 'model_parameters.h'
        assert summary['model_header_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert summary['training_data_sha256'] == hashlib.sha256(data.read_bytes()).hexdigest()
        assert summary['reference_ppm_sha256'] == reference_hash
        layers = load_model(source)
        models.append(layers)
        archive = np.load(directory / 'model_arrays.npz', allow_pickle=False)
        for i, (w, b) in enumerate(layers, 1):
            assert np.array_equal(w, archive[f'weight_{i}']) and np.array_equal(b, archive[f'bias_{i}'])
        checkpoint = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
        model = ColorNet(h)
        model.load_state_dict(checkpoint['state_dict'])
        assert checkpoint['reference_sha256'] == reference_hash
        assert all(np.array_equal(a, b) for pair1, pair2 in zip(model_arrays(model), layers) for a, b in zip(pair1, pair2))
        pixels = np.floor(np.clip(forward_fp32(layers, x).astype(float), 0, 1) * 255 + .5)
        mse = float(np.mean((pixels - y) ** 2))
        assert mse == summary['training_mse_rgb8']
        learning_mses.append({'hidden_size': h, 'mse': mse})
    cases = np.concatenate([x, np.array(list(itertools.product([0., 1.], repeat=3)), np.float32), np.random.default_rng(42).random((256, 3), dtype=np.float32)])
    runner = RUNNER.replace('color_retrain_v3.ino', 'color_retrain_v4.ino').replace('COLOR_RETRAIN_V3', 'COLOR_RETRAIN_V4')
    with tempfile.TemporaryDirectory(prefix='reference-v4-check-') as name:
        temp = Path(name)
        (temp / 'Arduino.h').write_text(STUB)
        (temp / 'main.cpp').write_text(runner)
        exe = temp / 'check'
        subprocess.run(['g++', '-std=c++11', '-O2', '-ffp-contract=off', '-fsanitize=address,undefined', '-fno-omit-frame-pointer', f'-I{HERE / "color_retrain_v4"}', f'-I{temp}', str(temp / 'main.cpp'), '-o', str(exe)], check=True, capture_output=True, text=True)
        result = subprocess.run([str(exe), 'predict'], input='\n'.join(' '.join(format(float(v), '.9g') for v in row) for row in cases) + '\n', check=True, capture_output=True, text=True)
        predictions = np.loadtxt(io.StringIO(result.stdout))
        max_error = 0
        for index, layers in enumerate(models):
            expected = forward_fp32(layers, cases)
            max_error = max(max_error, float(np.max(np.abs(predictions[:, 3 * index:3 * index + 3] - expected))))
            assert np.allclose(predictions[:, 3 * index:3 * index + 3], expected, atol=1e-6, rtol=1e-6)
            assert np.array_equal(predictions[:, 9 + 3 * index:12 + 3 * index], np.floor(np.clip(expected.astype(float), 0, 1) * 255 + .5))
        serial = subprocess.run([str(exe), 'capture'], check=True, capture_output=True, text=True).stdout
        frames = re.findall(r'========== CORRECTED CAPTURE START ==========\n(.*?)=========== CORRECTED CAPTURE END ===========', serial, re.S)
        assert len(frames) == 2
        patches = np.array([[184,55,62], [98,89,69], [27,39,74], [230,240,150], [66,138,151], [74,52,76], [0,0,0], [110,120,130], [400,400,400]], np.float32)
        teacher_checks = []
        for repeat, frame in enumerate(frames):
            assert 'FIRMWARE=COLOR_RETRAIN_V4' in frame and f'CAPTURE={repeat+1}\n' in frame
            assert f'REFERENCE_FILE_SHA256={reference_hash}\n' in frame
            source = patches if repeat == 0 else patches[::-1]
            mock_x = np.clip((source - np.array([10,20,30], np.float32)) / np.float32(200), 0, 1)
            images = re.findall(r'---------- PPM START ----------\n(.*?)----------- PPM END -----------', frame, re.S)
            lines = frame.splitlines()
            start = lines.index('mse_dataset,H,format,colors,mse')
            mses = list(csv.DictReader(io.StringIO('\n'.join(lines[start:start + 4]))))
            start = lines.index('quality_goal,H,max_mse,met_for_this_read')
            goals = list(csv.DictReader(io.StringIO('\n'.join(lines[start:start + 4]))))
            assert len(images) == len(mses) == len(goals) == 3
            for index, (image, layers) in enumerate(zip(images, models)):
                tokens = image.split()
                assert tokens[:4] == ['P3', '3', '3', '255']
                pixels = np.array(tokens[4:], int).reshape(9, 3)
                expected = np.floor(np.clip(forward_fp32(layers, mock_x).astype(float), 0, 1) * 255 + .5)
                assert np.array_equal(pixels, expected)
                mse = float(np.mean((pixels - target) ** 2))
                assert abs(float(mses[index]['mse']) - mse) < 1e-6
                assert int(mses[index]['H']) == [40,60,100][index]
                assert goals[index]['met_for_this_read'] == ('YES' if mse <= 300 else 'NO')
                mock_ppm = temp / 'mock.ppm'
                mock_ppm.write_text(image)
                completed = subprocess.run(['node', str(ROOT / 'learning_followup/reference_audit/run_teacher_calculator.cjs'), str(calculator), str(reference), str(mock_ppm)], check=True, capture_output=True, text=True)
                teacher = json.loads(completed.stdout)
                assert teacher['valid_full_image_comparison'] and teacher['compared_pixels'] == 9
                assert abs(teacher['mse_displayed'] - mse) <= .005001
                teacher_checks.append({'mock_capture': repeat + 1, 'hidden_size': [40,60,100][index], 'firmware_mse': mse, 'teacher_site_displayed_mse': teacher['mse_displayed']})
    report = {'scope': 'PC verification and mock captures, not a new physical measurement.',
              'training_targets_equal_provided_reference': 'PASS for all 45 rows',
              'embedded_reference_rgb_equal_provided_file': 'PASS for all 27 components',
              'reference_ppm_sha256': reference_hash,
              'provided_teacher_calculator_sha256': hashlib.sha256(Path(calculator).read_bytes()).hexdigest(),
              'checkpoint_npz_header_equality': 'PASS', 'recomputed_training_mse': learning_mses,
              'original_sensor_and_normalization_pipeline': 'UNCHANGED',
              'cpp11_asan_ubsan': 'PASS', 'differential_inputs': len(cases),
              'max_abs_fp32_difference': max_error, 'all_rgb8_components_equal': True,
              'two_mock_captures_and_n_restart': 'PASS', 'original_teacher_html_mse_agreement': teacher_checks,
              'new_physical_mse_v4': 'NOT_YET_RECEIVED'}
    (HERE / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: 45 labels and embedded reference match provided PPM; trained weights and C++ match; both mock captures agree with original teacher HTML.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calculator', type=Path, required=True)
    verify(parser.parse_args().calculator)
