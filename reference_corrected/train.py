"""Retrain from measured inputs using the actual provided reference PPM targets."""

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'learning_followup'), str(ROOT / 'retraining')]
from train_multiread import fit, cv_worker, write_csv
from train_current33 import export_header

CONFIG = {'name': 'provided_rgb45_mix_noise', 'mode': 'chart', 'mix': True, 'noise': .01}
REFERENCE = ROOT / 'learning_followup/reference_audit/provided/result.ppm'


def read_reference():
    tokens = REFERENCE.read_text().split()
    assert tokens[:4] == ['P3', '3', '3', '255'] and len(tokens) == 31
    values = np.array(tokens[4:], dtype=np.int32).reshape(9, 3)
    assert np.all((values >= 0) & (values <= 255))
    return values


def prepare():
    target = read_reference()
    with (ROOT / 'learning_followup/data_selected.csv').open(newline='') as f:
        original = list(csv.DictReader(f))
    assert len(original) == 36
    ids = [f'r{r}c{c}' for r in range(1, 4) for c in range(1, 4)]
    rows = []
    for index, row in enumerate(original):
        color = index % 9
        expected_suffix = str(color + 1) if row['group'] == 'original_3x3' else ids[color]
        assert row['sample_id'].endswith('_' + expected_suffix)
        rows.append({'sample_id': row['sample_id'], 'group': row['group'], 'color_id': ids[color],
                     **{f'input_{c}': row[f'input_{c}'] for c in 'rgb'},
                     **{f'target_{c}8': int(target[color, j]) for j, c in enumerate('rgb')},
                     **{f'previous_target_{c}8': row[f'target_{c}8'] for c in 'rgb'}})
    source = ROOT / 'learning_followup/physical_readings/capture_01'
    with (source / 'inputs.csv').open(newline='') as f:
        latest = list(csv.DictReader(f))
    summary = json.loads((source / 'summary.json').read_text())
    raw = np.array([[r[f'raw_{c}'] for c in 'rgb'] for r in latest], np.float32)
    black, white = [np.array(summary[f'{kind}_rgb'], np.float32) for kind in ['black', 'white']]
    inputs = np.clip((raw - black) / (white - black), 0, 1)
    assert [r['color_id'] for r in latest] == ids
    for i, row in enumerate(latest):
        rows.append({'sample_id': 'v3_capture01_' + ids[i], 'group': 'v3_capture01', 'color_id': ids[i],
                     **{f'input_{c}': format(float(inputs[i, j]), '.9g') for j, c in enumerate('rgb')},
                     **{f'target_{c}8': int(target[i, j]) for j, c in enumerate('rgb')},
                     **{f'previous_target_{c}8': row[f'target_{c}8'] for c in 'rgb'}})
    assert len(rows) == 45 and len({r['group'] for r in rows}) == 5
    write_csv(HERE / 'training45.csv', rows)
    write_csv(HERE / 'reference_rgb.csv', [{'color_id': color, **{f'target_{c}8': int(target[i, j]) for j, c in enumerate('rgb')}} for i, color in enumerate(ids)])
    (HERE / 'provenance.json').write_text(json.dumps({
        'reference_file': str(REFERENCE.relative_to(ROOT)),
        'reference_sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        'reference_dimensions': [3, 3], 'reference_maxval': 255,
        'reference_source': 'User supplied result.ppm as the reference used in teacher MSE calculator.',
        'training_rows': 45, 'distinct_colors': 9, 'reading_groups': 5,
        'prior36_source': 'learning_followup/data_selected.csv',
        'prior36_source_sha256': hashlib.sha256((ROOT / 'learning_followup/data_selected.csv').read_bytes()).hexdigest(),
        'additional9_source': 'learning_followup/physical_readings/capture_01/submitted_log.txt',
        'additional9_source_sha256': hashlib.sha256((source / 'submitted_log.txt').read_bytes()).hexdigest(),
        'labels_changed': 'All nine RGB targets now come directly from provided result.ppm. Old target columns retained for audit. Original datasets and weights are unchanged.',
        'role_change': 'V3 capture01 is now part of V4 fitting/development. It is not a prospective test for V4. Another new physical reading is required.',
        'scope': 'Adaptation to the same 3x3 nine-color chart, not the original 33 colors or an unseen chart.',
        'configuration': CONFIG, 'configuration_selection': 'Fixed previous mix/noise method; no new hyperparameter search.',
        'development_validation': 'Five whole-reading holdouts with H60 and 10000 epochs; development only, not independent final evaluation.',
    }, indent=2) + '\n')
    return rows


def train_final(task):
    rows, h = task
    output = HERE / f'models/h{h}'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'model_parameters.h').exists():
        raise FileExistsError('Frozen corrected model exists; use a new experiment directory.')
    model, layers, mse, seconds = fit(rows, h, CONFIG, 50000)
    data_hash = hashlib.sha256((HERE / 'training45.csv').read_bytes()).hexdigest()
    header = output / 'model_parameters.h'
    export_header(layers, h, data_hash, 42, 50000, header)
    header.write_text(header.read_text().replace('re-trained on the current 33 saved learning inputs.',
                     'trained on 45 measured inputs with verified provided result.ppm targets.'))
    np.savez(output / 'model_arrays.npz', **{f'{kind}_{i}': value for i, (w, b) in enumerate(layers, 1) for kind, value in [('weight', w), ('bias', b)]})
    torch.save({'hidden_size': h, 'epochs': 50000, 'state_dict': model.state_dict(),
                'config': CONFIG, 'data_sha256': data_hash,
                'reference_sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest()}, output / 'checkpoint.pt')
    report = {'hidden_size': h, 'config': CONFIG, 'training_rows': 45, 'distinct_colors': 9,
              'epochs': 50000, 'seed': 42, 'optimizer': 'Adam; cosine LR 0.003 to 0.00003',
              'l1_weights_only': 1e-6, 'training_mse_rgb8': mse, 'wall_seconds': seconds,
              'training_data_sha256': data_hash, 'reference_ppm_sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
              'model_header_sha256': hashlib.sha256(header.read_bytes()).hexdigest(),
              'scope': 'Training fit only. V3 capture01 now used for training. No prospective V4 physical reading received.'}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    if (HERE / 'provenance.json').exists():
        raise FileExistsError('Corrected training already started; do not overwrite frozen experiment.')
    rows = prepare()
    finals = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(train_final, (rows, h)) for h in [40, 60, 100]]):
            report = future.result()
            finals.append(report)
            print('FINAL', json.dumps(report), flush=True)
    write_csv(HERE / 'training_summary.csv', [{'hidden_size': r['hidden_size'], 'training_rows': 45,
              'training_mse_rgb8': r['training_mse_rgb8'], 'new_physical_mse': 'PENDING',
              'reference_ppm_sha256': r['reference_ppm_sha256']} for r in sorted(finals, key=lambda r: r['hidden_size'])])
    groups = list(dict.fromkeys(r['group'] for r in rows))
    tasks = [(rows, CONFIG, group, 10000) for group in groups]
    development = []
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(cv_worker, task) for task in tasks]):
            report = future.result()
            development.append(report)
            print('DEVELOPMENT', json.dumps(report), flush=True)
    write_csv(HERE / 'development_holdouts.csv', sorted(development, key=lambda r: r['held_out_reading']))
    (HERE / 'development_summary.json').write_text(json.dumps({
        'hidden_size': 60, 'reading_groups': 5, 'epochs_per_fold': 10000,
        'mean_mse': float(np.mean([r['held_out_mse'] for r in development])),
        'worst_mse': max(r['held_out_mse'] for r in development),
        'scope': 'Development cross-validation only. A new physical reading after freeze is still required.',
        'config_used_for_final_models': CONFIG,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
