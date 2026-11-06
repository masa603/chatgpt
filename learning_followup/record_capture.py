"""Verify and archive an actual COLOR_RETRAIN_V3 log, including pasted escapes."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'pruning'))
from threshold_tools import load_model
from export_csr_arduino import dense_layer_references


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def record(path, received_date):
    path = Path(path).resolve()
    output = path.parent
    original = path.read_text()
    # The submitted message escapes line boundaries and a few Markdown markers.
    text = '\n'.join(line.strip().strip('\\').strip() for line in original.splitlines()) + '\n'
    lines = text.splitlines()
    assert text.count('========== TRAINED CAPTURE START ==========') == 1
    assert text.count('=========== TRAINED CAPTURE END ===========') == 1
    required = ['FIRMWARE=COLOR_RETRAIN_V3', 'INPUT_SOURCE=NEW_PHYSICAL_SENSOR_READINGS',
                'FORMAT=DENSE_FP32_ONLY; NO_PRUNING_IN_THIS_STAGE',
                'CHART_ROLE=LEARNING_3X3_REPEAT; NOT_INDEPENDENT_FINAL_CHART',
                'TRAINING_ROWS=36', 'MSE_GOAL_THIS_READ=300', '# size,3,3']
    assert all(line in lines for line in required)
    data_hash = hashlib.sha256((HERE / 'data_selected.csv').read_bytes()).hexdigest()
    assert f'TRAINING_DATA_SHA256={data_hash}' in lines
    with (HERE / 'data_all60.csv').open(newline='') as f:
        targets = np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in list(csv.DictReader(f))[:9]], np.int32)
    ids = [f'r{r}c{c}' for r in range(1, 4) for c in range(1, 4)]

    def section(header, count):
        start = lines.index(header)
        return list(csv.DictReader(io.StringIO('\n'.join(lines[start:start + count + 1]))))

    raw = section('color_id,sample_completed_ms,raw_r,raw_g,raw_b', 9)
    normalized = section('color_id,input_r,input_g,input_b', 9)
    reported = section('mse_dataset,H,format,colors,mse', 3)
    goals = section('quality_goal,H,max_mse,met_for_this_read', 3)
    assert [r['color_id'] for r in raw] == ids == [r['color_id'] for r in normalized]
    black, white = [np.array(next(line for line in lines if line.startswith(f'# {kind}_rgb,')).split(',')[1:], np.float32) for kind in ['black', 'white']]
    raw_values = np.array([[r[f'raw_{c}'] for c in 'rgb'] for r in raw], np.float32)
    logged_x = np.array([[r[f'input_{c}'] for c in 'rgb'] for r in normalized], np.float32)
    assert np.all(np.isfinite(raw_values)) and np.all(np.isfinite(logged_x))
    assert np.all(np.isfinite(black)) and np.all(np.isfinite(white)) and np.all(white > black)
    assert np.all((logged_x >= 0) & (logged_x <= 1))
    x = np.clip((raw_values - black) / (white - black), 0, 1)
    assert np.allclose(x, logged_x, rtol=1e-6, atol=1e-7)
    timestamps = [int(r['sample_completed_ms']) for r in raw]
    assert timestamps == sorted(timestamps)
    pattern = r'H=(\d+)\nMODEL_SOURCE_SHA256=([^\n]+)\n---------- PPM START ----------\n(.*?)----------- PPM END -----------'
    images = re.findall(pattern, text, re.S)
    assert [int(h) for h, *_ in images] == [40, 60, 100]
    results, patch_errors, all_pixels = [], [], []
    for (h_text, model_hash, ppm), row, goal in zip(images, reported, goals):
        h = int(h_text)
        header = HERE / f'models/h{h}/model_parameters.h'
        assert hashlib.sha256(header.read_bytes()).hexdigest() == model_hash
        tokens = ppm.split()
        assert tokens[:4] == ['P3', '3', '3', '255'] and len(tokens) == 31
        pixels = np.array(tokens[4:], np.int32).reshape(9, 3)
        assert np.all((pixels >= 0) & (pixels <= 255))
        mse = float(np.mean((pixels - targets) ** 2))
        assert (int(row['H']), row['format'], int(row['colors'])) == (h, 'DENSE', 9)
        assert row['mse_dataset'] == 'new_learning_chart_read'
        assert abs(float(row['mse']) - mse) < 1e-6
        assert int(goal['H']) == h and int(goal['max_mse']) == 300
        assert goal['met_for_this_read'] == ('YES' if mse <= 300 else 'NO')
        model = load_model(header)
        expected_raw = dense_layer_references(model, x)[-1]
        expected_pixels = np.floor(np.clip(expected_raw.astype(float), 0, 1) * 255 + .5).astype(int)
        difference = int(np.max(np.abs(expected_pixels - pixels)))
        assert difference <= 1, f'Unexpected H={h} device/host difference: {difference}'
        name = f'h{h}_dense'
        (output / f'{name}.ppm').write_text(ppm)
        Image.fromarray(pixels.astype(np.uint8).reshape(3, 3, 3)).resize((360, 360), Image.Resampling.NEAREST).save(output / f'{name}.png')
        results.append({'capture_key': output.name, 'hidden_size': h, 'format': 'DENSE', 'colors': 9,
                        'mse_device_reported': float(row['mse']), 'mse_recomputed_from_ppm': mse,
                        'mse_goal_max': 300, 'goal_met_this_read': mse <= 300,
                        'pc_replay_max_rgb8_difference': difference,
                        'model_header_sha256': model_hash, 'ppm': f'{name}.ppm'})
        all_pixels.append(pixels)
        for i, (prediction, target) in enumerate(zip(pixels, targets)):
            patch_errors.append({'hidden_size': h, 'color_id': ids[i],
                                 **{f'predicted_{c}8': int(prediction[j]) for j, c in enumerate('rgb')},
                                 **{f'target_{c}8': int(target[j]) for j, c in enumerate('rgb')},
                                 'mse_rgb8': float(np.mean((prediction - target) ** 2))})
    inputs = [{**r, **{k: v for k, v in n.items() if k != 'color_id'},
               **{f'target_{c}8': int(targets[i, j]) for j, c in enumerate('rgb')}}
              for i, (r, n) in enumerate(zip(raw, normalized))]
    rgb = [{'color_id': color, **{f'h{h}_rgb': ' '.join(map(str, pixels[i])) for h, pixels in zip([40, 60, 100], all_pixels)}} for i, color in enumerate(ids)]
    write_csv(output / 'inputs.csv', inputs)
    write_csv(output / 'mse.csv', results)
    write_csv(output / 'patch_errors.csv', patch_errors)
    write_csv(output / 'rgb9.csv', rgb)
    (output / 'normalized_log.txt').write_text(text)
    summary = {'received_date_japan': received_date, 'execution_wall_timestamp': None,
               'firmware': 'COLOR_RETRAIN_V3',
               'capture_counter': int(next(line for line in lines if line.startswith('CAPTURE=')).split('=')[1]),
               'input_source': 'NEW_PHYSICAL_SENSOR_READINGS', 'physical_captures': 1,
               'chart_role': 'Repeated read of the nine-color learning chart, after freezing V3 weights.',
               'training_role_for_v3': 'NOT_USED_FOR_FITTING_OR_CONFIG_SELECTION',
               'independent_unseen_chart_evaluation': 'NOT_PERFORMED',
               'repetition_scope': 'One acquisition evaluated by three Dense models; not three physical repetitions.',
               'startup_self_check_log': 'NOT_SUPPLIED',
               'black_rgb': black.tolist(), 'white_rgb': white.tolist(), 'colors': 9,
               'normalization_check': 'PASS', 'normalization_max_abs_difference': float(np.max(np.abs(x - logged_x))),
               'model_hashes_and_ppm_mse_check': 'PASS',
               'results': results, 'target_source': 'Recovered Notebook display; original PNG not verified.',
               'submitted_log_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'normalization_of_pasted_text': 'Only strip line-boundary backslashes/whitespace; original submitted text retained.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8))
    for ax, image, title in zip(axes, [targets, *all_pixels], ['Reference (recovered)', *[f'H={r["hidden_size"]} Dense\nMSE {r["mse_device_reported"]:.2f}' for r in results]]):
        ax.imshow(image.astype(np.uint8).reshape(3, 3, 3), interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle('COLOR_RETRAIN_V3: one new physical reading, capture 1')
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=160)
    fig.savefig(output / 'comparison.pdf')
    plt.close(fig)
    print('PASS: original log retained, model hashes, normalization, three PPM MSEs and goal flags; actual pixels exported.')
    print(json.dumps(results))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--received-date', required=True)
    args = parser.parse_args()
    record(args.log, args.received_date)
