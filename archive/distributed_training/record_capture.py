"""Archive an actual V7 physical log and verify against the provided reference."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
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


def record(path, received_date, calculator=None):
    path = Path(path).resolve()
    directory = path.parent
    text = '\n'.join(line.strip().strip('\\').strip() for line in path.read_text().splitlines()) + '\n'
    lines = text.splitlines()
    assert text.count('========== DISTRIBUTED CAPTURE START ==========') == 1
    assert text.count('=========== DISTRIBUTED CAPTURE END ===========') == 1
    required = ['FIRMWARE=COLOR_RETRAIN_V7', 'INPUT_SOURCE=NEW_PHYSICAL_SENSOR_READINGS',
                'FORMAT=DENSE_FP32_ONLY; NO_PRUNING_IN_THIS_STAGE',
                'CHART_ROLE=LEARNING_3X3_REPEAT; NOT_INDEPENDENT_FINAL_CHART',
                'REFERENCE_SOURCE=DISTRIBUTED_NOTEBOOK_CMYK_DISPLAY_RECOVERED', 'TRAINING_ROWS=102', '# size,3,3']
    assert all(line in lines for line in required)
    reference = HERE / 'references/reference_image1_cmyk_3x3.ppm'
    reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    assert f'REFERENCE_FILE_SHA256={reference_hash}' in lines
    data_hash = hashlib.sha256((HERE / 'training102.csv').read_bytes()).hexdigest()
    assert f'TRAINING_DATA_SHA256={data_hash}' in lines
    tokens = reference.read_text().split()
    assert tokens[:4] == ['P3','3','3','255'] and len(tokens) == 31
    targets = np.array(tokens[4:], int).reshape(9,3)
    ids = [f'r{r}c{c}' for r in range(1,4) for c in range(1,4)]

    def section(header, count):
        start = lines.index(header)
        return list(csv.DictReader(io.StringIO('\n'.join(lines[start:start+count+1]))))

    raw = section('color_id,sample_completed_ms,raw_r,raw_g,raw_b',9)
    normalized = section('color_id,input_r,input_g,input_b',9)
    mses = section('mse_dataset,H,format,colors,mse',4)
    goals = section('quality_goal,H,max_mse,met_for_this_read',4)
    assert [r['color_id'] for r in raw] == ids == [r['color_id'] for r in normalized]
    raw_values = np.array([[r[f'raw_{c}'] for c in 'rgb'] for r in raw],np.float32)
    logged = np.array([[r[f'input_{c}'] for c in 'rgb'] for r in normalized],np.float32)
    black,white = [np.array(next(line for line in lines if line.startswith(f'# {kind}_rgb,')).split(',')[1:],np.float32) for kind in ['black','white']]
    assert all(np.all(np.isfinite(a)) for a in [raw_values,logged,black,white]) and np.all(white>black)
    assert np.all((logged>=0)&(logged<=1))
    reconstructed = np.clip((raw_values-black)/(white-black),0,1)
    assert np.allclose(reconstructed,logged,atol=1e-7,rtol=1e-6)
    timestamps = [int(r['sample_completed_ms']) for r in raw]
    assert timestamps == sorted(timestamps)
    images = re.findall(r'H=(\d+)\nMODEL_SOURCE_SHA256=([^\n]+)\n---------- PPM START ----------\n(.*?)----------- PPM END -----------',text,re.S)
    assert [int(h) for h,*_ in images] == [40,60,80,100]
    results, errors, all_pixels = [],[],[]
    for (h_text,model_hash,ppm),mse_row,goal in zip(images,mses,goals):
        h=int(h_text)
        header=HERE/f'models/h{h}/model_parameters.h'
        assert hashlib.sha256(header.read_bytes()).hexdigest()==model_hash
        tokens=ppm.split()
        assert tokens[:4]==['P3','3','3','255'] and len(tokens)==31
        pixels=np.array(tokens[4:],int).reshape(9,3)
        assert np.all((pixels>=0)&(pixels<=255))
        mse=float(np.mean((pixels-targets)**2))
        assert (int(mse_row['H']),mse_row['format'],int(mse_row['colors']))==(h,'DENSE',9)
        assert mse_row['mse_dataset']=='new_read_notebook_cmyk_reference' and abs(float(mse_row['mse'])-mse)<1e-6
        assert int(goal['H'])==h and int(goal['max_mse'])==300
        assert goal['met_for_this_read']==('YES' if mse<=300 else 'NO')
        predicted=dense_layer_references(load_model(header),reconstructed)[-1]
        expected=np.floor(np.clip(predicted.astype(float),0,1)*255+.5).astype(int)
        difference=int(np.max(np.abs(expected-pixels)))
        assert difference<=1, f'Unexpected device/host RGB8 difference H{h}: {difference}'
        name=f'h{h}_dense'
        destination=directory/f'{name}.ppm'
        destination.write_text(ppm)
        Image.fromarray(pixels.astype(np.uint8).reshape(3,3,3)).resize((360,360),Image.Resampling.NEAREST).save(directory/f'{name}.png')
        teacher=None
        if calculator:
            process=subprocess.run(['node',str(ROOT/'learning_followup/reference_audit/run_teacher_calculator.cjs'),str(calculator),str(reference),str(destination)],check=True,capture_output=True,text=True)
            teacher=json.loads(process.stdout)
            assert teacher['valid_full_image_comparison'] and abs(teacher['mse_displayed']-mse)<=.005001
        results.append({'capture_key':directory.name,'hidden_size':h,'format':'DENSE','colors':9,
                        'mse_device_reported':float(mse_row['mse']),'mse_recomputed_from_ppm':mse,
                        'teacher_site_display_mse':teacher['mse_displayed'] if teacher else None,
                        'goal_met_this_read':mse<=300,'pc_replay_max_rgb8_difference':difference,
                        'model_header_sha256':model_hash,'ppm':destination.name})
        all_pixels.append(pixels)
        for i,(pixel,target) in enumerate(zip(pixels,targets)):
            errors.append({'hidden_size':h,'color_id':ids[i],**{f'predicted_{c}8':int(pixel[j]) for j,c in enumerate('rgb')},
                           **{f'target_{c}8':int(target[j]) for j,c in enumerate('rgb')},'mse_rgb8':float(np.mean((pixel-target)**2))})
    inputs=[{**r,**{k:v for k,v in n.items() if k!='color_id'},**{f'target_{c}8':int(targets[i,j]) for j,c in enumerate('rgb')}} for i,(r,n) in enumerate(zip(raw,normalized))]
    write_csv(directory/'inputs.csv',inputs)
    write_csv(directory/'mse.csv',results)
    write_csv(directory/'patch_errors.csv',errors)
    write_csv(directory/'rgb9.csv',[{'color_id':color,**{f'h{h}_rgb':' '.join(map(str,pixels[i])) for h,pixels in zip([40,60,80,100],all_pixels)}} for i,color in enumerate(ids)])
    (directory/'normalized_log.txt').write_text(text)
    diagnostics={'white_calibration_raw':white.tolist(),'white_patch_raw':raw_values[8].tolist(),
                 'white_patch_normalized':reconstructed[8].tolist(),'black_calibration_raw':black.tolist(),
                 'black_patch_raw':raw_values[6].tolist(),'blue_patch_raw':raw_values[2].tolist(),
                 'blue_patch_normalized':reconstructed[2].tolist(),
                 'white_raw_ratio_to_calibration':(raw_values[8]/white).astype(float).tolist(),
                 'blue_normalized_distance_to_black':float(np.linalg.norm(reconstructed[2]-reconstructed[6])),
                 'interpretation':'Calibration and later same-patch values differ; cause requires a controlled check. Do not assume lighting alone explains within-capture drift.'}
    (directory/'calibration_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    summary={'received_date_japan':received_date,'execution_wall_timestamp':None,'firmware':'COLOR_RETRAIN_V7',
             'capture_counter':int(next(line for line in lines if line.startswith('CAPTURE=')).split('=')[1]),
             'physical_captures':1,'input_source':'NEW_PHYSICAL_SENSOR_READINGS',
             'role':'New read after V7 freeze; same nine-color learning chart, not unseen-chart final evaluation.',
             'training_role':'NOT_USED_FOR_V7_FITTING_OR_DEVELOPMENT','reference_ppm_sha256':reference_hash,'reference_basis':'DISTRIBUTED_NOTEBOOK_CMYK_DISPLAY_RECOVERED',
             'training_data_sha256':data_hash,'black_rgb':black.tolist(),'white_rgb':white.tolist(),
             'model_hashes_ppm_mse_and_goal_flags':'PASS','raw_to_normalized_input':'PASS',
             'normalization_max_abs_difference':float(np.max(np.abs(reconstructed-logged))),
             'original_teacher_html_mse':'PASS' if calculator else 'NOT_RUN',
             'teacher_calculator_sha256':hashlib.sha256(Path(calculator).read_bytes()).hexdigest() if calculator else None,
             'results':results,'lighting':'User requested data-only learning changes; no setup/angle changes required by this version. Actual illumination and geometry for this reading not specified in log.',
             'startup_self_check_log':'NOT_SUPPLIED','submitted_log_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('PASS: original log retained; provided-reference MSE, hashes, normalization, model replay and teacher-site values checked.')
    print(json.dumps(results))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path)
    parser.add_argument('--received-date',required=True)
    parser.add_argument('--calculator',type=Path)
    args=parser.parse_args()
    record(args.log,args.received_date,args.calculator)
