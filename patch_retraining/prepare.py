"""Validate all five labelled patch batches and prepare measured inputs for V8."""

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]; HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'retraining'))
from train_current33 import input_collision_bounds


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def main():
    folder=HERE/'physical_patch_data/session01';submitted=folder/'submitted_log.txt'
    text='\n'.join(line.strip().strip('\\').strip() for line in submitted.read_text().splitlines())+'\n'
    frames=re.findall(r'========== PATCH LEARNING START ==========\n(.*?)=========== PATCH LEARNING END ===========',text,re.S)
    assert len(frames)==5
    reference=ROOT/'distributed_training/references/reference_image1_cmyk_3x3.ppm'
    targets=np.array(reference.read_text().split()[4:],int).reshape(9,3)
    mapping={f'r{r}c{c}':targets[(r-1)*3+c-1] for r in range(1,4) for c in range(1,4)}
    expected=[('r2c3','PURPLE'),('r2c3','PURPLE'),('r3c1','BLACK'),('r3c2','GRAY'),('r3c3','WHITE')]
    patches=[];diagnostics=[];timestamps=[]
    for i,(frame,(patch,label)) in enumerate(zip(frames,expected),1):
        lines=frame.splitlines()
        assert all(value in lines for value in ['FIRMWARE=PATCH_LEARNING_CAPTURE_V1','INPUT_SOURCE=NEW_PHYSICAL_SENSOR_READINGS',
            'INTENDED_ROLE=ADDITIONAL_TRAINING_DATA; NOT_A_FULL_CHART_MSE','LABEL_SOURCE=USER_COMMAND; NOT_INFERRED_FROM_INPUT',
            f'REFERENCE_FILE_SHA256={sha(reference)}','CALIBRATION=1',f'BATCH={i}',f'PATCH_ID={patch}',f'PATCH_LABEL={label}'])
        black,white=[np.array(next(line for line in lines if line.startswith(f'# {kind}_rgb,')).split(',')[1:],np.float32) for kind in ['black','white']]
        assert np.all(white>black)
        start=next(j for j,line in enumerate(lines) if line.startswith('repeat,sample_completed_ms,'))
        rows=list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))));assert len(rows)==3
        for repeat,row in enumerate(rows,1):
            assert int(row['repeat'])==repeat
            raw=np.array([row[f'raw_{c}'] for c in 'rgb'],np.float32)
            logged=np.array([row[f'input_{c}'] for c in 'rgb'],np.float32)
            unclipped=np.array([row[f'unclipped_{c}'] for c in 'rgb'],np.float32)
            assert all(np.all(np.isfinite(a)) for a in [raw,logged,unclipped])
            actual=(raw-black)/(white-black)
            assert np.allclose(actual,unclipped,atol=1e-7,rtol=1e-6)
            assert np.allclose(np.clip(actual,0,1),logged,atol=1e-7,rtol=1e-6)
            timestamps.append(int(row['sample_completed_ms']))
            patches.append({'sample_id':f'patch_session01_b{i:02}_s{repeat}','group':'patch_session01_cal1',
                            'batch':i,'color_id':patch,'patch_label':label,**row,
                            **{f'target_{c}8':int(mapping[patch][j]) for j,c in enumerate('rgb')}})
        values=np.array([[row[f'raw_{c}'] for c in 'rgb'] for row in rows],float)
        diagnostics.append({'batch':i,'patch':patch,'label':label,'raw_min':values.min(0).tolist(),'raw_max':values.max(0).tolist()})
    assert timestamps==sorted(timestamps)
    write_csv(folder/'samples15.csv',patches)
    (folder/'normalized_log.txt').write_text(text)
    report={'source':'Actual user-supplied labelled sensor logs','received_date_japan':'2026-11-08',
            'submitted_log_sha256':sha(submitted),'reference_ppm_sha256':sha(reference),
            'batches':5,'samples':15,'distinct_colors':4,'calibration_groups':1,
            'purple_samples':6,'black_samples':3,'gray_samples':3,'white_samples':3,
            'rows_removed':0,'first_purple_batch_with_variation_retained':True,
            'normalization_and_unclipped_values':'PASS','within_batch_variation':diagnostics,
            'scope':'Additional training data. Three rapid repeats per placement are not independent chart captures. No nine-color MSE is available from these four colors.'}
    (folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    rows=read_csv(ROOT/'distributed_training/training102.csv')
    common=['sample_id','group','chart_id','color_id','input_r','input_g','input_b','target_r8','target_g8','target_b8','source_role']
    rows=[{k:r[k] for k in common} for r in rows]
    for r in read_csv(ROOT/'distributed_training/physical_readings/capture_01/inputs.csv'):
        rows.append({'sample_id':f'v7_capture01_{r["color_id"]}','group':'v7_capture01','chart_id':'train_3x3',
                     'color_id':r['color_id'],**{f'input_{c}':r[f'input_{c}'] for c in 'rgb'},
                     **{f'target_{c}8':r[f'target_{c}8'] for c in 'rgb'},
                     'source_role':'V7 new read, now V8 development, then final training; not a V8 test.'})
    for r in patches:
        rows.append({k:r[k] for k in ['sample_id','group','color_id','input_r','input_g','input_b','target_r8','target_g8','target_b8']}|
                    {'chart_id':'train_3x3','source_role':'User-labelled patch training; rapid within-placement repeats, one calibration group.'})
    assert len(rows)==126 and len({r['sample_id'] for r in rows})==126
    write_csv(HERE/'training126.csv',rows)
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32)
    y=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in rows],np.float32)
    (HERE/'input_collision_bounds.json').write_text(json.dumps(input_collision_bounds(x,y),indent=2)+'\n')
    provenance={'sources':{'v7_training102':sha(ROOT/'distributed_training/training102.csv'),
                  'v7_physical_input':sha(ROOT/'distributed_training/physical_readings/capture_01/inputs.csv'),
                  'patch_log':sha(submitted)},'training126_sha256':sha(HERE/'training126.csv'),
                'reference_ppm_sha256':sha(reference),'reference_source':'DISTRIBUTED_NOTEBOOK_CMYK_DISPLAY_RECOVERED',
                'rows':126,'distinct_reference_colors':33,'calibration_groups':len({r['group'] for r in rows}),
                'development_split':'Hold V7 nine-color read out together. Train on original102 plus all15 labelled patch samples (117).',
                'final_role_change':'After method selection, V7 nine-color read also enters all126 final training; no V8 physical read has been received.',
                'template_notebook_X_used':False,'sensor_or_inference_changes':False,'models':[40,60,80,100]}
    (HERE/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('PASS: all five batches, fifteen samples and normalization. Prepared126 measured rows; no patch sample discarded; V7 nine-color group held out for development.')


if __name__=='__main__':main()
