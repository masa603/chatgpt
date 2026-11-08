"""Verify V9 models, complete135-row provenance and unchanged physical acquisition."""

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

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'pruning'),str(ROOT/'retraining')]
from experiment_today.verify_fresh_scanner import STUB,function_body
from learning_followup.verify_scanner import RUNNER
from threshold_tools import load_model
from train_current33 import ColorNet,model_arrays,forward_fp32

HIDDEN=[40,60,80,100]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(calculator):
    data=HERE/'training135.csv'
    with data.open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==135 and len({r['group'] for r in rows})==14
    with (ROOT/'distributed_training/references/target_rgb33.csv').open(newline='') as f:
        mapping={(r['chart_id'],r['color_id']):[int(r[f'target_{c}8']) for c in 'rgb'] for r in csv.DictReader(f)}
    assert len(mapping)==33
    for r in rows:assert [int(r[f'target_{c}8']) for c in 'rgb']==mapping[r['chart_id'],r['color_id']]
    reference=ROOT/'distributed_training/references/reference_image1_cmyk_3x3.ppm'
    tokens=reference.read_text().split();assert tokens[:4]==['P3','3','3','255'] and len(tokens)==31
    target=np.array(tokens[4:],int).reshape(9,3)
    provenance=json.loads((ROOT/'distributed_training/provenance.json').read_text())
    assert provenance['template_X_used_as_measured_input'] is False
    assert sha(reference)==provenance['reference_extraction'][0]['reference_ppm_sha256']
    assert sha(ROOT/'distributed_training/materials/train_L1_normalization.ipynb')==provenance['notebook_source_sha256']
    scanner=(HERE/'color_retrain_v9/color_retrain_v9.ino').read_text()
    original=(ROOT/'hdjsk/hdjsk.ino').read_text()
    for name in ['ledOff','readRGB','readAverage','normalizeInput','outputTo255','handleCalibration']:
        assert function_body(original,name)==function_body(scanner,name)
    header=(HERE/'color_retrain_v9/trained_models_v9.h').read_text()
    assert 'void predict(' not in header and 'void inferAll(' not in header and 'bool selfCheck(' not in header
    assert 'void predict(int index,' in scanner and 'bool selfCheck()' in scanner
    match=re.search(r'REFERENCE_RGB\[9\]\[3\]\s*=\s*\{([\s\S]*?)\};',header)
    assert match and np.array_equal(np.array(re.findall(r'\d+',match[1]),int).reshape(9,3),target)
    assert 'constexpr int MODEL_COUNT=4;' in header and 'trainedImages[4][3][3][3]' in scanner
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32)
    y=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in rows],float)
    with (ROOT/'patch_retraining/training126.csv').open(newline='') as f:original_rows=list(csv.DictReader(f))
    for original,retained in zip(original_rows,rows[:126]):
        assert all(retained[k]==original[k] for k in retained)
    with (ROOT/'patch_retraining/physical_patch_data/session01/samples15.csv').open(newline='') as f:patches=list(csv.DictReader(f))
    assert len(patches)==15 and all(rows[111+i]['sample_id']==r['sample_id'] for i,r in enumerate(patches))
    for i,patch in enumerate(patches):
        for c in 'rgb':
            assert rows[111+i][f'input_{c}']==patch[f'input_{c}']
            assert rows[111+i][f'target_{c}8']==patch[f'target_{c}8']
    with (HERE/'sample_weights_development.csv').open(newline='') as f:development_weights=list(csv.DictReader(f))
    assert len(development_weights)==126 and all(r['group']!='v8_capture01' and float(r['weight'])>0 for r in development_weights)
    with (HERE/'sample_weights_final.csv').open(newline='') as f:final_weights=list(csv.DictReader(f))
    assert len(final_weights)==135 and all(float(r['weight'])>0 for r in final_weights)
    models=[];metrics=[]
    lower=json.loads((HERE/'input_collision_bounds.json').read_text())['minimum_rgb8_mse']
    for h in HIDDEN:
        p=HERE/f'models/h{h}';s=json.loads((p/'summary.json').read_text())
        parent_stage=s.get('parent_stage',8)
        parent_folder='distributed_training' if parent_stage==7 else 'patch_retraining'
        assert s[f'parent_v{parent_stage}_checkpoint_sha256']==sha(ROOT/f'{parent_folder}/models/h{h}/checkpoint.pt')
        selected=json.loads((HERE/'selected_configs.json').read_text())[str(h)]
        assert s['selected_development_checkpoint_sha256']==sha(HERE/selected['checkpoint'])
        assert s['training_data_sha256']==sha(data) and s['reference_ppm_sha256']==sha(reference)
        assert s['model_header_sha256']==sha(p/'model_parameters.h')
        layers=load_model(p/'model_parameters.h');models.append(layers)
        assert [w.shape for w,b in layers]==[(h,3),(h,h),(3,h)]
        archive=np.load(p/'model_arrays.npz',allow_pickle=False)
        for i,(w,b) in enumerate(layers,1):assert np.array_equal(w,archive[f'weight_{i}']) and np.array_equal(b,archive[f'bias_{i}'])
        checkpoint=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert checkpoint['data_sha256']==sha(data) and checkpoint['reference_sha256']==sha(reference)
        model=ColorNet(h);model.load_state_dict(checkpoint['state_dict'])
        assert all(np.array_equal(a,b) for pair1,pair2 in zip(model_arrays(model),layers) for a,b in zip(pair1,pair2))
        pixels=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5)
        mse=float(np.mean((pixels-y)**2));assert mse==s['training_mse_rgb8'] and mse>=lower-1e-9
        metrics.append({'hidden_size':h,'unweighted_training_mse':mse,'array_payload_bytes':sum(w.nbytes+b.nbytes for w,b in layers)})
    cases=np.concatenate([x,np.array(list(itertools.product((0.,1.),repeat=3)),np.float32),np.random.default_rng(42).random((256,3),dtype=np.float32)])
    runner=RUNNER.replace('color_retrain_v3.ino','color_retrain_v9.ino').replace('COLOR_RETRAIN_V3','COLOR_RETRAIN_V9')
    runner=runner.replace('out[3][3]','out[4][3]').replace('m<3','m<4').replace('m==2&&c==2','m==3&&c==2')
    teacher_checks=[]
    with tempfile.TemporaryDirectory(prefix='four-model-v7-check-') as name:
        temp=Path(name);(temp/'Arduino.h').write_text(STUB);(temp/'main.cpp').write_text(runner);exe=temp/'check'
        subprocess.run(['g++','-std=c++11','-O2','-ffp-contract=off','-fsanitize=address,undefined','-fno-omit-frame-pointer',f'-I{HERE/"color_retrain_v9"}',f'-I{temp}',str(temp/'main.cpp'),'-o',str(exe)],check=True,capture_output=True,text=True)
        result=subprocess.run([str(exe),'predict'],input='\n'.join(' '.join(format(float(v),'.9g') for v in r) for r in cases)+'\n',check=True,capture_output=True,text=True)
        actual=np.loadtxt(io.StringIO(result.stdout));assert actual.shape==(len(cases),24)
        max_error=0
        for m,layers in enumerate(models):
            expected=forward_fp32(layers,cases)
            max_error=max(max_error,float(np.max(np.abs(actual[:,3*m:3*m+3]-expected))))
            assert np.allclose(actual[:,3*m:3*m+3],expected,rtol=1e-6,atol=1e-6)
            assert np.array_equal(actual[:,12+3*m:15+3*m],np.floor(np.clip(expected.astype(float),0,1)*255+.5))
        serial=subprocess.run([str(exe),'capture'],check=True,capture_output=True,text=True).stdout
        frames=re.findall(r'========== CALIBRATION ADAPTED CAPTURE START ==========\n(.*?)=========== CALIBRATION ADAPTED CAPTURE END ===========',serial,re.S);assert len(frames)==2
        patches=np.array([[184,55,62],[98,89,69],[27,39,74],[230,240,150],[66,138,151],[74,52,76],[0,0,0],[110,120,130],[400,400,400]],np.float32)
        for repeat,frame in enumerate(frames):
            assert 'FIRMWARE=COLOR_RETRAIN_V9' in frame and f'CAPTURE={repeat+1}\n' in frame
            assert 'MODEL_COUNT=4; HIDDEN_SIZES=40,60,80,100' in frame
            assert 'REFERENCE_SOURCE=DISTRIBUTED_NOTEBOOK_CMYK_DISPLAY_RECOVERED' in frame
            assert f'REFERENCE_FILE_SHA256={sha(reference)}\n' in frame
            images=re.findall(r'---------- PPM START ----------\n(.*?)----------- PPM END -----------',frame,re.S)
            lines=frame.splitlines();start=lines.index('mse_dataset,H,format,colors,mse')
            mses=list(csv.DictReader(io.StringIO('\n'.join(lines[start:start+5]))))
            start=lines.index('quality_goal,H,max_mse,met_for_this_read')
            goals=list(csv.DictReader(io.StringIO('\n'.join(lines[start:start+5]))))
            assert len(images)==len(mses)==len(goals)==4
            source=patches if repeat==0 else patches[::-1]
            test_x=np.clip((source-np.array([10,20,30],np.float32))/np.float32(200),0,1)
            for m,(image,layers) in enumerate(zip(images,models)):
                tokens=image.split();assert tokens[:4]==['P3','3','3','255'] and len(tokens)==31
                pixels=np.array(tokens[4:],int).reshape(9,3)
                expected=np.floor(np.clip(forward_fp32(layers,test_x).astype(float),0,1)*255+.5)
                assert np.array_equal(pixels,expected)
                mse=float(np.mean((pixels-target)**2))
                assert int(mses[m]['H'])==HIDDEN[m] and abs(float(mses[m]['mse'])-mse)<1e-6
                assert int(goals[m]['H'])==HIDDEN[m] and goals[m]['met_for_this_read']==('YES' if mse<=300 else 'NO')
                ppm=temp/'mock.ppm';ppm.write_text(image)
                result=subprocess.run(['node',str(ROOT/'learning_followup/reference_audit/run_teacher_calculator.cjs'),str(calculator),str(reference),str(ppm)],check=True,capture_output=True,text=True)
                teacher=json.loads(result.stdout)
                assert teacher['valid_full_image_comparison'] and abs(teacher['mse_displayed']-mse)<=.005001
                teacher_checks.append({'mock_capture':repeat+1,'hidden_size':HIDDEN[m],'firmware_mse':mse,'teacher_site_mse':teacher['mse_displayed']})
    report={'scope':'Host numerical and mock acquisition checks, not new physical accuracy.',
            'hidden_sizes':HIDDEN,'model_count':4,'training_rows':135,'distinct_reference_colors':33,
            'reference_source':'Distributed Notebook saved display RGB; raw original PNG bytes not locally verified.',
            'reference_ppm_sha256':sha(reference),'notebook_sha256':provenance['notebook_source_sha256'],
            'all_training_targets_and_embedded_27_components_match_reference':True,
            'template_example_X_used':False,'checkpoint_npz_header_equality':'PASS',
            'original_adc_led_calibration_normalization':'UNCHANGED','cpp11_asan_ubsan':'PASS',
            'differential_inputs':len(cases),'all_four_models_rgb8_exact':True,'max_abs_fp32_difference':max_error,
            'training_metrics':metrics,'eight_mock_image_mses_match_original_teacher_html':teacher_checks,
            'two_mock_captures_and_n_restart':'PASS',
            'all_original126_and_latest9_samples_preserved':'PASS',
            'development126_weights_exclude_v8_nine_color_group':'PASS',
            'parent_and_selected_development_checkpoint_hashes':'PASS',
            'parameter_header_separated_from_executable_sketch':'PASS',
            'new_physical_v9_mse':'NOT_YET_MEASURED'}
    physical=HERE/'physical_readings/capture_01/summary.json'
    if physical.exists():
        received=json.loads(physical.read_text())
        assert received['firmware']=='COLOR_RETRAIN_V9' and received['reference_ppm_sha256']==sha(reference)
        report['new_physical_v9_mse']={str(r['hidden_size']):r['mse_recomputed_from_ppm'] for r in received['results']}
        report['new_physical_v9_record']='physical_readings/capture_01/summary.json'
        report['new_physical_v9_independent_unseen_chart_evaluation']=False
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: four models including H80; all reference/weight checks, 399-input differential, two mock captures and teacher-site agreement; ASan/UBSan.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--calculator',type=Path,required=True);verify(p.parse_args().calculator)
