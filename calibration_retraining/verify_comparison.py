"""Check all eight conditions and benchmark controls on a host; no physical results invented."""

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'pruning')]
from experiment_today.verify_fresh_scanner import STUB,function_body
from learning_followup.verify_scanner import RUNNER
from threshold_tools import load_model,prune,load_validation
from export_csr_arduino import dense_layer_references


def verify(calculator):
    folder=HERE/'compare_all_v9';sketch=(folder/'compare_all_v9.ino').read_text()
    original=(ROOT/'hdjsk/hdjsk.ino').read_text()
    for name in ['ledOff','readRGB','readAverage','normalizeInput','outputTo255','handleCalibration']:
        assert function_body(sketch,name)==function_body(original,name)
    conditions=[];dense=[];sparse=[]
    for h in [40,60,80,100]:
        layers=load_model(HERE/f'models/h{h}/model_parameters.h');dense.append(layers)
        report=json.loads((HERE/f'selection/h{h}/summary.json').read_text());chosen=report['selected'] or report['diagnostic_only']
        sparse.append(prune(layers,chosen['threshold']))
        assert chosen['csr_bytes']<chosen['dense_bytes'] and chosen['validation_mse']<=report['dense_validation_mse']*1.05
        assert chosen['training9_mse']<=300
        conditions.append({'H':h,'threshold':chosen['threshold'],'dense_bytes':chosen['dense_bytes'],'csr_bytes':chosen['csr_bytes']})
    models=dense+sparse
    with (HERE/'training135.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    validation=load_validation(ROOT/'pruning/validation.csv',expected_reads=2)
    cases=np.concatenate([np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32),validation['inputs'],np.random.default_rng(42).random((256,3),dtype=np.float32)])
    runner=RUNNER.replace('color_retrain_v3.ino','compare_all_v9.ino').replace('COLOR_RETRAIN_V3','COLOR_COMPARE_V9')
    runner=runner.replace('out[3][3]','out[8][3]').replace('trained_models::inferAll','comparison_models::inferAll').replace('m<3','m<8').replace('m==2&&c==2','m==7&&c==2')
    runner=runner.replace('unsigned long millis(){return clockMs;}','unsigned long millis(){return clockMs;}\nunsigned long mockMicros=0;unsigned long micros(){mockMicros+=100;return mockMicros;}')
    runner=runner.replace(' std::cout<<Serial.output.str();',' int before=sensorReads;send("t\\r\\n");assert(sensorReads==before && imageFinished);\n std::cout<<Serial.output.str();')
    stub=STUB+'\nunsigned long micros();\n'
    reference=ROOT/'distributed_training/references/reference_image1_cmyk_3x3.ppm'
    target=np.array(reference.read_text().split()[4:],int).reshape(9,3)
    site_checks=[]
    with tempfile.TemporaryDirectory(prefix='v9-eight-conditions-check-') as name:
        temp=Path(name);(temp/'Arduino.h').write_text(stub);(temp/'main.cpp').write_text(runner);exe=temp/'check'
        compiled=subprocess.run(['g++','-std=c++11','-O2','-ffp-contract=off','-fsanitize=address,undefined','-fno-omit-frame-pointer',f'-I{folder}',f'-I{temp}',str(temp/'main.cpp'),'-o',str(exe)],capture_output=True,text=True)
        if compiled.returncode:raise RuntimeError(compiled.stderr)
        prediction=subprocess.run([str(exe),'predict'],input='\n'.join(' '.join(format(float(v),'.9g') for v in x) for x in cases)+'\n',check=True,capture_output=True,text=True)
        actual=np.loadtxt(io.StringIO(prediction.stdout));assert actual.shape==(len(cases),48)
        max_error=0
        for index,layers in enumerate(models):
            expected=dense_layer_references(layers,cases)[-1]
            error=float(np.max(np.abs(actual[:,index*3:index*3+3]-expected)));max_error=max(max_error,error)
            assert np.allclose(actual[:,index*3:index*3+3],expected,atol=1e-6,rtol=1e-6)
            pixels=np.floor(np.clip(expected.astype(float),0,1)*255+.5)
            assert np.array_equal(actual[:,24+index*3:27+index*3],pixels)
        capture=subprocess.run([str(exe),'capture'],capture_output=True,text=True)
        if capture.returncode:raise RuntimeError(capture.stderr)
        serial=capture.stdout
        assert 'ALL EIGHT MODELS CHECK: PASS' in serial
        frames=re.findall(r'========== PAIRED CAPTURE START ==========\n(.*?)=========== PAIRED CAPTURE END ===========',serial,re.S);assert len(frames)==2
        samples=np.array([[184,55,62],[98,89,69],[27,39,74],[230,240,150],[66,138,151],[74,52,76],[0,0,0],[110,120,130],[400,400,400]],np.float32)
        for repeat,frame in enumerate(frames):
            images=re.findall(r'H=(\d+)\nFORMAT=(DENSE|CSR)\nMODEL_SOURCE_SHA256=([^\n]+)\nTHRESHOLD=([^\n]+)\n---------- PPM START ----------\n(.*?)----------- PPM END -----------',frame,re.S)
            assert len(images)==8
            lines=frame.splitlines();start=lines.index('mse_dataset,H,format,colors,mse')
            metrics=list(csv.DictReader(io.StringIO('\n'.join(lines[start:start+9]))));assert len(metrics)==8
            source=samples if repeat==0 else samples[::-1]
            x=np.clip((source-np.array([10,20,30],np.float32))/200,0,1)
            for index,(image,layers,row) in enumerate(zip(images,models,metrics)):
                h,fmt,source_hash,threshold,ppm=image
                assert int(h)==[40,60,80,100][index%4] and fmt==('DENSE' if index<4 else 'CSR')
                assert row['format']==fmt and int(row['H'])==int(h) and int(row['colors'])==9
                assert source_hash==hashlib.sha256((HERE/f'models/h{h}/model_parameters.h').read_bytes()).hexdigest()
                expected_threshold=0 if index<4 else conditions[index%4]['threshold']
                assert abs(float(threshold)-expected_threshold)<1e-15
                tokens=ppm.split();assert tokens[:4]==['P3','3','3','255'] and len(tokens)==31
                pixels=np.array(tokens[4:],int).reshape(9,3)
                expected=np.floor(np.clip(dense_layer_references(layers,x)[-1].astype(float),0,1)*255+.5)
                assert np.array_equal(pixels,expected)
                mse=float(np.mean((pixels-target)**2));assert abs(float(row['mse'])-mse)<1e-6
                output=temp/'synthetic.ppm';output.write_text(ppm)
                website=subprocess.run(['node',str(ROOT/'learning_followup/reference_audit/run_teacher_calculator.cjs'),str(calculator),str(reference),str(output)],check=True,capture_output=True,text=True)
                teacher=json.loads(website.stdout);assert teacher['valid_full_image_comparison'] and abs(teacher['mse_displayed']-mse)<=.005001
                site_checks.append({'mock_capture':repeat+1,'H':int(h),'format':fmt,'mse':mse})
        bench=re.search(r'========== FOUR PAIR BENCHMARK START ==========\n(.*?)=========== FOUR PAIR BENCHMARK END ===========',serial,re.S).group(1)
        sections=re.findall(r'H=(\d+)\n(.*?)(?=\nH=|\Z)',bench,re.S);assert len(sections)==4
        for index,(h,section) in enumerate(sections):
            assert int(h)==conditions[index]['H'] and 'TIMING VALID: YES' in section
            assert f'DENSE_MODEL_BYTES={conditions[index]["dense_bytes"]}' in section
            assert f'CSR_MODEL_BYTES={conditions[index]["csr_bytes"]}' in section
            lines=section.splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('group,first_model,'))
            groups=list(csv.DictReader(io.StringIO('\n'.join(lines[start:start+6]))));assert len(groups)==5
            for group,row in enumerate(groups):
                assert row['first_model']==('Dense' if group%2==0 else 'CSR')
                begin=group*100
                inputs=validation['inputs'][np.arange(begin,begin+100)%128]
                for fmt,model in [('dense',models[index]),('csr',models[index+4])]:
                    out=dense_layer_references(model,inputs)[-1]
                    expected=float((out[:,0]+out[:,1]+out[:,2]).astype(float).sum())
                    assert abs(float(row[f'{fmt}_checksum'])-expected)<1e-5
    report={'scope':'Host-only mock verification; no new physical readings or real board timings.',
        'conditions':8,'hidden_sizes':[40,60,80,100],'frozen_v9_retrained_after_selection':False,
        'separate_code_and_numeric_header':True,'sensor_and_normalization_functions':'UNCHANGED',
        'validation_and_training_guard':'PASS','differential_inputs':len(cases),'models_per_input':8,
        'max_abs_error':max_error,'all_rgb8_values_exact':True,'cpp11_asan_ubsan':'PASS',
        'csr_index_and_reference_checks':'PASS','two_full_paired_captures_and_n_restart':'PASS',
        'sixteen_mock_ppms_match_original_teacher_html':True,
        't_runs_all_four_pairs_without_sensor_reads':'PASS','warmup10_calls100_groups5_checksums_and_alternating_order':'PASS',
        'model_payload_sizes_match_selection':'PASS','benchmark_clock':'FAKE_HOST_CLOCK; no hardware timings recorded',
        'new_physical_results':'PENDING','independent_final_evaluation':'NOT_PERFORMED'}
    (HERE/'comparison_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: eight conditions,519 inputs, two fresh mock scans,16 website MSEs, all four timing controls and checksums; ASan/UBSan. No physical/timing result created.')


if __name__=='__main__':verify(Path(sys.argv[1]))
