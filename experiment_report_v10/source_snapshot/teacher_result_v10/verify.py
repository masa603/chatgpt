"""Audit teacher targets, frozen weights and all eight t-command outputs on a host."""

import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT),str(ROOT/'pruning'),str(ROOT/'retraining')]
from prepare import read_csv,sha
from train_current33 import ColorNet,model_arrays
from experiment_today.verify_fresh_scanner import STUB
from threshold_tools import load_model,prune
from export_csr_arduino import dense_layer_references
import csv

RUNNER=r'''
#include "Arduino.h"
#include <cassert>
#include <iostream>
MockSerial Serial;
unsigned long clockMs=0,clockUs=0;int sensorReads=0;
unsigned long millis(){return clockMs;}
unsigned long micros(){clockUs+=100;return clockUs;}
void delay(unsigned long ms){clockMs+=ms;}
void pinMode(uint8_t,int){assert(false);}
int digitalRead(uint8_t){assert(false);return 0;}
int analogRead(uint8_t){sensorReads++;assert(false);return 0;}
void analogWrite(uint8_t,int){assert(false);}
#include "teacher_compare_v10.ino"
void send(const std::string &text){Serial.feed(text);while(Serial.available())loop();}
int main(int argc,char **argv){
  assert(argc==2);
  if(std::string(argv[1])=="predict"){
    float input[3],output[8][3];std::cout<<std::setprecision(9);
    while(std::cin>>input[0]>>input[1]>>input[2]){
      comparison_models::inferAll(input,output);
      for(int m=0;m<8;m++)for(int c=0;c<3;c++)std::cout<<output[m][c]<<' ';
      for(int m=0;m<8;m++)for(int c=0;c<3;c++){
        float v=output[m][c];if(v<0)v=0;if(v>1)v=1;
        std::cout<<static_cast<int>(v*255.0+.5)<<(m==7&&c==2?'\n':' ');
      }
    }return 0;
  }
  setup();assert(sensorReads==0);
  assert(Serial.output.str().find("FIRMWARE=TEACHER_RESULT_COMPARE_V10")!=std::string::npos);
  assert(Serial.output.str().find("ALL EIGHT MODELS CHECK: PASS")!=std::string::npos);
  send("\r\n \t");send("b\nw\nr\n");assert(sensorReads==0);
  send("t\r\n");send("T\n");assert(sensorReads==0);
  std::cout<<Serial.output.str();
}
'''


def rgb(layers,x):
    return np.floor(np.clip(dense_layer_references(layers,x)[-1].astype(float),0,1)*255+.5).astype(int)


def verify(calculator):
    folder=HERE/'teacher_compare_v10';reference=HERE/'reference/result.ppm'
    assert reference.read_bytes()==(ROOT/'learning_followup/reference_audit/provided/result.ppm').read_bytes()
    assert sha(reference)=='582b783b099db5bd6fae89ee0355a17f427fb4a4fc3f2750a8acb8398e5383e2'
    assert sha(calculator)=='37a6d9e935cdfa7e6ecf495bea9b46ae5ea9701aaed8caea9ba91d683fda968e'
    target=np.array(reference.read_text().split()[4:],int).reshape(9,3)
    train=read_csv(HERE/'training135.csv');old=read_csv(ROOT/'calibration_retraining/training135.csv');held=read_csv(HERE/'heldout9.csv')
    assert len(train)==len(old)==135 and len(held)==9
    assert {r['sample_id'] for r in train}.isdisjoint(r['sample_id'] for r in held)
    assert all(r['group']!='v9_capture01' for r in train)
    for row,prior in zip(train,old):
        for key in ['sample_id','group','chart_id','color_id','input_r','input_g','input_b']:assert row[key]==prior[key]
        if row['chart_id']=='train_3x3':
            r,c=map(int,re.fullmatch(r'r(\d)c(\d)',row['color_id']).groups())
            assert [int(row[f'target_{ch}8']) for ch in 'rgb']==target[(r-1)*3+c-1].tolist()
        else:
            for ch in 'rgb':assert row[f'target_{ch}8']==prior[f'target_{ch}8']
    assert sum(r['chart_id']=='train_3x3' for r in train)==111
    weights=read_csv(HERE/'sample_weights.csv');assert len(weights)==135 and all(float(r['weight'])>0 for r in weights)
    tx=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in train],np.float32)
    hx=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in held],np.float32)
    ty=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in train],int)
    summaries=[];dense=[];sparse=[];sizes=[]
    for h in [40,60,80,100]:
        path=HERE/f'models/h{h}';s=json.loads((path/'summary.json').read_text());summaries.append(s)
        assert s['training_data_sha256']==sha(HERE/'training135.csv') and s['teacher_reference_sha256']==sha(reference)
        assert s['model_header_sha256']==sha(path/'model_parameters.h')
        assert s['parent_checkpoint_sha256']==sha(ROOT/f'calibration_retraining/models/h{h}/checkpoint.pt')
        layers=load_model(path/'model_parameters.h');dense.append(layers)
        archive=np.load(path/'model_arrays.npz',allow_pickle=False)
        ck=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert ck['data_sha256']==sha(HERE/'training135.csv') and ck['reference_sha256']==sha(reference)
        model=ColorNet(h);model.load_state_dict(ck['state_dict'])
        for i,((w,b),(cw,cb)) in enumerate(zip(layers,model_arrays(model)),1):
            assert np.array_equal(w,archive[f'weight_{i}']) and np.array_equal(b,archive[f'bias_{i}'])
            assert np.array_equal(w,cw) and np.array_equal(b,cb)
        assert float(np.mean((rgb(layers,tx)-ty)**2))==s['training_mse_rgb8']
        assert float(np.mean((rgb(layers,hx)-target)**2))==s['heldout9_mse_teacher_reference']
        report=json.loads((HERE/f'selection/h{h}/summary.json').read_text());chosen=report['selected']
        assert chosen and report['model_sha256']==sha(path/'model_parameters.h')
        assert chosen['csr_bytes']<chosen['dense_bytes'] and chosen['validation_mse']<=report['dense_validation_mse']*1.05
        assert chosen['training9_mse']<=300 and 'v8_capture01' in report['training9_guard']['source']
        sparse.append(prune(layers,chosen['threshold']));sizes.append(chosen)
    models=dense+sparse
    cases=np.concatenate([tx,hx,np.random.default_rng(42).random((256,3),dtype=np.float32)])
    checks=[];max_error=0
    with tempfile.TemporaryDirectory(prefix='teacher-v10-check-') as name:
        temp=Path(name);(temp/'Arduino.h').write_text(STUB+'\nunsigned long micros();\n')
        (temp/'main.cpp').write_text(RUNNER);exe=temp/'check'
        compiled=subprocess.run(['g++','-std=c++11','-O2','-ffp-contract=off','-fsanitize=address,undefined','-fno-omit-frame-pointer',f'-I{temp}',f'-I{folder}',str(temp/'main.cpp'),'-o',str(exe)],capture_output=True,text=True)
        if compiled.returncode:raise RuntimeError(compiled.stderr)
        result=subprocess.run([str(exe),'predict'],input='\n'.join(' '.join(format(float(v),'.9g') for v in x) for x in cases)+'\n',capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
        actual=np.loadtxt(io.StringIO(result.stdout));assert actual.shape==(400,48)
        for i,layers in enumerate(models):
            expected=dense_layer_references(layers,cases)[-1]
            error=float(np.max(np.abs(actual[:,i*3:i*3+3]-expected)));max_error=max(max_error,error)
            assert np.allclose(actual[:,i*3:i*3+3],expected,atol=1e-6,rtol=1e-6)
            assert np.array_equal(actual[:,24+i*3:27+i*3],rgb(layers,cases))
        executed=subprocess.run([str(exe),'replay'],capture_output=True,text=True)
        if executed.returncode:raise RuntimeError(executed.stderr)
        serial=executed.stdout
        assert 'REPLAY REFUSED' not in serial
        frames=re.findall(r'========== TEACHER RESULT COMPARISON START ==========\n(.*?)=========== TEACHER RESULT COMPARISON END ===========',serial,re.S)
        benchmarks=re.findall(r'========== FOUR PAIR BENCHMARK START ==========\n(.*?)=========== FOUR PAIR BENCHMARK END ===========',serial,re.S)
        assert len(frames)==len(benchmarks)==2
        for frame in frames:
            assert 'ALL EIGHT RGB NUMERICAL CHECK: PASS' in frame and 'NO_NEW_SENSOR_CAPTURE' in frame
            assert 'TEACHER_REQUIRED_ORIGINAL_RESULT_PPM' in frame and sha(reference) in frame
            lines=frame.splitlines();start=lines.index('mse_dataset,H,format,colors,mse')
            metrics=list(csv.DictReader(io.StringIO('\n'.join(lines[start:start+9]))));assert len(metrics)==8
            start=lines.index('quality_goal,H,format,max_mse,met_on_saved_input')
            goals=list(csv.DictReader(io.StringIO('\n'.join(lines[start:start+9]))));assert len(goals)==8
            images=re.findall(r'H=(\d+)\nFORMAT=(DENSE|CSR)\nMODEL_SOURCE_SHA256=([^\n]+)\nTHRESHOLD=([^\n]+)\n---------- PPM START ----------\n(.*?)----------- PPM END -----------',frame,re.S)
            assert len(images)==8
            for i,(image,model,row) in enumerate(zip(images,models,metrics)):
                h,fmt,model_sha,threshold,ppm=image
                assert ppm.split()[:4]==['P3','3','3','255'] and len(ppm.split())==31
                pixels=np.array(ppm.split()[4:],int).reshape(9,3)
                assert np.array_equal(pixels,rgb(model,hx))
                assert int(h)==[40,60,80,100][i%4] and fmt==('DENSE' if i<4 else 'CSR')
                assert int(row['H'])==int(h) and row['format']==fmt and int(row['colors'])==9
                assert model_sha==summaries[i%4]['model_header_sha256']
                assert abs(float(threshold)-(0 if i<4 else sizes[i%4]['threshold']))<1e-15
                mse=float(np.mean((pixels-target)**2));assert abs(float(row['mse'])-mse)<1e-6
                assert goals[i]['met_on_saved_input']==('YES' if mse<=300 else 'NO')
                output=temp/'replay.ppm';output.write_text(ppm)
                saved=HERE/f'site_outputs/h{h}_{fmt.lower()}.ppm'
                assert saved.read_text().split()==ppm.split()
                website=subprocess.run(['node',str(ROOT/'learning_followup/reference_audit/run_teacher_calculator.cjs'),str(calculator),str(reference),str(output)],check=True,capture_output=True,text=True)
                website=json.loads(website.stdout)
                assert website['valid_full_image_comparison'] and abs(website['mse_displayed']-mse)<=.005001
                checks.append({'H':int(h),'format':fmt,'mse':mse,'teacher_html_mse_displayed':website['mse_displayed']})
        for frame in benchmarks:
            assert 'INPUT_COUNT=9;' in frame and 'HELDOUT_ARCHIVED_REAL9' in frame
            sections=re.findall(r'H=(\d+)\n(.*?)(?=\nH=|\Z)',frame,re.S);assert len(sections)==4
            for index,(h,section) in enumerate(sections):
                assert int(h)==[40,60,80,100][index] and 'TIMING VALID: YES' in section
                assert f'DENSE_MODEL_BYTES={sizes[index]["dense_bytes"]}' in section
                assert f'CSR_MODEL_BYTES={sizes[index]["csr_bytes"]}' in section
                lines=section.splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('group,first_model,'))
                groups=list(csv.DictReader(io.StringIO('\n'.join(lines[start:start+6]))));assert len(groups)==5
                for group,row in enumerate(groups):
                    assert row['first_model']==('Dense' if group%2==0 else 'CSR')
                    x=hx[np.arange(group*100,group*100+100)%9]
                    for fmt,model in [('dense',models[index]),('csr',models[index+4])]:
                        prediction=dense_layer_references(model,x)[-1]
                        expected=float((prediction[:,0]+prediction[:,1]+prediction[:,2]).astype(float).sum())
                        assert abs(float(row[f'{fmt}_checksum'])-expected)<1e-5
    report={'scope':'PC and C++ host verification on archived real inputs; no new sensor capture or Arduino timing.',
        'teacher_result_exact_original_bytes':True,'all111_3x3_targets_match_teacher':True,'all135_measured_inputs_unchanged':True,
        'original24_6x4_targets_retained':True,'all135_loss_weights_positive':True,'latest9_excluded_from_fitting_and_threshold_selection':True,
        'frozen_headers_match_npz_and_torch_checkpoints':True,'differential_inputs':len(cases),'models_per_input':8,
        'max_abs_error':max_error,'all_rgb8_values_exact':True,'cpp11_asan_ubsan':'PASS','csr_index_and_self_tests':'PASS',
        't_and_T_each_print_all8_MSE_PPM_and_all4_benchmarks':'PASS','sensor_reads':0,
        'warmup10_with_modulo9_and100_calls5_groups_alternating_order_checksums':'PASS',
        'sixteen_ppms_match_original_teacher_html':True,'clock':'FAKE_HOST_CLOCK; hardware speed not measured.',
        'heldout9_scores':checks[:8],'actual_UNO_R4_build':'NOT_PERFORMED; optional CLI download blocked by network403.',
        'new_physical_capture_count':0,'hardware_replay':'PENDING','hardware_benchmarks':'PENDING','independent_final_chart':'NOT_PERFORMED'}
    (HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: original teacher result, corrected111 labels, unchanged135 actual inputs,8 frozen models x400 inputs; t/T16PPMs match original teacher HTML, all4 benchmark controls, zero sensor reads, ASan/UBSan. Host only.')


if __name__=='__main__':verify(Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'reference/mse_calculator.html')
