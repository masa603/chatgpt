"""Verify the paper-free t command against the actual capture and unchanged models."""

import csv
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
from experiment_today.verify_fresh_scanner import STUB
from threshold_tools import load_model,prune,load_validation
from export_csr_arduino import dense_layer_references

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
#include "compare_saved_v9.ino"
void send(const std::string &text){Serial.feed(text);while(Serial.available())loop();}
int main(){
  setup();assert(sensorReads==0);
  assert(Serial.output.str().find("ALL EIGHT MODELS CHECK: PASS")!=std::string::npos);
  send("\\r\\n");
  send("b\nw\nr\n");assert(sensorReads==0);
  send("t\r\n");send("T\n");assert(sensorReads==0);
  std::cout<<Serial.output.str();
}
'''


def verify(calculator):
    folder=HERE/'compare_saved_v9';cap=HERE/'physical_readings/capture_01'
    assert (folder/'saved_parameters_v9.h').read_text().startswith((HERE/'compare_all_v9/comparison_parameters_v9.h').read_text())
    with (cap/'inputs.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    received=json.loads((cap/'summary.json').read_text())
    raw=np.array([[r[f'raw_{c}'] for c in 'rgb'] for r in rows],np.float32)
    black=np.array(received['black_rgb'],np.float32);white=np.array(received['white_rgb'],np.float32)
    inputs=np.clip((raw-black)/(white-black),0,1)
    target=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in rows],int)
    dense=[];sparse=[]
    for h in [40,60,80,100]:
        layers=load_model(HERE/f'models/h{h}/model_parameters.h');dense.append(layers)
        selection=json.loads((HERE/f'selection/h{h}/summary.json').read_text())
        sparse.append(prune(layers,selection['selected']['threshold']))
    models=dense+sparse;validation=load_validation(ROOT/'pruning/validation.csv',expected_reads=2)
    with tempfile.TemporaryDirectory(prefix='v9-saved-comparison-check-') as name:
        temp=Path(name);(temp/'Arduino.h').write_text(STUB+'\nunsigned long micros();\n')
        (temp/'main.cpp').write_text(RUNNER);exe=temp/'check'
        compiled=subprocess.run(['g++','-std=c++11','-O2','-ffp-contract=off','-fsanitize=address,undefined','-fno-omit-frame-pointer',f'-I{temp}',f'-I{folder}',str(temp/'main.cpp'),'-o',str(exe)],capture_output=True,text=True)
        if compiled.returncode:raise RuntimeError(compiled.stderr)
        executed=subprocess.run([str(exe)],capture_output=True,text=True)
        if executed.returncode:raise RuntimeError(executed.stderr)
        text=executed.stdout
        frames=re.findall(r'========== SAVED CAPTURE COMPARISON START ==========\n(.*?)=========== SAVED CAPTURE COMPARISON END ===========',text,re.S)
        benchmarks=re.findall(r'========== FOUR PAIR BENCHMARK START ==========\n(.*?)=========== FOUR PAIR BENCHMARK END ===========',text,re.S)
        assert len(frames)==len(benchmarks)==2 and 'SAVED REPLAY REFUSED' not in text
        checks=[]
        for frame in frames:
            assert 'ORIGINAL_DENSE_RGB_CHECK: EXACT_MATCH' in frame
            assert 'NO_NEW_SENSOR_CAPTURE' in frame
            lines=frame.splitlines();start=lines.index('mse_dataset,H,format,colors,mse')
            metrics=list(csv.DictReader(io.StringIO('\n'.join(lines[start:start+9]))));assert len(metrics)==8
            start=lines.index('H,format,notebook_cmyk_mse,site_result_pure_rgb_mse')
            dual=list(csv.DictReader(io.StringIO('\n'.join(lines[start:start+9]))));assert len(dual)==8
            pure=np.array((ROOT/'learning_followup/reference_audit/provided/result.ppm').read_text().split()[4:],int).reshape(9,3)
            images=re.findall(r'H=(\d+)\nFORMAT=(DENSE|CSR)\nMODEL_SOURCE_SHA256=([^\n]+)\nTHRESHOLD=([^\n]+)\n---------- PPM START ----------\n(.*?)----------- PPM END -----------',frame,re.S)
            assert len(images)==8
            for i,(image,model,row) in enumerate(zip(images,models,metrics)):
                h,fmt,model_sha,threshold,ppm=image
                expected=np.floor(np.clip(dense_layer_references(model,inputs)[-1].astype(float),0,1)*255+.5).astype(int)
                pixels=np.array(ppm.split()[4:],int).reshape(9,3)
                assert np.array_equal(pixels,expected)
                assert int(h)==[40,60,80,100][i%4] and fmt==('DENSE' if i<4 else 'CSR')
                assert int(row['H'])==int(h) and row['format']==fmt and int(row['colors'])==9
                mse=float(np.mean((pixels-target)**2));assert abs(float(row['mse'])-mse)<1e-6
                assert abs(float(dual[i]['notebook_cmyk_mse'])-mse)<1e-6
                assert abs(float(dual[i]['site_result_pure_rgb_mse'])-float(np.mean((pixels-pure)**2)))<1e-6
                if i==0:assert abs(float(dual[i]['site_result_pure_rgb_mse'])-3471.0)<1e-6
                if i<4:
                    original=np.array((cap/f'h{h}_dense.ppm').read_text().split()[4:],int).reshape(9,3)
                    assert np.array_equal(original,pixels)
                output=temp/'replay.ppm';output.write_text(ppm)
                reference=ROOT/'distributed_training/references/reference_image1_cmyk_3x3.ppm'
                result=subprocess.run(['node',str(ROOT/'learning_followup/reference_audit/run_teacher_calculator.cjs'),str(calculator),str(reference),str(output)],check=True,capture_output=True,text=True)
                website=json.loads(result.stdout);assert website['valid_full_image_comparison'] and abs(website['mse_displayed']-mse)<=.005001
                checks.append({'H':int(h),'format':fmt,'mse':mse})
        for frame in benchmarks:
            sections=re.findall(r'H=(\d+)\n(.*?)(?=\nH=|\Z)',frame,re.S);assert len(sections)==4
            for index,(h,section) in enumerate(sections):
                assert int(h)==[40,60,80,100][index] and 'TIMING VALID: YES' in section
                lines=section.splitlines();start=next(i for i,line in enumerate(lines) if line.startswith('group,first_model,'))
                groups=list(csv.DictReader(io.StringIO('\n'.join(lines[start:start+6]))));assert len(groups)==5
                for group,row in enumerate(groups):
                    assert row['first_model']==('Dense' if group%2==0 else 'CSR')
                    x=validation['inputs'][np.arange(group*100,group*100+100)%128]
                    for fmt,model in [('dense',models[index]),('csr',models[index+4])]:
                        prediction=dense_layer_references(model,x)[-1]
                        expected=float((prediction[:,0]+prediction[:,1]+prediction[:,2]).astype(float).sum())
                        assert abs(float(row[f'{fmt}_checksum'])-expected)<1e-5
    report={'scope':'Host mock t-command controls and actual saved-input numerical checks; not hardware speed or a new sensor capture.',
        'cpp11_asan_ubsan':'PASS','model_header_prefix_unchanged':True,'weights_and_thresholds_changed':False,
        't_and_uppercase_T_each_emit_replay_and_all4_benchmarks':'PASS','sensor_reads':0,
        'original4_dense_rgb_exact_match':True,'all8_replay_rgb_match_pc':True,
        'sixteen_ppms_match_original_teacher_html':True,'five_groups_each_and_alternating_order_checksums':'PASS',
        'both_reference_mse_columns_match_same_pixels':True,'screenshot_H40_pure_rgb_3471_reproduced':True,
        'replay_mses':checks[:8],'clock':'FAKE_HOST_CLOCK; not measured board times.',
        'hardware_saved_replay':'PENDING','hardware_benchmark':'PENDING'}
    (HERE/'saved_comparison_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: t/T each replay all8 with original Dense pixels exact, then all4 timed groups;16 website MSEs, zero sensor reads; ASan/UBSan. No hardware timing claimed.')


if __name__=='__main__':verify(Path(sys.argv[1]))
