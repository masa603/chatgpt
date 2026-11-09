"""Exercise ten real-reading scanner cycles using synthetic host inputs, not fake device data."""

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
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'pruning')]
from prepare import sha
from experiment_today.verify_fresh_scanner import STUB,function_body
from threshold_tools import load_model,prune
from export_csr_arduino import dense_layer_references

RUNNER=r'''
#include "Arduino.h"
#include <cassert>
#include <iostream>
MockSerial Serial;
unsigned long clockMs=0;int leds[3]={0},raw[3]={10,20,30},sensorReads=0,buttonReads=0;
unsigned long millis(){return clockMs;}
void delay(unsigned long ms){clockMs+=ms;}
void pinMode(uint8_t,int){}
int digitalRead(uint8_t){buttonReads++;return LOW;}
void analogWrite(uint8_t pin,int value){assert(pin==8||pin==6||pin==7);leds[pin==8?0:pin==6?1:2]=value;}
int analogRead(uint8_t pin){assert(pin==A0);int channel=-1;for(int c=0;c<3;c++)if(leds[c]){assert(channel==-1&&leds[c]==255);channel=c;}assert(channel>=0);sensorReads++;return raw[channel];}
#include "teacher_fresh_v10.ino"
void send(const std::string &text){Serial.feed(text);while(Serial.available())loop();}
void patch(int r,int g,int b){raw[0]=r;raw[1]=g;raw[2]=b;}
int main(){
  setup();assert(imageSize==3&&sensorReads==0&&buttonReads==0&&completedCaptures==0);
  assert(Serial.output.str().find("FIRMWARE=TEACHER_FRESH_V10")!=std::string::npos);
  assert(Serial.output.str().find("ALL EIGHT MODELS CHECK: PASS")!=std::string::npos);
  send("r\r\nw\r\nt\r\n");assert(sensorReads==0&&completedCaptures==0);
  patch(10,20,30);send("b\r\n");send("w\r\n");
  assert(calibrationStep==0&&sensorReads==60);send("r\r\n");assert(sensorReads==60);
  patch(10,20,30);send("b\r\n");patch(210,220,230);send("w\r\n");
  patch(184,55,62);send("r\r\n");assert(sensorReads==150&&currentColumn==1&&completedCaptures==0);
  send("n\r\n");assert(captureNumber==2&&calibrationStep==0&&currentColumn==0&&!imageFinished&&completedCaptures==0);
  int patches[9][3]={{184,55,62},{98,89,69},{27,39,74},{230,240,150},{66,138,151},{74,52,76},{0,0,0},{110,120,130},{400,400,400}};
  for(int repeat=0;repeat<10;repeat++){
    if(repeat)send("n\r\n");
    assert(captureNumber==repeat+2&&calibrationStep==0&&currentRow==0&&currentColumn==0&&!imageFinished&&completedCaptures==repeat);
    patch(10,20,30);send("b\r\n");patch(210,220,230);send("w\r\n");assert(calibrationStep==2);
    for(int point=0;point<9;point++){
      patch(patches[point][0]+repeat,patches[point][1]+repeat,patches[point][2]+repeat);send("r\r\n");
      if(point<8)assert(completedCaptures==repeat);
    }
    assert(imageFinished&&completedCaptures==repeat+1&&sensorReads==150+(repeat+1)*330);
    send("r\r\nt\r\n");assert(sensorReads==150+(repeat+1)*330&&completedCaptures==repeat+1);
  }
  assert(completedCaptures==10&&buttonReads==0&&leds[0]==0&&leds[1]==0&&leds[2]==0);
  std::cout<<Serial.output.str();
}
'''


def verify():
    folder=HERE/'teacher_fresh_v10';code=(folder/'teacher_fresh_v10.ino').read_text()
    frozen=HERE/'teacher_compare_v10'
    assert (folder/'teacher_parameters_v10.h').read_bytes()==(frozen/'teacher_parameters_v10.h').read_bytes()
    assert code[:code.index('void outputNNInputs();')]==(frozen/'teacher_compare_v10.ino').read_text().split('bool printTeacherReferenceResults()')[0]
    original=(ROOT/'hdjsk/hdjsk.ino').read_text()
    for name in ['ledOff','readAverage','readRGB','normalizeInput','outputTo255','handleCalibration']:
        assert function_body(code,name)==function_body(original,name),name
    assert 'BENCHMARK_INPUTS[' not in code and 'runBenchmarks' not in code
    target=np.array((HERE/'reference/result.ppm').read_text().split()[4:],int).reshape(9,3)
    dense=[];sparse=[];thresholds=[]
    for h in [40,60,80,100]:
        layers=load_model(HERE/f'models/h{h}/model_parameters.h');dense.append(layers)
        chosen=json.loads((HERE/f'selection/h{h}/summary.json').read_text())['selected'];thresholds.append(chosen['threshold'])
        sparse.append(prune(layers,chosen['threshold']))
    models=dense+sparse
    patches=np.array([[184,55,62],[98,89,69],[27,39,74],[230,240,150],[66,138,151],[74,52,76],[0,0,0],[110,120,130],[400,400,400]],np.float32)
    max_error=0
    with tempfile.TemporaryDirectory(prefix='v10-fresh-controls-') as name:
        temp=Path(name);(temp/'Arduino.h').write_text(STUB);(temp/'main.cpp').write_text(RUNNER);exe=temp/'check'
        compiled=subprocess.run(['g++','-std=c++11','-O2','-ffp-contract=off','-fsanitize=address,undefined','-fno-omit-frame-pointer',f'-I{temp}',f'-I{folder}',str(temp/'main.cpp'),'-o',str(exe)],capture_output=True,text=True)
        if compiled.returncode:raise RuntimeError(compiled.stderr)
        result=subprocess.run([str(exe)],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
        serial=result.stdout
        assert serial.count('INCOMPLETE CAPTURE ABORTED')==1
        frames=re.findall(r'========== TEACHER FRESH CAPTURE START ==========\n(.*?)=========== TEACHER FRESH CAPTURE END ===========',serial,re.S)
        assert len(frames)==10
        for repeat,frame in enumerate(frames):
            assert f'CAPTURE={repeat+2}\n' in frame and f'COMPLETED_CAPTURES_THIS_BOOT={repeat+1}\n' in frame
            assert 'FIRMWARE=TEACHER_FRESH_V10' in frame and 'INPUT_SOURCE=NEW_PHYSICAL_SENSOR_READINGS' in frame
            assert 'MODEL_GENERATION=FROZEN_V10' in frame and 'TEACHER_REQUIRED_ORIGINAL_RESULT_PPM' in frame
            assert sha(HERE/'reference/result.ppm') in frame and sha(HERE/'training135.csv') in frame
            lines=frame.splitlines();i=lines.index('mse_dataset,H,format,colors,mse')
            metrics=list(csv.DictReader(io.StringIO('\n'.join(lines[i:i+9]))));assert len(metrics)==8
            images=re.findall(r'H=(\d+)\nFORMAT=(DENSE|CSR)\nMODEL_SOURCE_SHA256=([^\n]+)\nTHRESHOLD=([^\n]+)\n---------- PPM START ----------\n(.*?)----------- PPM END -----------',frame,re.S)
            assert len(images)==8
            raw=patches+repeat;x=np.clip((raw-np.array([10,20,30],np.float32))/np.float32(200),0,1)
            start=lines.index('color_id,sample_completed_ms,raw_r,raw_g,raw_b')
            raw_rows=list(csv.DictReader(io.StringIO('\n'.join(lines[start:start+10]))));assert len(raw_rows)==9
            assert np.array_equal(np.array([[r[f'raw_{c}'] for c in 'rgb'] for r in raw_rows],np.float32),raw)
            start=lines.index('color_id,input_r,input_g,input_b')
            input_rows=list(csv.DictReader(io.StringIO('\n'.join(lines[start:start+10]))));assert len(input_rows)==9
            observed=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in input_rows],np.float32)
            assert np.allclose(observed,x,atol=1e-8,rtol=1e-7)
            for index,(image,model,row) in enumerate(zip(images,models,metrics)):
                h,fmt,model_sha,threshold,ppm=image;assert int(h)==[40,60,80,100][index%4] and fmt==('DENSE' if index<4 else 'CSR')
                assert model_sha==sha(HERE/f'models/h{h}/model_parameters.h')
                assert abs(float(threshold)-(0 if index<4 else thresholds[index%4]))<1e-15
                tokens=ppm.split();assert tokens[:4]==['P3','3','3','255'] and len(tokens)==31
                pixels=np.array(tokens[4:],int).reshape(9,3)
                expected=np.floor(np.clip(dense_layer_references(model,x)[-1].astype(float),0,1)*255+.5).astype(int)
                assert np.array_equal(pixels,expected)
                mse=float(np.mean((pixels-target)**2));error=abs(float(row['mse'])-mse);max_error=max(max_error,error)
                assert int(row['H'])==int(h) and row['format']==fmt and int(row['colors'])==9 and error<5.1e-7
                output=temp/'synthetic_test_only.ppm';output.write_text(ppm)
                site=subprocess.run(['node',str(ROOT/'learning_followup/reference_audit/run_teacher_calculator.cjs'),str(HERE/'reference/mse_calculator.html'),str(HERE/'reference/result.ppm'),str(output)],check=True,capture_output=True,text=True)
                site=json.loads(site.stdout);assert site['valid_full_image_comparison'] and abs(site['mse_displayed']-mse)<=.005001
    report={'scope':'Synthetic host functional tests only. These are not ten real paper readings and no synthetic captures are exported as measurement data.',
        'frozen_parameter_header_sha256':sha(folder/'teacher_parameters_v10.h'),
        'parameter_header_byte_identical_to_verified_V10':True,'inference_kernels_unchanged':True,'six_sensor_functions_unchanged':True,
        'cpp11_asan_ubsan':'PASS','calibration_failure_then_retry':'PASS','read_before_calibration_does_not_sample':'PASS',
        'incomplete_capture_abort_logged_not_counted':'PASS','ten_complete_host_cycles_each_read9_fresh_inputs':'PASS',
        'extra_r_or_t_does_not_repeat_completed_capture':'PASS','all80_PPMs_match_frozen_PC_and_original_teacher_HTML':'PASS',
        'no_physical_buttons_read':True,'no_archived_inputs_used_for_outputs':True,'synthetic_raw_sensor_reads':3450,
        'max_printed_mse_rounding_error':max_error,'physical_V10_captures_received':0,'actual_fresh_firmware_build':'PENDING_USER_ARDUINO_IDE'}
    (HERE/'fresh_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: frozenV10 header/kernels; original sensor functions;10 synthetic host cycles,80PPMs and original teacher HTML, calibration failure/abort/restart controls, ASan/UBSan. No real paper readings claimed.')


if __name__=='__main__':verify()
