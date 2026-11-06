"""Verify actual trained parameters and prospective-read firmware on a PC mock."""

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'pruning'),str(ROOT/'retraining')]
from experiment_today.verify_fresh_scanner import STUB,function_body
from threshold_tools import load_model
from train_current33 import ColorNet,model_arrays,forward_fp32

RUNNER=r'''
#include "Arduino.h"
#include <cassert>
#include <iostream>
MockSerial Serial;
unsigned long clockMs=0;
int leds[3]={0},raw[3]={10,20,30},sensorReads=0,buttonReads=0;
unsigned long millis(){return clockMs;}
void delay(unsigned long ms){clockMs+=ms;}
void pinMode(uint8_t,int){}
int digitalRead(uint8_t){buttonReads++;return LOW;}
void analogWrite(uint8_t pin,int value){assert(pin==8||pin==6||pin==7);leds[pin==8?0:pin==6?1:2]=value;}
int analogRead(uint8_t pin){assert(pin==A0);int active=-1;for(int c=0;c<3;c++)if(leds[c]){assert(active==-1&&leds[c]==255);active=c;}assert(active>=0);sensorReads++;return raw[active];}
#include "color_retrain_v3.ino"
void send(const std::string&s){Serial.feed(s);while(Serial.available())loop();}
int main(int argc,char**argv){
 assert(argc==2);
 if(std::string(argv[1])=="predict"){
  float input[3],out[3][3];std::cout<<std::setprecision(9);
  while(std::cin>>input[0]>>input[1]>>input[2]){
   trained_models::inferAll(input,out);
   for(int m=0;m<3;m++)for(int c=0;c<3;c++)std::cout<<out[m][c]<<' ';
   for(int m=0;m<3;m++)for(int c=0;c<3;c++)std::cout<<outputTo255(out[m][c])<<(m==2&&c==2?'\n':' ');
  }return 0;
 }
 setup();assert(imageSize==3&&sensorReads==0&&buttonReads==0);
 assert(Serial.output.str().find("FIRMWARE=COLOR_RETRAIN_V3")!=std::string::npos);
 assert(Serial.output.str().find("ALL MODELS CHECK: PASS")!=std::string::npos);
 send("r\r\n");assert(sensorReads==0);
 int patches[9][3]={{184,55,62},{98,89,69},{27,39,74},{230,240,150},{66,138,151},{74,52,76},{0,0,0},{110,120,130},{400,400,400}};
 for(int repeat=0;repeat<2;repeat++){
  if(repeat){send("n\r\n");assert(captureNumber==2&&calibrationStep==0&&currentRow==0&&currentColumn==0&&!imageFinished);}
  raw[0]=10;raw[1]=20;raw[2]=30;send("b\r\n");assert(calibrationStep==1);
  raw[0]=210;raw[1]=220;raw[2]=230;send("w\r\n");assert(calibrationStep==2);
  for(int i=0;i<9;i++){int index=repeat?8-i:i;for(int c=0;c<3;c++)raw[c]=patches[index][c];send("r\r\n");}
  assert(imageFinished&&sensorReads==(repeat+1)*330);send("r\r\n");assert(sensorReads==(repeat+1)*330);
 }
 std::cout<<Serial.output.str();
}
'''


def main():
    with (HERE/'data_all60.csv').open(newline='') as f: allrows=list(csv.DictReader(f))
    with (HERE/'data_selected.csv').open(newline='') as f: selected=list(csv.DictReader(f))
    assert len(allrows)==60 and len(selected)==36 and len({r['group'] for r in selected})==4
    assert all(r['group']!='original_6x4' for r in selected)
    original=(ROOT/'hdjsk/hdjsk.ino').read_text();scanner=(HERE/'color_retrain_v3/color_retrain_v3.ino').read_text()
    for name in ['ledOff','readRGB','readAverage','normalizeInput','outputTo255','handleCalibration']:
        assert function_body(original,name)==function_body(scanner,name)
    models=[]
    for h in [40,60,100]:
        p=HERE/f'models/h{h}';s=json.loads((p/'summary.json').read_text())
        assert s['training_data_sha256']==hashlib.sha256((HERE/'data_selected.csv').read_bytes()).hexdigest()
        assert s['model_header_sha256']==hashlib.sha256((p/'model_parameters.h').read_bytes()).hexdigest()
        layers=load_model(p/'model_parameters.h');models.append(layers)
        archive=np.load(p/'model_arrays.npz',allow_pickle=False)
        for i,(w,b) in enumerate(layers,1):assert np.array_equal(w,archive[f'weight_{i}']) and np.array_equal(b,archive[f'bias_{i}'])
        ck=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=True);model=ColorNet(h);model.load_state_dict(ck['state_dict'])
        assert all(np.array_equal(a,b) for pair1,pair2 in zip(model_arrays(model),layers) for a,b in zip(pair1,pair2))
        x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in selected],np.float32);y=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in selected],float)
        pixels=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5)
        assert np.mean((pixels-y)**2)==s['training_mse_rgb8']
    with (ROOT/'pruning/validation.csv').open(newline='') as f: numeric_rows=allrows+list(csv.DictReader(f))
    cases=np.concatenate([np.array([[r[f'input_{c}'] for c in 'rgb'] for r in numeric_rows],np.float32),np.random.default_rng(42).random((256,3),dtype=np.float32)])
    with tempfile.TemporaryDirectory(prefix='retrain-v3-check-') as name:
        t=Path(name);(t/'Arduino.h').write_text(STUB);(t/'main.cpp').write_text(RUNNER);exe=t/'check'
        subprocess.run(['g++','-std=c++11','-O2','-ffp-contract=off','-fsanitize=address,undefined','-fno-omit-frame-pointer',f'-I{HERE/"color_retrain_v3"}',f'-I{t}',str(t/'main.cpp'),'-o',str(exe)],check=True,capture_output=True,text=True)
        result=subprocess.run([str(exe),'predict'],input='\n'.join(' '.join(format(float(v),'.9g') for v in row) for row in cases)+'\n',check=True,capture_output=True,text=True)
        predicted=np.loadtxt(io.StringIO(result.stdout));maxerror=0
        for m,layers in enumerate(models):
            expected=forward_fp32(layers,cases);maxerror=max(maxerror,float(np.max(np.abs(predicted[:,3*m:3*m+3]-expected))))
            assert np.allclose(predicted[:,3*m:3*m+3],expected,rtol=1e-6,atol=1e-6)
            assert np.array_equal(predicted[:,9+3*m:12+3*m],np.floor(np.clip(expected.astype(float),0,1)*255+.5))
        serial=subprocess.run([str(exe),'capture'],check=True,capture_output=True,text=True).stdout
    import re
    frames=re.findall(r'========== TRAINED CAPTURE START ==========\n(.*?)=========== TRAINED CAPTURE END ===========',serial,re.S);assert len(frames)==2
    patches=np.array([[184,55,62],[98,89,69],[27,39,74],[230,240,150],[66,138,151],[74,52,76],[0,0,0],[110,120,130],[400,400,400]],np.float32)
    target=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in allrows[:9]],float)
    for repeat,frame in enumerate(frames):
        assert 'FIRMWARE=COLOR_RETRAIN_V3' in frame and f'CAPTURE={repeat+1}\n' in frame
        source=patches if repeat==0 else patches[::-1];x=np.clip((source-np.array([10,20,30],np.float32))/np.float32(200),0,1)
        images=re.findall(r'---------- PPM START ----------\n(.*?)----------- PPM END -----------',frame,re.S);assert len(images)==3
        lines=frame.splitlines();start=lines.index('mse_dataset,H,format,colors,mse');mses=list(csv.DictReader(io.StringIO('\n'.join(lines[start:start+4]))))
        for m,(image,layers) in enumerate(zip(images,models)):
            tokens=image.split();assert tokens[:4]==['P3','3','3','255'];pixels=np.array(tokens[4:],int).reshape(9,3)
            expected=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5);assert np.array_equal(pixels,expected)
            assert int(mses[m]['H'])==[40,60,100][m] and mses[m]['format']=='DENSE'
            assert abs(float(mses[m]['mse'])-np.mean((pixels-target)**2))<1e-6
    (HERE/'verification.json').write_text(json.dumps({'scope':'Host verification only; prospective physical reading not received.','dataset_rows':36,'distinct_colors':9,'reading_groups':4,'checkpoint_npz_header_equality':'PASS','reported_training_mse_recomputed':'PASS','cpp11_asan_ubsan':'PASS','differential_inputs':len(cases),'max_abs_fp32_difference':maxerror,'rgb8_exact_agreement':True,'two_mock_captures_and_restart':'PASS','original_sensor_pipeline':'UNCHANGED','actual_new_physical_mse':'PENDING'},indent=2)+'\n')
    print('PASS: frozen learning parameters, metrics, 444 inputs x 3 Dense models and two complete mock captures; ASan/UBSan.')


if __name__=='__main__':main()
