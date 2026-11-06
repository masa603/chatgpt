"""Package three frozen adapted Dense models with unchanged sensor acquisition."""

import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'experiment_today'),str(ROOT/'pruning')]
from build_fresh_scanner import replace_function
from export_csr_arduino import array_definition,dense_layer_references
from threshold_tools import load_model


def build():
    output=HERE/'color_retrain_v3';output.mkdir(exist_ok=True)
    data=HERE/'data_selected.csv';sha=hashlib.sha256(data.read_bytes()).hexdigest()
    with data.open(newline='') as f: rows=list(csv.DictReader(f))
    with (HERE/'data_all60.csv').open(newline='') as f: all_rows=list(csv.DictReader(f))
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32)
    reference=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in all_rows[:9]],np.uint8)
    indices=[0,len(rows)//4,len(rows)//2,3*len(rows)//4]
    cases=np.concatenate([np.array(list(itertools.product((0.,1.),repeat=3)),np.float32),x[indices]])
    header='#pragma once\n#include <Arduino.h>\n#include <cmath>\n'
    descriptions=[]
    for h in [40,60,100]:
        directory=HERE/f'models/h{h}';summary=json.loads((directory/'summary.json').read_text());source=directory/'model_parameters.h'
        assert summary['training_data_sha256']==sha and summary['model_header_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        text='\n'.join(line for line in source.read_text().splitlines() if not line.startswith(('#include','#pragma')))
        namespace=f'h{h}';header+=f'namespace {namespace} {{\n'+text+'\n'
        layers=load_model(source);header+=array_definition('EXPECTED',dense_layer_references(layers,cases)[-1],'float')
        header+=f'const char MODEL_SHA256[] = "{summary["model_header_sha256"]}";\n}}\n'
        descriptions.append('{'+str(h)+', '+', '.join(f'{namespace}::{n}' for n in ['weight_1','bias_1','weight_2','bias_2','weight_3','bias_3','MODEL_SHA256','EXPECTED'])+'}')
    definitions='const Model models[3] = {\n'+',\n'.join(descriptions)+'\n};\n'
    definitions+=f'const char DATA_SHA256[] = "{sha}";\nconstexpr int TRAINING_ROWS={len(rows)};\n'
    definitions+=array_definition('CHECK_INPUTS',cases,'float')+array_definition('REFERENCE_RGB',reference,'uint8_t')
    header+=(HERE/'dense_kernels.h').read_text().replace('// INSERT_DEFINITIONS',definitions)
    (output/'trained_models_v3.h').write_text(header)
    sketch=(ROOT/'hdjsk/hdjsk.ino').read_text().replace('#include "model_parameters.h"','#include "trained_models_v3.h"\nvoid outputNNInputs();')
    sketch=sketch.replace('#define HIDDEN_SIZE1 40\n#define HIDDEN_SIZE2 40\n','').replace('#define MAX_IMAGE_SIZE 8','#define MAX_IMAGE_SIZE 3')
    sketch=sketch.replace('int imageSize = 0;','int imageSize = 3;\nuint8_t trainedImages[3][3][3][3];\nfloat rawHistory[3][3][3];\nunsigned long sampleCompletedMs[3][3];\nunsigned int captureNumber=1;')
    sketch=replace_function(sketch,'predict','''void predict(float input[3],float output[3]) {
  float allOutputs[3][3];trained_models::inferAll(input,allOutputs);
  for(int model=0;model<3;model++) for(int c=0;c<3;c++) {
    float v=allOutputs[model][c];if(v<0) v=0;if(v>1) v=1;
    trainedImages[model][currentRow][currentColumn][c]=static_cast<int>(v*255.0+.5);
  }
  for(int c=0;c<3;c++) output[c]=allOutputs[1][c];
}''')
    sketch=replace_function(sketch,'outputPPM',(HERE/'capture_output.cpp').read_text())
    sketch=sketch.replace('    outputPPM();\n    outputNNInputs();','    outputPPM();')
    sketch=sketch.replace('inputHistory[currentRow][currentColumn][channel] = RGBInput[channel];','inputHistory[currentRow][currentColumn][channel] = RGBInput[channel];\n    rawHistory[currentRow][currentColumn][channel]=rawRGB[channel];')
    start=sketch.index('void handleRead()');sketch=sketch[:start]+sketch[start:].replace('  readRGB();','  readRGB();\n  sampleCompletedMs[currentRow][currentColumn]=millis();',1)
    sketch=sketch.replace("    case 'b':","    case 'n':\n    case 'N':\n      calibrationStep=0;currentRow=0;currentColumn=0;imageFinished=false;captureNumber++;\n      Serial.println(\"新しい読取：黒 b → 白 w → 各色 r\");\n      break;\n    case 'b':",1)
    sketch=sketch.replace('  delay(1000);','  delay(1000);\n  Serial.println("FIRMWARE=COLOR_RETRAIN_V3");',1)
    sketch=sketch.replace('  inputImageSize();','  if(!trained_models::selfCheck()) { Serial.println("MODEL CHECK FAILED");while(true) delay(1000); }')
    sketch=replace_function(sketch,'inputImageSize','')
    sketch=sketch.replace('ニューラルネット版準備完了','再学習Dense版準備完了：3x3、画像サイズ入力不要')
    sketch=sketch.replace('--- NN補正後 ---','--- 60次元Dense補正後（全3モデルは9色後に表示）---')
    (output/'color_retrain_v3.ino').write_text(sketch)
    print('Generated COLOR_RETRAIN_V3 with three frozen adapted Dense models.')


if __name__=='__main__':build()
