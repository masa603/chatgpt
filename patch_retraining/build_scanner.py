"""Package four patch-adapted V8 models; original sensor pipeline retained."""

import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'pruning'),str(ROOT/'experiment_today')]
from threshold_tools import load_model
from export_csr_arduino import array_definition,dense_layer_references
from build_fresh_scanner import replace_function


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    output=HERE/'color_retrain_v8';output.mkdir(exist_ok=True)
    reference=ROOT/'distributed_training/references/reference_image1_cmyk_3x3.ppm'
    targets=np.array(reference.read_text().split()[4:],np.uint8).reshape(9,3)
    (output/'reference_image1_cmyk_3x3.ppm').write_bytes(reference.read_bytes())
    data=HERE/'training126.csv'
    with data.open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==126
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32)
    cases=np.concatenate([np.array(list(itertools.product((0.,1.),repeat=3)),np.float32),x[[60,66,116,125]]])
    header='#pragma once\n#include <Arduino.h>\n#include <cmath>\n'
    definitions=[]
    for h in [40,60,80,100]:
        directory=HERE/f'models/h{h}';source=directory/'model_parameters.h'
        report=json.loads((directory/'summary.json').read_text())
        assert report['model_header_sha256']==sha(source) and report['training_data_sha256']==sha(data)
        assert report['reference_ppm_sha256']==sha(reference)
        content='\n'.join(line for line in source.read_text().splitlines() if not line.startswith(('#include','#pragma')))
        namespace=f'h{h}';header+=f'namespace {namespace} {{\n'+content+'\n'
        header+=array_definition('EXPECTED',dense_layer_references(load_model(source),cases)[-1],'float')
        header+=f'const char MODEL_SHA256[] = "{sha(source)}";\n}}\n'
        definitions.append('{'+str(h)+', '+', '.join(f'{namespace}::{name}' for name in ['weight_1','bias_1','weight_2','bias_2','weight_3','bias_3','MODEL_SHA256','EXPECTED'])+'}')
    constants='constexpr int MODEL_COUNT=4;\nconst Model models[4] = {\n'+',\n'.join(definitions)+'\n};\n'
    constants+=f'const char DATA_SHA256[] = "{sha(data)}";\nconstexpr int TRAINING_ROWS=126;\n'
    constants+=f'const char REFERENCE_SHA256[] = "{sha(reference)}";\n'
    constants+=f'const char REFERENCE_NOTEBOOK_SHA256[] = "{sha(ROOT/"distributed_training/materials/train_L1_normalization.ipynb")}";\n'
    constants+=array_definition('CHECK_INPUTS',cases,'float')+array_definition('REFERENCE_RGB',targets,'uint8_t')
    kernels=(ROOT/'learning_followup/dense_kernels.h').read_text()
    kernels=kernels.replace('float output[3][3]','float output[4][3]').replace('for(int i=0;i<3;i++)','for(int i=0;i<MODEL_COUNT;i++)')
    kernels=kernels.replace('  Serial.println("FORMAT=DENSE_FP32_ONLY");','  Serial.println("FORMAT=DENSE_FP32_ONLY");\n  Serial.println("MODEL_COUNT=4; HIDDEN_SIZES=40,60,80,100");\n  Serial.println("REFERENCE_SOURCE=DISTRIBUTED_NOTEBOOK_CMYK_DISPLAY_RECOVERED");\n  Serial.print("REFERENCE_FILE_SHA256=");Serial.println(REFERENCE_SHA256);')
    header+=kernels.replace('// INSERT_DEFINITIONS',constants)
    (output/'trained_models_v8.h').write_text(header)
    sketch=(ROOT/'learning_followup/color_retrain_v3/color_retrain_v3.ino').read_text()
    sketch=sketch.replace('trained_models_v3.h','trained_models_v8.h').replace('COLOR_RETRAIN_V3','COLOR_RETRAIN_V8')
    sketch=sketch.replace('trainedImages[3][3][3][3]','trainedImages[4][3][3][3]').replace('float allOutputs[3][3]','float allOutputs[4][3]').replace('model<3','model<trained_models::MODEL_COUNT')
    sketch=replace_function(sketch,'outputPPM',(HERE/'capture_output.cpp').read_text())
    sketch=sketch.replace('全3モデル','全4モデル').replace('再学習Dense版準備完了','追加15件を反映した4モデル準備完了')
    (output/'color_retrain_v8.ino').write_text(sketch)
    single=HERE/'color_retrain_v8_single';single.mkdir(exist_ok=True)
    combined=header.replace('#pragma once\n','')+'\n'+sketch.replace('#include <Arduino.h>\n','').replace('#include \"trained_models_v8.h\"\n','')
    (single/'color_retrain_v8_single.ino').write_text(combined)
    print('Generated COLOR_RETRAIN_V8 for H40/60/80/100 with distributed CMYK reference.')


if __name__=='__main__':build()
