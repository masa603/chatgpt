"""Build a separate-code/parameter t program using only the teacher-required reference."""

import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT/'pruning')]
from prepare import read_csv,write_csv,sha
from threshold_tools import load_model,prune,storage
from export_csr_arduino import array_definition,dense_layer_references


def build():
    folder=HERE/'teacher_compare_v10';folder.mkdir(exist_ok=True)
    reference=HERE/'reference/result.ppm'
    assert sha(reference)=='582b783b099db5bd6fae89ee0355a17f427fb4a4fc3f2750a8acb8398e5383e2'
    target=np.array(reference.read_text().split()[4:],np.uint8).reshape(9,3)
    training=read_csv(HERE/'training135.csv');heldout=read_csv(HERE/'heldout9.csv')
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in training],np.float32)
    inputs=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in heldout],np.float32)
    cases=np.concatenate([np.array(list(itertools.product((0.,1.),repeat=3)),np.float32),x[[60,66,116,134]]])
    header='#pragma once\n#include <Arduino.h>\n'
    dense_definitions=[];sparse_definitions=[];dense_sizes=[];models=[];metrics=[]
    for h in [40,60,80,100]:
        source=HERE/f'models/h{h}/model_parameters.h';summary=json.loads((source.parent/'summary.json').read_text())
        assert summary['teacher_reference_sha256']==sha(reference) and summary['model_header_sha256']==sha(source)
        dense=load_model(source);namespace=f'h{h}'
        content='\n'.join(line for line in source.read_text().splitlines() if not line.startswith(('#include','#pragma')))
        header+=f'namespace {namespace} {{\n'+content+'\n'
        header+=array_definition('EXPECTED',dense_layer_references(dense,cases)[-1],'float')
        header+=f'const char MODEL_SHA256[]="{sha(source)}";\n}}\n'
        dense_definitions.append('{'+str(h)+', '+', '.join(f'{namespace}::{n}' for n in ['weight_1','bias_1','weight_2','bias_2','weight_3','bias_3','MODEL_SHA256','EXPECTED'])+'}')
        report=json.loads((HERE/f'selection/h{h}/summary.json').read_text())
        assert report['model_sha256']==sha(source)
        chosen=report['selected'] or report['diagnostic_only'];assert chosen is not None
        sparse=prune(dense,chosen['threshold']);namespace=f'csr_h{h}';header+=f'namespace {namespace} {{\n'
        layers=[];count=0
        for i,(w,b) in enumerate(sparse,1):
            rr,cc=np.nonzero(w);nnz=len(rr);stored=max(nnz,1)
            values=w[rr,cc].astype(np.float32) if nnz else np.zeros(1,np.float32)
            columns=cc.astype(np.uint8) if nnz else np.zeros(1,np.uint8)
            pointers=np.concatenate([[0],np.cumsum(np.count_nonzero(w,axis=1))]).astype(np.uint16)
            header+=array_definition(f'values_{i}',values,'float')+array_definition(f'columns_{i}',columns,'uint8_t')+array_definition(f'rows_{i}',pointers,'uint16_t')
            layers.append('{'+f'{namespace}::values_{i}, h{h}::bias_{i}, {namespace}::columns_{i}, {namespace}::rows_{i}, {w.shape[0]}, {w.shape[1]}, {nnz}, {stored}'+'}')
            count+=values.nbytes+columns.nbytes+pointers.nbytes+b.nbytes
        assert count==chosen['csr_bytes']
        header+=array_definition('EXPECTED',dense_layer_references(sparse,cases)[-1],'float')+'}\n'
        sparse_definitions.append('{{'+', '.join(layers)+'}, '+format(chosen['threshold'],'.17g')+f', {namespace}::EXPECTED, {count}, '+('true' if report['selected'] else 'false')+'}')
        dense_sizes.append(storage(dense)['dense_bytes']);models.append((dense,sparse))
        metrics.append({'H':h,'threshold':chosen['threshold'],'dense_bytes':dense_sizes[-1],'csr_bytes':count,
            'model_source_sha256':sha(source),'selection_status':report['selection_status'],
            'adopted':report['selected'] is not None,'threshold_guard_source':'Training v8_capture01 only; latest9 remain excluded.'})
    header+='namespace trained_models {\nstruct Model {int hidden;const float *w1,*b1,*w2,*b2,*w3,*b3;const char *sha256;const float (*expected)[3];};\n'
    header+='constexpr int MODEL_COUNT=4;\nconst Model models[4]={\n'+',\n'.join(dense_definitions)+'\n};\n'
    header+=f'const char DATA_SHA256[]="{sha(HERE/"training135.csv")}";\nconstexpr int TRAINING_ROWS=135;\n'
    header+=f'const char REFERENCE_SHA256[]="{sha(reference)}";\n'
    header+=array_definition('CHECK_INPUTS',cases,'float')+array_definition('REFERENCE_RGB',target,'uint8_t')+'}\n'
    header+='namespace comparison_models {\nstruct SparseLayer {const float *values,*bias;const uint8_t *col;const uint16_t *row;uint16_t rows,cols,nnz,stored;};\n'
    header+='struct SparseModel {SparseLayer layers[3];double threshold;const float (*expected)[3];uint32_t bytes;bool eligible;};\n'
    header+='constexpr int MODEL_COUNT=8;\nconst SparseModel sparse[4]={\n'+',\n'.join(sparse_definitions)+'\n};\n'
    header+=array_definition('DENSE_BYTES',np.array(dense_sizes,np.uint32),'uint32_t')
    header+='constexpr int BENCHMARK_INPUT_COUNT=9;\n'
    header+=f'const char BENCHMARK_INPUT_SHA256[]="{sha(HERE/"heldout9.csv")}";\n'
    header+=array_definition('BENCHMARK_INPUTS',inputs,'float')+'}\n'
    all_models=[pair[0] for pair in models]+[pair[1] for pair in models]
    expected=np.array([np.floor(np.clip(dense_layer_references(layers,inputs)[-1].astype(float),0,1)*255+.5).astype(np.uint8) for layers in all_models])
    header+='namespace evaluation_data {\n'+array_definition('EXPECTED_RGB',expected.reshape(72,3),'uint8_t')
    header+=f'const char SOURCE_PHYSICAL_LOG_SHA256[]="{sha(ROOT/"calibration_retraining/physical_readings/capture_01/submitted_log.txt")}";\n}}\n'
    (folder/'teacher_parameters_v10.h').write_text(header)
    old=(ROOT/'calibration_retraining/color_retrain_v9/color_retrain_v9.ino').read_text()
    start=old.index('namespace trained_models {');end=old.index('\nvoid outputNNInputs();',start)
    dense_code=old[start:end].replace('DISTRIBUTED_NOTEBOOK_CMYK_DISPLAY_RECOVERED','TEACHER_REQUIRED_ORIGINAL_RESULT_PPM')
    code='#include <Arduino.h>\n#include <cmath>\n#include "teacher_parameters_v10.h"\n\n'+dense_code+'\n'
    code+=(ROOT/'calibration_retraining/comparison_kernels.cpp').read_text()+'\n'
    code+=(HERE/'teacher_replay.cpp').read_text()+'\n'
    bench=(ROOT/'calibration_retraining/comparison_benchmark.cpp').read_text()
    bench=bench.replace('COLOR_COMPARE_V9','TEACHER_RESULT_COMPARE_V10').replace('MODEL_GENERATION=FROZEN_V9','MODEL_GENERATION=FROZEN_TEACHER_RESULT_V10')
    bench=bench.replace('INPUT_SOURCE=ARCHIVED_VALIDATION; NO_NEW_SENSOR_CAPTURE','INPUT_SOURCE=HELDOUT_ARCHIVED_REAL9; NO_NEW_SENSOR_CAPTURE').replace('INPUT_COUNT=128;','INPUT_COUNT=9;')
    bench=bench.replace('BENCHMARK_INPUTS[call]','BENCHMARK_INPUTS[call%comparison_models::BENCHMARK_INPUT_COUNT]')
    marker='  Serial.println("========== FOUR PAIR BENCHMARK START ==========");'
    bench=bench.replace(marker,'  if(!printTeacherReferenceResults())return;\n'+marker)
    code+=bench+'''
void setup() {
  Serial.begin(9600);delay(1000);
  Serial.println("FIRMWARE=TEACHER_RESULT_COMPARE_V10");
  Serial.println("REFERENCE=result.ppm; TEACHER_REQUIRED_ORIGINAL_FILE");
  if(!comparison_models::selfCheckAll()){Serial.println("MODEL CHECK FAILED");while(true)delay(1000);}
  Serial.println("tを1回送信すると、学習に使っていない保存済み9色のMSEと4組の速度を出します。");
}

void loop() {
  if(Serial.available()) {
    char c=static_cast<char>(Serial.read());
    if(c=='t'||c=='T')runBenchmarks();
    else if(c!='\\r'&&c!='\\n'&&c!=' '&&c!='\\t')Serial.println("半角のtを送信してください。");
  }
  delay(1);
}
'''
    assert 'CMYK' not in code and 'COLOR_COMPARE_V9' not in code
    (folder/'teacher_compare_v10.ino').write_text(code)
    outputs=[]
    for i,pixels in enumerate(expected):
        h=[40,60,80,100][i%4];fmt='DENSE' if i<4 else 'CSR'
        path=HERE/'site_outputs'/f'h{h}_{fmt.lower()}.ppm';path.parent.mkdir(exist_ok=True)
        path.write_text('P3\n3 3\n255\n'+'\n'.join(' '.join(map(str,row.flatten())) for row in pixels.reshape(3,3,3))+'\n')
        outputs.append({'H':h,'format':fmt,'mse':float(np.mean((pixels.astype(float)-target)**2)),
                        'scope':'PC inference on held-out actual inputs; not a new physical or device result.'})
    (HERE/'comparison_pc_summary.json').write_text(json.dumps({'teacher_reference_sha256':sha(reference),
        'parameters_sha256':sha(folder/'teacher_parameters_v10.h'),'conditions':metrics,'heldout_outputs':outputs,
        'heldout9_used_to_select_threshold':False,'new_physical_capture_count':0,'device_replay':'PENDING','device_time':'PENDING'},indent=2)+'\n')
    print('Built teacher-only V10 two-file program: t evaluates all8 on held-out9 then times all4 pairs. Original result.ppm unchanged.')


if __name__=='__main__':build()
