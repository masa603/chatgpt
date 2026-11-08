"""One upload: all four frozen V9 Dense/CSR pairs, fresh colors and saved-input timing."""

import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'pruning'),str(ROOT/'experiment_today')]
from threshold_tools import load_model,prune,load_validation,storage
from export_csr_arduino import array_definition,dense_layer_references
from build_fresh_scanner import replace_function


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    output=HERE/'compare_all_v9';output.mkdir(exist_ok=True)
    header=(HERE/'color_retrain_v9/trained_models_v9.h').read_text()
    with (HERE/'training135.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32)
    cases=np.concatenate([np.array(list(itertools.product((0.,1.),repeat=3)),np.float32),x[[60,66,116,134]]])
    definitions=[];metrics=[];dense_bytes=[]
    for h in [40,60,80,100]:
        source=HERE/f'models/h{h}/model_parameters.h'
        assert sha(source) in header
        report=json.loads((HERE/f'selection/h{h}/summary.json').read_text())
        assert report['model_sha256']==sha(source) and report['validation_csv_sha256']==sha(ROOT/'pruning/validation.csv')
        chosen=report['selected'] or report['diagnostic_only'];assert chosen is not None
        dense=load_model(source);sparse=prune(dense,chosen['threshold'])
        dense_bytes.append(storage(dense)['dense_bytes'])
        namespace=f'csr_h{h}';header+=f'\nnamespace {namespace} {{\n'
        layer_defs=[];byte_count=0
        for i,(w,b) in enumerate(sparse,1):
            rr,cc=np.nonzero(w);nnz=len(rr);stored=max(1,nnz)
            values=w[rr,cc].astype(np.float32) if nnz else np.zeros(1,np.float32)
            columns=cc.astype(np.uint8) if nnz else np.zeros(1,np.uint8)
            pointers=np.concatenate([[0],np.cumsum(np.count_nonzero(w,axis=1))]).astype(np.uint16)
            header+=array_definition(f'values_{i}',values,'float')
            header+=array_definition(f'columns_{i}',columns,'uint8_t')
            header+=array_definition(f'rows_{i}',pointers,'uint16_t')
            layer_defs.append('{'+f'{namespace}::values_{i}, h{h}::bias_{i}, {namespace}::columns_{i}, {namespace}::rows_{i}, {w.shape[0]}, {w.shape[1]}, {nnz}, {stored}'+'}')
            byte_count+=values.nbytes+columns.nbytes+pointers.nbytes+b.nbytes
        assert byte_count==chosen['csr_bytes']
        header+=array_definition('EXPECTED',dense_layer_references(sparse,cases)[-1],'float')+'}\n'
        definitions.append('{{'+', '.join(layer_defs)+'}, '+format(chosen['threshold'],'.17g')+f', {namespace}::EXPECTED, {byte_count}, '+('true' if report['selected'] is not None else 'false')+'}')
        metrics.append({'H':h,'dense_bytes':dense_bytes[-1],'csr_bytes':byte_count,'threshold':chosen['threshold'],
                        'model_sha256':sha(source),'selection_status':report['selection_status'],
                        'validation_dense_mse':report['dense_validation_mse'],'validation_csr_mse':chosen['validation_mse'],
                        'training9_guard_mse':chosen.get('training9_mse'),'adopted':report['selected'] is not None})
    header+='\nnamespace comparison_models {\n'
    header+='struct SparseLayer {const float *values,*bias;const uint8_t *col;const uint16_t *row;uint16_t rows,cols,nnz,stored;};\n'
    header+='struct SparseModel {SparseLayer layers[3];double threshold;const float (*expected)[3];uint32_t bytes;bool eligible;};\n'
    header+='constexpr int MODEL_COUNT=8;\nconst SparseModel sparse[4] = {\n'+',\n'.join(definitions)+'\n};\n'
    header+=array_definition('DENSE_BYTES',np.array(dense_bytes,np.uint32),'uint32_t')
    validation=load_validation(ROOT/'pruning/validation.csv',expected_reads=2)
    assert len(validation['inputs'])==128
    header+='constexpr int BENCHMARK_INPUT_COUNT=128;\n'
    header+=f'const char BENCHMARK_INPUT_SHA256[]="{sha(ROOT/"pruning/validation.csv")}";\n'
    header+=array_definition('BENCHMARK_INPUTS',validation['inputs'],'float')+'}\n'
    (output/'comparison_parameters_v9.h').write_text(header)
    sketch=(HERE/'color_retrain_v9/color_retrain_v9.ino').read_text()
    sketch=sketch.replace('trained_models_v9.h','comparison_parameters_v9.h').replace('COLOR_RETRAIN_V9','COLOR_COMPARE_V9')
    sketch=sketch.replace('trainedImages[4][3][3][3]','trainedImages[8][3][3][3]').replace('float allOutputs[4][3]','float allOutputs[8][3]')
    sketch=sketch.replace('trained_models::inferAll(input,allOutputs)','comparison_models::inferAll(input,allOutputs)')
    sketch=sketch.replace('model<trained_models::MODEL_COUNT','model<comparison_models::MODEL_COUNT')
    sketch=sketch.replace('void outputNNInputs();',(HERE/'comparison_kernels.cpp').read_text()+'\nvoid outputNNInputs();')
    sketch=replace_function(sketch,'outputPPM',(HERE/'comparison_capture.cpp').read_text())
    sketch=sketch.replace('void handleSerialCommand() {',(HERE/'comparison_benchmark.cpp').read_text()+'\nvoid handleSerialCommand() {')
    sketch=sketch.replace("  switch (command) {","  switch (command) {\n    case 't':\n    case 'T':\n      runBenchmarks();return;",1)
    sketch=sketch.replace('if(!trained_models::selfCheck())','if(!comparison_models::selfCheckAll())')
    sketch=sketch.replace('全4モデル','全8条件').replace('入力変動を追加学習した4モデル準備完了','V9のDense/CSR・全8条件準備完了')
    sketch=sketch.replace('③ 各色を当てて r を1回ずつ送信','③ 各色を当てて r を1回ずつ送信。速度・容量比較は t')
    (output/'compare_all_v9.ino').write_text(sketch)
    (HERE/'comparison_pc_summary.json').write_text(json.dumps({'source':'Frozen V9 models; no new sensor readings or board timings yet.',
        'conditions':metrics,'combined_dense_and_csr_payload_bytes':sum(r['dense_bytes']+r['csr_bytes'] for r in metrics),
        'comparison_header_sha256':sha(output/'comparison_parameters_v9.h'),
        'benchmark_inputs_sha256':sha(ROOT/'pruning/validation.csv'),'independent_final_evaluation':'NOT_PERFORMED'},indent=2)+'\n')
    print('Built eight-condition comparison: paired fresh RGB/MSE, model arrays, and t timing for all four H; code and data separate.')


if __name__=='__main__':build()
