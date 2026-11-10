"""Export exactly one frozen V10 H/format per preprocessing configuration."""

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT/'pruning')]
from prepare import sha
from threshold_tools import load_model,prune,storage
from export_csr_arduino import array_definition


def build():
    folder=HERE/'memory_probe_v10';folder.mkdir(exist_ok=True)
    header='''// Final V10 arrays. Only one H and one format survive preprocessing.
#pragma once
#include <stddef.h>
#include <stdint.h>
static_assert(sizeof(float)==4,"FP32 required");
#if PROBE_CSR != 0 && PROBE_CSR != 1
#error "PROBE_CSR must be 0 (Dense) or 1 (CSR)"
#endif
namespace memory_model {
struct DenseModel {int hidden;const float *w1,*b1,*w2,*b2,*w3,*b3;};
struct SparseLayer {const float *values,*bias;const uint8_t *col;const uint16_t *row;uint16_t rows,cols,nnz,stored;};
struct SparseModel {SparseLayer layers[3];};
constexpr int DECLARED_WORK_ARRAY_BYTES=800;
'''
    conditions=[]
    for index,h in enumerate([40,60,80,100]):
        source=HERE/f'models/h{h}/model_parameters.h';layers=load_model(source)
        summary=json.loads((HERE/f'selection/h{h}/summary.json').read_text());chosen=summary['selected']
        assert chosen and summary['model_sha256']==sha(source)
        sparse=prune(layers,chosen['threshold'])
        header+=('#if' if index==0 else '#elif')+f' PROBE_H == {h}\n'
        header+=f'constexpr int HIDDEN_SIZE={h};\nconst char MODEL_SOURCE_SHA256[]="{sha(source)}";\n'
        header+='#if PROBE_CSR == 0\nconstexpr double THRESHOLD=0;\n'
        for i,(w,b) in enumerate(layers,1):
            header+=array_definition(f'weight_{i}',w.ravel(),'float')+array_definition(f'bias_{i}',b,'float')
        header+='const DenseModel MODEL={'+str(h)+',weight_1,bias_1,weight_2,bias_2,weight_3,bias_3};\n'
        size=' + '.join(f'sizeof(weight_{i})+sizeof(bias_{i})' for i in [1,2,3])
        dense_bytes=storage(layers)['dense_bytes']
        header+=f'constexpr size_t MODEL_ARRAY_BYTES={size};\nstatic_assert(MODEL_ARRAY_BYTES=={dense_bytes},"Dense array bytes");\n'
        header+='#else\n'+f'constexpr double THRESHOLD={chosen["threshold"]:.17g};\n'
        definitions=[];size_terms=[];csr_bytes=0
        for i,(w,b) in enumerate(sparse,1):
            rr,cc=np.nonzero(w);nnz=len(rr);stored=max(nnz,1)
            values=w[rr,cc].astype(np.float32) if nnz else np.zeros(1,np.float32)
            cols=cc.astype(np.uint8) if nnz else np.zeros(1,np.uint8)
            row=np.concatenate([[0],np.cumsum(np.count_nonzero(w,axis=1))]).astype(np.uint16)
            header+=array_definition(f'values_{i}',values,'float')+array_definition(f'columns_{i}',cols,'uint8_t')
            header+=array_definition(f'rows_{i}',row,'uint16_t')+array_definition(f'bias_{i}',b,'float')
            definitions.append('{'+f'values_{i},bias_{i},columns_{i},rows_{i},{w.shape[0]},{w.shape[1]},{nnz},{stored}'+'}')
            size_terms.append(f'sizeof(values_{i})+sizeof(columns_{i})+sizeof(rows_{i})+sizeof(bias_{i})')
            csr_bytes+=values.nbytes+cols.nbytes+row.nbytes+b.nbytes
        assert csr_bytes==chosen['csr_bytes']
        header+='const SparseModel MODEL={{'+','.join(definitions)+'}};\n'
        header+='constexpr size_t MODEL_ARRAY_BYTES='+' + '.join(size_terms)+';\n'
        header+=f'static_assert(MODEL_ARRAY_BYTES=={csr_bytes},"CSR array bytes");\n#endif\n'
        for mode,fmt,size in [(0,'DENSE',dense_bytes),(1,'CSR',csr_bytes)]:
            conditions.append({'H':h,'format':fmt,'PROBE_CSR':mode,'model_array_bytes':size,
                'threshold':0 if mode==0 else chosen['threshold'],'model_source_sha256':sha(source)})
    header+='#else\n#error "PROBE_H must be40,60,80,or100"\n#endif\n}\n'
    (folder/'memory_model_v10.h').write_text(header)
    (folder/'conditions.json').write_text(json.dumps({'firmware':'V10_SINGLE_MODEL_MEMORY','scope':'Final V10 models; one model/format per build.',
        'training_data_sha256':sha(HERE/'training135.csv'),'reference_sha256':sha(HERE/'reference/result.ppm'),
        'conditions':conditions},indent=2)+'\n')
    with (HERE/'single_model_memory_template.csv').open('w',newline='') as f:
        fields=['H','format','status','model_array_bytes','flash_used_bytes','static_ram_used_bytes']
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for condition in conditions:
            writer.writerow({k:condition[k] for k in ['H','format','model_array_bytes']}|{'status':'PENDING_MAC_ARDUINO_BUILD'})
    print('Built guarded V10 numeric header,8 condition records; final frozen weights/thresholds preserved.')


if __name__=='__main__':build()
