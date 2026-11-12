"""Freeze teacher-reference V10 and select CSR thresholds on the archived, separate validation chart."""

import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import sys
import csv

import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'pruning'))
import threshold_tools as thresholds
from export_csr_arduino import dense_layer_references


def device_rgb8(layers,inputs):
    # Match the sketch's double-literal final scaling exactly after FP32 inference.
    values=dense_layer_references(layers,np.asarray(inputs,np.float32))[-1].astype(np.float64)
    return np.floor(np.clip(values,0,1)*255+.5).astype(np.uint8)


thresholds.predict_rgb8=device_rgb8


def select(h):
    directory=HERE/f'selection/h{h}'
    layers,records,report=thresholds.run(HERE/f'models/h{h}/model_parameters.h',directory,ROOT/'pruning/validation.csv',expected_reads=2)
    maxima=[float(np.abs(w).max()) for w,b in layers]
    for row in records:
        padding=5*sum(maximum==0 or row['threshold']>maximum for maximum in maxima)
        row['cpp_padding_bytes']=padding;row['csr_bytes']+=padding
        row['memory_ok']=row['csr_bytes']<row['dense_bytes'];row['eligible']=row['memory_ok'] and row['quality_ok']
    report['selected']=thresholds.choose_threshold(records)
    report['diagnostic_only']=None if report['selected'] is not None else min(records,key=lambda r:(r['csr_bytes'],r['validation_mse'],r['threshold']))
    report['postprocessing']='Sequential FP32 inference; double-literal255 scaling and half-up rounding, same as Arduino sketch.'
    report['cpp_padding_rule']='For each empty CSR layer store one float and one uint8 placeholder:5 bytes; count them in candidate capacity.'
    if report['selected'] is not None:thresholds.save_csr_npz(thresholds.prune(layers,report['selected']['threshold']),directory/'selected_csr.npz')
    with (directory/'candidates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    (directory/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    chosen=report['selected'] or report['diagnostic_only']
    assert chosen is not None
    return {'H':h,'selection_status':report['selection_status'],'selected':chosen,
            'adopted_by_validation_rule':report['selected'] is not None,
            'note':'Provisional because archived validation has input collisions; not a claim of independent final accuracy.'}


if __name__=='__main__':
    output=HERE/'compression_selection.json'
    if output.exists():raise FileExistsError('Frozen selection already exists; do not overwrite.')
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(select,h) for h in [40,60,80,100]]):
            r=future.result();results.append(r);print(json.dumps(r),flush=True)
    output.write_text(json.dumps(sorted(results,key=lambda r:r['H']),indent=2)+'\n')
