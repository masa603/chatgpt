"""Add the user's <=300 target as a training-only guard, before any new paired capture."""

import csv
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'pruning'))
import threshold_tools as tt
from export_csr_arduino import dense_layer_references


def finalize(h):
    directory=HERE/f'selection/h{h}'
    report=json.loads((directory/'summary.json').read_text())
    if 'training9_guard' in report:raise FileExistsError('Thresholds already frozen with training-only guard.')
    (directory/'plan_only_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (directory/'candidates.csv').open(newline='') as f:records=list(csv.DictReader(f))
    with (HERE/'training135.csv').open(newline='') as f:rows=[r for r in csv.DictReader(f) if r['group']=='v8_capture01']
    assert len(rows)==9
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32)
    y=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in rows],float)
    layers=tt.load_model(HERE/f'models/h{h}/model_parameters.h')
    decoded=[]
    for r in records:
        row={k:(v=='True' if k in ['memory_ok','quality_ok','eligible'] else int(v) if k in ['nonzero','csr_bytes','dense_bytes','cpp_padding_bytes'] else None if v=='' or v=='None' else float(v)) for k,v in r.items()}
        row['plan_eligible']=row['eligible'];row['training9_mse']=None;row['training9_goal_ok']=None
        decoded.append(row)
    selected=None
    for row in sorted((r for r in decoded if r['plan_eligible']),key=lambda r:(r['csr_bytes'],r['validation_mse'],r['threshold'])):
        sparse=tt.prune(layers,row['threshold'])
        output=dense_layer_references(sparse,x)[-1].astype(float)
        pixels=np.floor(np.clip(output,0,1)*255+.5)
        row['training9_mse']=float(np.mean((pixels-y)**2));row['training9_goal_ok']=row['training9_mse']<=300
        if row['training9_goal_ok']:
            selected=row;break
    # Candidates after the first accepted one are not evaluated on training9; they are not called failures.
    for row in decoded:row['eligible']=False if not row['plan_eligible'] else row['training9_goal_ok']
    report['selected']=selected
    report['training9_guard']={'source':'v8_capture01 now in training135, not independent validation/test',
        'limit_rgb8_mse':300,'purpose':'Additional user quality target on saved learning inputs; not a new physical accuracy claim.',
        'selection':'Start with plan-eligible candidates, test increasing CSR capacity (then validation MSE, threshold); stop at first passing training9<=300.'}
    if selected is None:
        original=json.loads((directory/'plan_only_summary.json').read_text())
        report['diagnostic_only']=original['selected'] or original['diagnostic_only']
        report['selection_status']='no_candidate_meets_plan_and_training_guard'
    else:
        report['diagnostic_only']=None
        tt.save_csr_npz(tt.prune(layers,selected['threshold']),directory/'selected_csr.npz')
    with (directory/'candidates.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(decoded[0]),lineterminator='\n');writer.writeheader();writer.writerows(decoded)
    (directory/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return {'H':h,'selection_status':report['selection_status'],'selected':selected or report['diagnostic_only'],
            'adopted_by_validation_and_training_guard':selected is not None,
            'scope':'Frozen before new comparison readings. Separate validation plus training-only guard; independent final chart unavailable.'}


if __name__=='__main__':
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(finalize,h) for h in [40,60,80,100]]):
            row=future.result();results.append(row);print(json.dumps(row),flush=True)
    (HERE/'compression_selection.json').write_text(json.dumps(sorted(results,key=lambda r:r['H']),indent=2)+'\n')
