"""Adapt Dense models to actual repeated acquisitions, with reading-group validation."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "retraining"))
from train_current33 import ColorNet, model_arrays, forward_fp32, export_header

CONFIGS = [
    {"name":"all60", "mode":"all", "mix":False, "noise":0.0},
    {"name":"reliable56_mix", "mode":"reliable", "mix":True, "noise":.01},
    {"name":"chart36_mix_noise", "mode":"chart", "mix":True, "noise":.01},
    {"name":"chart36_mix", "mode":"chart", "mix":True, "noise":0.0},
]
GROUPS = ["original_3x3", "reread_01", "reread_03", "reread_latest"]


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader();w.writerows(rows)


def dataset():
    with (ROOT/'retraining/data/current33.csv').open(newline='') as f: original=list(csv.DictReader(f))
    rows=[]
    for r in original:
        rows.append({"sample_id":"original_"+r['sample_id'],"group":"original_3x3" if r['chart_id']=='train_3x3' else 'original_6x4',
                     "input_r":r['input_r'],"input_g":r['input_g'],"input_b":r['input_b'],
                     **{f'target_{c}8':r[f'target_{c}8'] for c in 'rgb'},
                     "zero_input_conflict":int(int(r['sample_id']) in [13,19,22,32])})
    sources=[('reread_01',ROOT/'experiment_today/fresh_readings/capture_01'),('reread_03',ROOT/'experiment_today/fresh_readings/capture_03'),('reread_latest',ROOT/'experiment_today/version_mismatch/old_firmware_retry_01')]
    provenance=[]
    for group,p in sources:
        s=json.loads((p/'summary.json').read_text())
        with (p/'inputs.csv').open(newline='') as f: data=list(csv.DictReader(f))
        raw=np.array([[r[f'raw_{c}'] for c in 'rgb'] for r in data],np.float32)
        b,w=np.array(s['black_rgb'],np.float32),np.array(s['white_rgb'],np.float32)
        x=np.clip((raw-b)/(w-b),0,1)
        for i,(r,values) in enumerate(zip(data,x)):
            rows.append({'sample_id':group+'_'+r['color_id'],'group':group,
                         **{f'input_{c}':format(float(values[j]),'.9g') for j,c in enumerate('rgb')},
                         **{f'target_{c}8':r[f'target_{c}8'] for c in 'rgb'},'zero_input_conflict':0})
        provenance.append({'group':group,'raw_log':str((p/'raw_log.txt').relative_to(ROOT)),
                           'sha256':hashlib.sha256((p/'raw_log.txt').read_bytes()).hexdigest()})
    assert len(rows)==60
    write_csv(HERE/'data_all60.csv',rows)
    (HERE/'data_provenance.json').write_text(json.dumps({'rows':60,'three_by_three_readings':4,
        'original_rows':33,'additional_measured_rows':27,'sources':provenance,
        'invalid_capture_2':'Excluded for user-reported wrong patch; never included.',
        'zero_conflict_records':'Four original records have identical all-zero NN inputs and different targets; retained in all60, excluded as information-lost records only in reliable56.',
        'chart_only_mode':'36 rows of the same nine colors across four readings. Scope is current 3x3-chart adaptation, not 33-color/unseen-chart generalization.',
        'roles':'Four reading groups used for development cross-validation and final fitting. None is an independent final evaluation after this change. A fresh prospective reading is required.'},indent=2)+'\n')
    return rows


def select_rows(rows,config):
    if config['mode']=='chart': return [r for r in rows if r['group']!='original_6x4']
    if config['mode']=='reliable': return [r for r in rows if not int(r['zero_input_conflict'])]
    return rows


def fit(rows,h,config,epochs):
    torch.set_num_threads(1); torch.manual_seed(42);np.random.seed(42)
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32)
    y8=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in rows],float)
    tx,ty=torch.from_numpy(x),torch.from_numpy((y8/255).astype(np.float32))
    model=ColorNet(h).float().cpu();opt=torch.optim.Adam(model.parameters(),lr=.003)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=3e-5)
    partners=[]
    for target in y8:
        group=np.flatnonzero(np.all(y8==target,axis=1)).tolist()
        partners.append([group[i%len(group)] for i in range(4)])
    partners=torch.tensor(partners,dtype=torch.long); indices=torch.arange(len(rows))
    start=time.monotonic()
    for epoch in range(epochs):
        augmented=tx
        if config['mix']:
            chosen=partners[indices,torch.randint(0,4,(len(rows),))]
            alpha=torch.rand((len(rows),1))
            augmented=alpha*tx+(1-alpha)*tx[chosen]
        if config['noise']: augmented=torch.clamp(augmented+torch.randn_like(augmented)*config['noise'],0,1)
        opt.zero_grad();out=model(augmented)
        loss=torch.mean((out-ty)**2)+1e-6*sum(p.abs().sum() for name,p in model.named_parameters() if 'weight' in name)
        loss.backward();opt.step();schedule.step()
    model.eval();layers=model_arrays(model)
    output=forward_fp32(layers,x);pixels=np.floor(np.clip(output.astype(float),0,1)*255+.5)
    return model,layers,float(np.mean((pixels-y8)**2)),time.monotonic()-start


def cv_worker(task):
    rows,config,held,epochs=task
    available=select_rows(rows,config);training=[r for r in available if r['group']!=held]
    validation=[r for r in rows if r['group']==held]
    _,layers,train_mse,seconds=fit(training,60,config,epochs)
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in validation],np.float32)
    y=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in validation],float)
    pred=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5)
    return {'config':config['name'],'hidden_size':60,'held_out_reading':held,'training_rows':len(training),
            'validation_rows':9,'epochs':epochs,'training_mse':train_mse,'held_out_mse':float(np.mean((pred-y)**2)),'seconds':seconds}


def final_worker(task):
    rows,config,h,epochs=task; selected=select_rows(rows,config)
    output=HERE/f'models/h{h}';output.mkdir(parents=True,exist_ok=True)
    if (output/'model_parameters.h').exists(): raise FileExistsError('Final model already frozen; use another experiment directory.')
    model,layers,mse,seconds=fit(selected,h,config,epochs)
    data=HERE/'data_selected.csv';datahash=hashlib.sha256(data.read_bytes()).hexdigest()
    export_header(layers,h,datahash,42,epochs,output/'model_parameters.h')
    text=(output/'model_parameters.h').read_text().replace('re-trained on the current 33 saved learning inputs.',f'adapted to {len(selected)} multisession learning inputs; scope recorded in summary.json.')
    (output/'model_parameters.h').write_text(text)
    np.savez(output/'model_arrays.npz',**{f'{kind}_{i}':v for i,(w,b) in enumerate(layers,1) for kind,v in [('weight',w),('bias',b)]})
    torch.save({'hidden_size':h,'epochs':epochs,'state_dict':model.state_dict(),'config':config,'data_sha256':datahash},output/'checkpoint.pt')
    report={'hidden_size':h,'config':config,'training_rows':len(selected),'epochs':epochs,'seed':42,
            'optimizer':'Adam, cosine learning rate 0.003 to 0.00003','l1_weights_only':1e-6,
            'training_mse_rgb8':mse,'wall_seconds':seconds,'training_data_sha256':datahash,
            'model_header_sha256':hashlib.sha256((output/'model_parameters.h').read_bytes()).hexdigest(),
            'scope':'Multiread adaptation. Training fit is not a prospective device measurement. All available reading groups used after development selection.',
            'next_test':'Freeze now, then acquire a new physical 3x3 reading; do not treat old training read replays as confirmation.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cv-epochs',type=int,default=10000);parser.add_argument('--epochs',type=int,default=50000);args=parser.parse_args()
    HERE.mkdir(exist_ok=True)
    if (HERE/'selected_config.json').exists():raise FileExistsError('Frozen adaptation already exists; do not overwrite.')
    rows=dataset();jobs=[(rows,c,g,args.cv_epochs) for c in CONFIGS for g in GROUPS];results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(cv_worker,j) for j in jobs]):
            r=future.result();results.append(r);print('CV',json.dumps(r),flush=True)
    results.sort(key=lambda r:(r['config'],r['held_out_reading']));write_csv(HERE/'development_cv.csv',results)
    scores=[{'config':c['name'],'mean_held_out_mse':float(np.mean([r['held_out_mse'] for r in results if r['config']==c['name']]))} for c in CONFIGS]
    write_csv(HERE/'development_cv_summary.csv',scores);best=min(scores,key=lambda r:r['mean_held_out_mse']);config=next(c for c in CONFIGS if c['name']==best['config'])
    selection={'selected_config':config,'development_selection_hidden_size':60,'development_mean_mse':best['mean_held_out_mse'],
               'selection_rule':'Minimum mean MSE over four whole-reading holdouts, not random pixel split. Development selection only; new physical reading still required.',
               'goal':'User tightened target to MSE <= 300 on a NEW physical reading, ideally below 100. Neither training fit nor development selection proves this target.'}
    (HERE/'selected_config.json').write_text(json.dumps(selection,indent=2)+'\n');write_csv(HERE/'data_selected.csv',select_rows(rows,config))
    print('SELECTED',json.dumps(selection),flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(final_worker,(rows,config,h,args.epochs)) for h in [40,60,100]]):
            print('FINAL',json.dumps(future.result()),flush=True)


if __name__=='__main__':main()
