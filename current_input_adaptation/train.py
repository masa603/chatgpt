"""Develop a new Dense model using archived measured low-input data, without leakage claims."""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'retraining'), str(ROOT/'learning_followup')]
from train_current33 import ColorNet, model_arrays, forward_fp32, export_header
from train_multiread import write_csv

CONFIGS = [{'name':'exact_measured','noise':0.0}, {'name':'local_noise_limited','noise':0.005}]
REFERENCE = ROOT/'learning_followup/reference_audit/provided/result.ppm'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(raw, summary):
    black, white = [np.array(summary[f'{kind}_rgb'],np.float32) for kind in ['black','white']]
    return np.clip((np.asarray(raw,np.float32)-black)/(white-black),0,1)


def partial(text):
    return '\n'.join(line.strip().strip('\\').strip() for line in text.splitlines())


def prepare():
    tokens=REFERENCE.read_text().split()
    assert tokens[:4]==['P3','3','3','255'] and len(tokens)==31
    reference=np.array(tokens[4:],int).reshape(9,3)
    ids=[f'r{r}c{c}' for r in range(1,4) for c in range(1,4)]
    with (ROOT/'reference_corrected/training45.csv').open(newline='') as f:
        source=list(csv.DictReader(f))
    rows=[{k:r[k] for k in ['sample_id','group','color_id','input_r','input_g','input_b','target_r8','target_g8','target_b8']} for r in source]
    for r in rows:r['source_role']='previous_v4_training'

    def append(sample_id,group,color,x,role):
        rows.append({'sample_id':sample_id,'group':group,'color_id':color,
                     **{f'input_{c}':format(float(x[j]),'.9g') for j,c in enumerate('rgb')},
                     **{f'target_{c}8':int(reference[ids.index(color),j]) for j,c in enumerate('rgb')},'source_role':role})

    directory=ROOT/'reference_corrected/physical_readings/capture_01'
    with (directory/'inputs.csv').open(newline='') as f: data=list(csv.DictReader(f))
    summary=json.loads((directory/'summary.json').read_text())
    x=normalized([[r[f'raw_{c}'] for c in 'rgb'] for r in data],summary)
    for i,r in enumerate(data):append('v4_capture01_'+r['color_id'],'v4_capture01',r['color_id'],x[i],'previous_v4_prospective_capture_now_development')
    diagnostics=ROOT/'reference_corrected/diagnostics'
    for key in ['white_check_01','white_check_02']:
        d=json.loads((diagnostics/key/'summary.json').read_text())
        x=np.asarray(d['recomputed_nn_input'] if key.endswith('01') else d['recomputed_nn_inputs'],np.float32).reshape(-1,3)
        for i,values in enumerate(x):append(f'{key}_{i+1}',key,'r3c3',values,'white_only_diagnostic_now_development')
    calibration=json.loads((diagnostics/'white_check_02/summary.json').read_text())
    text=partial((diagnostics/'blue_black_01/submitted_log.txt').read_text())
    raw=np.array(re.findall(r'RAW R:([0-9.]+) G:([0-9.]+) B:([0-9.]+)',text),np.float32)
    printed=np.array(re.findall(r'\nR:([0-9.]+) G:([0-9.]+) B:([0-9.]+)',text),float)
    x=normalized(raw,calibration)
    assert raw.shape==(2,3) and np.all(np.abs(printed-x)<.00005001)
    for i,color in enumerate(['r1c3','r3c1']):
        append(f'blue_black_01_{color}','white_check_02',color,x[i],'blue_black_diagnostic_same_calibration_as_white_check_02')
    assert len(rows)==60
    for r in rows:assert [int(r[f'target_{c}8']) for c in 'rgb']==reference[ids.index(r['color_id'])].tolist()
    write_csv(HERE/'training60.csv',rows)
    sources=[ROOT/'reference_corrected/training45.csv',directory/'submitted_log.txt',diagnostics/'white_check_01/submitted_log.txt',diagnostics/'white_check_02/submitted_log.txt',diagnostics/'blue_black_01/submitted_log.txt']
    (HERE/'provenance.json').write_text(json.dumps({
        'reference':str(REFERENCE.relative_to(ROOT)),'reference_sha256':file_hash(REFERENCE),
        'sources':[{'path':str(p.relative_to(ROOT)),'sha256':file_hash(p)} for p in sources],
        'rows':60,'full_chart_reading_groups':6,'partial_calibration_groups':2,'distinct_colors':9,
        'role_change':'V4 capture01 and all subsequent diagnostics now used for development/fitting. None is a prospective V5 test.',
        'calibration_grouping':'white_check_02 and blue_black_01 share a calibration and are held out together. white_check_01 uses another calibration.',
        'scope':'Same nine learning-chart colors; not original 33 colors or unseen-chart evaluation.',
        'preservation':'Earlier weights, measurements, and their reported failures retained. No selection of samples by favorable MSE.',
        'selection':'Compare two declared methods with H60, whole-calibration-group holdouts. Choose lowest mean MSE over six full-chart holdouts; partial-group metrics reported separately.',
        'final_training':'After selection fit all60, H40/60/100, 50000 epochs, seed42; original sensor acquisition and normalization unchanged.',
        'loss':'Equal total weight per reference color to avoid overweighting additional white diagnostics.',
        'augmentation':'No mixing across illumination sessions. Local noise method caps per-sample sigma to min(0.005, 0.1*nearest differently labelled input distance); exact method uses no noise.',
        'prospective_physical_accuracy':'NOT_YET_MEASURED'},indent=2)+'\n')
    return rows


def fit(rows,h,config,epochs):
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32)
    y=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in rows],np.float32)
    same=(y[:,None,:]==y[None,:,:]).all(axis=2)
    distance=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)
    nearest=np.where(same,np.inf,distance).min(axis=1)
    assert np.all(nearest>0), 'Identical inputs have different targets; cannot treat them as separable.'
    sigma=np.minimum(config['noise'],nearest*.1).astype(np.float32)
    weights=torch.from_numpy((1/same.sum(axis=1)).astype(np.float32))
    tx,ty=torch.from_numpy(x),torch.from_numpy(y/255)
    noise=torch.from_numpy(sigma[:,None])
    model=ColorNet(h).float();opt=torch.optim.Adam(model.parameters(),lr=.003)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=3e-5)
    start=time.monotonic()
    for epoch in range(epochs):
        ax=torch.clamp(tx+torch.randn_like(tx)*noise,0,1) if config['noise'] else tx
        opt.zero_grad();prediction=model(ax)
        loss=((((prediction-ty)**2).mean(dim=1)*weights).sum()/weights.sum()+1e-6*sum(p.abs().sum() for name,p in model.named_parameters() if 'weight' in name))
        assert torch.isfinite(loss)
        loss.backward();opt.step();schedule.step()
    model.eval();layers=model_arrays(model)
    pixel=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5)
    return model,layers,float(np.mean((pixel-y.astype(float))**2)),time.monotonic()-start,float(sigma.min()),float(sigma.max())


def cv(task):
    rows,config,group=task
    train=[r for r in rows if r['group']!=group];test=[r for r in rows if r['group']==group]
    _,layers,train_mse,seconds,low,high=fit(train,60,config,10000)
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in test],np.float32)
    y=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in test],float)
    predicted=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5)
    return {'method':config['name'],'held_out_group':group,'validation_rows':len(test),'full_chart':len(test)==9,
            'hidden_size':60,'epochs':10000,'training_mse':train_mse,'held_out_mse':float(np.mean((predicted-y)**2)),
            'training_sigma_min':low,'training_sigma_max':high,'seconds':seconds}


def final(task):
    rows,config,h=task
    output=HERE/f'models/h{h}';output.mkdir(parents=True,exist_ok=True)
    assert not (output/'model_parameters.h').exists()
    model,layers,mse,seconds,low,high=fit(rows,h,config,50000)
    data_hash=file_hash(HERE/'training60.csv')
    header=output/'model_parameters.h'
    export_header(layers,h,data_hash,42,50000,header)
    header.write_text(header.read_text().replace('re-trained on the current 33 saved learning inputs.','adapted on 60 measured inputs; provided result.ppm targets; V5.'))
    np.savez(output/'model_arrays.npz',**{f'{kind}_{i}':v for i,(w,b) in enumerate(layers,1) for kind,v in [('weight',w),('bias',b)]})
    torch.save({'hidden_size':h,'epochs':50000,'state_dict':model.state_dict(),'config':config,'data_sha256':data_hash,'reference_sha256':file_hash(REFERENCE)},output/'checkpoint.pt')
    report={'hidden_size':h,'config':config,'training_rows':60,'distinct_colors':9,'epochs':50000,'seed':42,
            'optimizer':'Adam; cosine LR 0.003 to 0.00003','l1_weights_only':1e-6,'training_mse_rgb8':mse,
            'training_noise_sigma_min':low,'training_noise_sigma_max':high,'wall_seconds':seconds,
            'training_data_sha256':data_hash,'reference_ppm_sha256':file_hash(REFERENCE),'model_header_sha256':file_hash(header),
            'scope':'Training/development only. Includes previous V4 failed capture and diagnostics. Requires a new physical V5 read.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    if (HERE/'provenance.json').exists():raise FileExistsError('Adaptation already started; do not overwrite.')
    rows=prepare();groups=list(dict.fromkeys(r['group'] for r in rows));results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(cv,(rows,c,g)) for c in CONFIGS for g in groups]):
            r=future.result();results.append(r);print('CV',json.dumps(r),flush=True)
    write_csv(HERE/'development_holdouts.csv',sorted(results,key=lambda r:(r['method'],r['held_out_group'])))
    scores=[{'method':c['name'],'full_chart_mean_mse':float(np.mean([r['held_out_mse'] for r in results if r['method']==c['name'] and r['full_chart']]))} for c in CONFIGS]
    write_csv(HERE/'development_method_comparison.csv',scores)
    selected=min(scores,key=lambda r:r['full_chart_mean_mse']);config=next(c for c in CONFIGS if c['name']==selected['method'])
    (HERE/'selected_config.json').write_text(json.dumps({'config':config,'selection':selected,'prospective_mse':'NOT_MEASURED'},indent=2)+'\n')
    print('SELECTED',json.dumps(selected),flush=True)
    reports=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(final,(rows,config,h)) for h in [40,60,100]]):
            r=future.result();reports.append(r);print('FINAL',json.dumps(r),flush=True)
    write_csv(HERE/'training_summary.csv',[{'hidden_size':r['hidden_size'],'training_rows':60,'training_mse_rgb8':r['training_mse_rgb8'],'new_physical_mse':'PENDING','reference_ppm_sha256':r['reference_ppm_sha256']} for r in sorted(reports,key=lambda r:r['hidden_size'])])


if __name__=='__main__':main()
