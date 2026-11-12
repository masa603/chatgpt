"""Data-only neutral-color adaptation; latest full reading is development, then fitting."""

import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import csv
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'retraining'),str(ROOT/'learning_followup')]
from train_current33 import ColorNet,model_arrays,forward_fp32,export_header
from train_multiread import write_csv

REFERENCE=ROOT/'learning_followup/reference_audit/provided/result.ppm'
DEVELOPMENT_GROUP='v5_capture01'
CONFIGS=[
    {'name':'replay_only','gray_weight':1,'white_weight':1,'neutral_augmentation':False},
    {'name':'gray_focus','gray_weight':4,'white_weight':1,'neutral_augmentation':False},
    {'name':'neutral_augmentation','gray_weight':3,'white_weight':2,'neutral_augmentation':True},
    {'name':'neutral_augmentation_strong','gray_weight':6,'white_weight':3,'neutral_augmentation':True},
]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    with (ROOT/'current_input_adaptation/training60.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    source=ROOT/'current_input_adaptation/physical_readings/capture_01'
    with (source/'inputs.csv').open(newline='') as f: latest=list(csv.DictReader(f))
    summary=json.loads((source/'summary.json').read_text())
    raw=np.array([[r[f'raw_{c}'] for c in 'rgb'] for r in latest],np.float32)
    black,white=[np.array(summary[f'{kind}_rgb'],np.float32) for kind in ['black','white']]
    x=np.clip((raw-black)/(white-black),0,1)
    targets=np.array(REFERENCE.read_text().split()[4:],int).reshape(9,3)
    ids=[f'r{r}c{c}' for r in range(1,4) for c in range(1,4)]
    assert len(rows)==60 and [r['color_id'] for r in latest]==ids
    assert np.allclose(x,np.array([[r[f'input_{c}'] for c in 'rgb'] for r in latest],np.float32),atol=1e-7,rtol=1e-6)
    for i,r in enumerate(latest):
        rows.append({'sample_id':f'{DEVELOPMENT_GROUP}_{ids[i]}','group':DEVELOPMENT_GROUP,'color_id':ids[i],
                     **{f'input_{c}':format(float(x[i,j]),'.9g') for j,c in enumerate('rgb')},
                     **{f'target_{c}8':int(targets[i,j]) for j,c in enumerate('rgb')},
                     'source_role':'V5 new physical reading, now V6 development selection and final fitting'})
    assert len(rows)==69 and len({r['group'] for r in rows})==9
    for r in rows:assert [int(r[f'target_{c}8']) for c in 'rgb']==targets[ids.index(r['color_id'])].tolist()
    write_csv(HERE/'training69.csv',rows)
    parents={}
    for h in [40,60,100]:
        directory=ROOT/f'current_input_adaptation/models/h{h}'
        checkpoint=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert checkpoint['data_sha256']==sha(ROOT/'current_input_adaptation/training60.csv')
        assert checkpoint['reference_sha256']==sha(REFERENCE)
        assert DEVELOPMENT_GROUP not in {r['group'] for r in rows[:60]}
        parents[str(h)]={'checkpoint':str((directory/'checkpoint.pt').relative_to(ROOT)),
                         'checkpoint_sha256':sha(directory/'checkpoint.pt'),
                         'source_model_header_sha256':sha(directory/'model_parameters.h')}
    (HERE/'protocol.json').write_text(json.dumps({
        'user_request':'Improve using learning data, without requiring geometry/setup changes.',
        'reference':str(REFERENCE.relative_to(ROOT)),'reference_sha256':sha(REFERENCE),
        'base_rows':60,'new_rows':9,'all_rows':69,'reference_colors':9,
        'base_training_csv_sha256':sha(ROOT/'current_input_adaptation/training60.csv'),
        'new_source_log':str((source/'submitted_log.txt').relative_to(ROOT)),
        'new_source_log_sha256':sha(source/'submitted_log.txt'),
        'parent_models':parents,
        'development_split':'All 9 rows of latest V5 capture held out together from 60-row parameter updates. Parent V5 checkpoints also never trained on these 9 rows.',
        'selection':'For each H independently, minimum RGB8 MSE over all 9 development colors among four declared conditions. No pixel-random split.',
        'development_limit':'Latest reading already inspected and used to design/select conditions. It is development, not independent final test or new physical V6 result.',
        'final_role_change':'After condition selection, add latest 9 rows to final fitting on all69. Another new physical read required after V6 freeze.',
        'configs':CONFIGS,
        'development_epochs':15000,'final_epochs':30000,'seed':42,
        'optimizer':'Adam, cosine LR 0.0003 to 0.000003; initialize from each frozen V5 checkpoint.',
        'loss':'Equal total base weight per reference color, multiplied by declared gray/white weights; RGB MSE plus weights-only L1 1e-6.',
        'neutral_augmentation':'Training only: gray channel gains independently uniform[0.1,1.35], white uniform[0.6,1.25], clip0..1. Reject proposal within Euclidean distance0.025 of any differently labelled measured training input. Rejected proposals use original input.',
        'local_noise':'Sigma min(0.002,0.1*nearest differently labelled training input distance), at the proposed point. All measured rows remain in replay.',
        'inference':'Original 3->H->H->3 Dense FP32, original sensor acquisition, normalization and clipping unchanged; no palette snap, gray override or target/position-based prediction.',
        'preservation':'Keep all previous measurements and their actual high MSE. No reference-image changes or favorable-reading filtering.'},indent=2)+'\n')
    return rows


def fit(rows,h,config,epochs):
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    parent=ROOT/f'current_input_adaptation/models/h{h}/checkpoint.pt'
    checkpoint=torch.load(parent,map_location='cpu',weights_only=True)
    model=ColorNet(h).float();model.load_state_dict(checkpoint['state_dict'])
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32)
    y=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in rows],np.float32)
    tx,ty=torch.from_numpy(x),torch.from_numpy(y/255)
    same=torch.from_numpy((y[:,None,:]==y[None,:,:]).all(axis=2))
    original_dist=((tx[:,None,:]-tx[None,:,:])**2).sum(2).sqrt().masked_fill(same,float('inf')).min(1).values
    assert torch.all(original_dist>0)
    gray=torch.tensor([r['color_id']=='r3c2' for r in rows])
    white=torch.tensor([r['color_id']=='r3c3' for r in rows])
    counts=same.sum(1).float()
    weights=torch.ones(len(rows))/counts
    weights[gray]*=config['gray_weight'];weights[white]*=config['white_weight']
    low=torch.ones((len(rows),1));high=torch.ones((len(rows),1))
    if config['neutral_augmentation']:
        low[gray]=.1;high[gray]=1.35;low[white]=.6;high[white]=1.25
    opt=torch.optim.Adam(model.parameters(),lr=.0003)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=.000003)
    start=time.monotonic()
    for epoch in range(epochs):
        proposed=tx
        if config['neutral_augmentation']:
            gains=low+(high-low)*torch.rand_like(tx)
            proposed=torch.clamp(tx*gains,0,1)
        distances=((proposed[:,None,:]-tx[None,:,:])**2).sum(2).sqrt().masked_fill(same,float('inf')).min(1).values
        keep=distances>=.025
        augmented=torch.where(keep[:,None],proposed,tx)
        nearest=torch.where(keep,distances,original_dist)
        noise=torch.minimum(torch.full_like(nearest,.002),nearest*.1)[:,None]
        augmented=torch.clamp(augmented+torch.randn_like(tx)*noise,0,1)
        opt.zero_grad();prediction=model(augmented)
        loss=((((prediction-ty)**2).mean(1)*weights).sum()/weights.sum()+1e-6*sum(p.abs().sum() for name,p in model.named_parameters() if 'weight' in name))
        assert torch.isfinite(loss)
        loss.backward();opt.step();schedule.step()
    model.eval();layers=model_arrays(model)
    pixel=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5)
    return model,layers,float(np.mean((pixel-y)**2)),time.monotonic()-start


def development(task):
    rows,h,config=task
    training=[r for r in rows if r['group']!=DEVELOPMENT_GROUP]
    held=[r for r in rows if r['group']==DEVELOPMENT_GROUP]
    assert len(training)==60 and len(held)==9
    _,layers,train_mse,seconds=fit(training,h,config,15000)
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in held],np.float32)
    y=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in held],float)
    pixel=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5)
    return {'hidden_size':h,'method':config['name'],'training_rows':60,'held_out_group':DEVELOPMENT_GROUP,
            'development_rows':9,'epochs':15000,'training_mse':train_mse,'development_mse_all9':float(np.mean((pixel-y)**2)),
            'development_gray_mse':float(np.mean((pixel[7]-y[7])**2)),
            'development_white_mse':float(np.mean((pixel[8]-y[8])**2)),
            'development_rgb9':pixel.astype(int).tolist(),'seconds':seconds}


def final(task):
    rows,h,config=task;directory=HERE/f'models/h{h}';directory.mkdir(parents=True,exist_ok=True)
    assert not (directory/'model_parameters.h').exists()
    model,layers,mse,seconds=fit(rows,h,config,30000)
    data_hash=sha(HERE/'training69.csv');header=directory/'model_parameters.h'
    export_header(layers,h,data_hash,42,30000,header)
    header.write_text(header.read_text().replace('re-trained on the current 33 saved learning inputs.','fine-tuned on 69 measured inputs; unchanged provided result.ppm; V6.'))
    np.savez(directory/'model_arrays.npz',**{f'{kind}_{i}':v for i,(w,b) in enumerate(layers,1) for kind,v in [('weight',w),('bias',b)]})
    torch.save({'hidden_size':h,'epochs':30000,'state_dict':model.state_dict(),'config':config,'data_sha256':data_hash,'reference_sha256':sha(REFERENCE),'parent_checkpoint_sha256':sha(ROOT/f'current_input_adaptation/models/h{h}/checkpoint.pt')},directory/'checkpoint.pt')
    report={'hidden_size':h,'config':config,'training_rows':69,'distinct_colors':9,'epochs':30000,'seed':42,
            'optimizer':'Adam; cosine LR 0.0003 to 0.000003; initialized from V5 of same H.',
            'l1_weights_only':1e-6,'training_mse_rgb8':mse,'wall_seconds':seconds,
            'training_data_sha256':data_hash,'reference_ppm_sha256':sha(REFERENCE),
            'model_header_sha256':sha(header),'parent_checkpoint_sha256':sha(ROOT/f'current_input_adaptation/models/h{h}/checkpoint.pt'),
            'scope':'All69 fitting after development selection; latest V5 read now included. New prospective physical V6 read still required.'}
    (directory/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    if (HERE/'protocol.json').exists():raise FileExistsError('Experiment already started; do not overwrite.')
    rows=prepare();results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(development,(rows,h,c)) for h in [40,60,100] for c in CONFIGS]):
            r=future.result();results.append(r);print('DEVELOPMENT',json.dumps(r),flush=True)
    (HERE/'development_results.json').write_text(json.dumps(results,indent=2)+'\n')
    scalar=[{k:v for k,v in r.items() if k!='development_rgb9'} for r in sorted(results,key=lambda r:(r['hidden_size'],r['method']))]
    write_csv(HERE/'development_results.csv',scalar)
    selected={}
    for h in [40,60,100]:
        best=min([r for r in results if r['hidden_size']==h],key=lambda r:r['development_mse_all9'])
        config=next(c for c in CONFIGS if c['name']==best['method'])
        selected[str(h)]={'config':config,'development_result':best}
    (HERE/'selected_configs.json').write_text(json.dumps(selected,indent=2)+'\n')
    print('SELECTED',json.dumps({h:dict(method=s['config']['name'],development_mse=s['development_result']['development_mse_all9']) for h,s in selected.items()}),flush=True)
    finals=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(final,(rows,h,selected[str(h)]['config'])) for h in [40,60,100]]):
            r=future.result();finals.append(r);print('FINAL',json.dumps(r),flush=True)
    write_csv(HERE/'training_summary.csv',[{'hidden_size':r['hidden_size'],'training_rows':69,'training_mse_rgb8':r['training_mse_rgb8'],'new_physical_mse':'PENDING','reference_ppm_sha256':r['reference_ppm_sha256']} for r in sorted(finals,key=lambda r:r['hidden_size'])])


if __name__=='__main__':main()
