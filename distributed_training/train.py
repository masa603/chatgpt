"""Compare data-only reliability learning against the distributed CMYK reference."""

import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import csv
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'retraining'))
from train_current33 import ColorNet,model_arrays,forward_fp32,export_header
from prepare import csv_write,sha

HIDDEN=[40,60,80,100]
CONFIGS=[{'name':'mse_baseline','reliability':False,'huber':False,'jitter':False},
         {'name':'reliability_huber','reliability':True,'huber':True,'jitter':False},
         {'name':'reliability_huber_jitter','reliability':True,'huber':True,'jitter':True}]


def arrays(rows):
    return np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32),np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in rows],np.float32)


def reliability(x,y):
    distance=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)
    np.fill_diagonal(distance,np.inf)
    neighbours=np.argsort(distance,axis=1,kind='stable')[:,:5]
    q=[];modes=[]
    for i,indices in enumerate(neighbours):
        unique,counts=np.unique(y[indices],axis=0,return_counts=True)
        choices=[]
        for label,count in zip(unique,counts):
            members=[j for j in indices if np.array_equal(y[j],label)]
            choices.append((-int(count),min(float(distance[i,j]) for j in members),tuple(label.tolist())))
        mode=np.array(min(choices)[2],np.float32)
        error=float(np.sqrt(np.mean(((y[i]-mode)/255)**2)))
        q.append(max(.1,1/(1+(error/.15)**2)));modes.append(mode)
    q=np.array(q,np.float32)
    contrast=np.sqrt(np.mean(((y[:,None,:]-y[None,:,:])/255)**2,axis=2))>.15
    mask=contrast&(q[None,:]>=.5)
    nearest=np.where(mask,distance,np.inf).min(1)
    widths=np.minimum(.03,.15*nearest).astype(np.float32)
    return q,np.array(modes),widths


def fit(rows,h,config,epochs):
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    x,y=arrays(rows);q,_,widths=reliability(x,y)
    tx,ty=torch.from_numpy(x),torch.from_numpy(y/255)
    weights=torch.from_numpy(q if config['reliability'] else np.ones(len(rows),np.float32))
    half=torch.from_numpy(widths[:,None])
    model=ColorNet(h).float();optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.00003)
    start=time.monotonic()
    for epoch in range(epochs):
        ax=torch.clamp(tx+(torch.rand_like(tx)*2-1)*half,0,1) if config['jitter'] else tx
        optimizer.zero_grad();output=model(ax)
        loss_rows=F.smooth_l1_loss(output,ty,beta=.05,reduction='none').mean(1) if config['huber'] else ((output-ty)**2).mean(1)
        loss=(loss_rows*weights).sum()/weights.sum()+1e-6*sum(p.abs().sum() for name,p in model.named_parameters() if 'weight' in name)
        assert torch.isfinite(loss)
        loss.backward();optimizer.step();schedule.step()
    model.eval();layers=model_arrays(model)
    pixels=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5)
    return model,layers,float(np.mean((pixels-y)**2)),time.monotonic()-start


def development(task):
    rows,h,config=task;training=[r for r in rows if r['group']!='v6_capture01'];held=[r for r in rows if r['group']=='v6_capture01']
    assert len(training)==93 and len(held)==9
    _,layers,training_mse,seconds=fit(training,h,config,20000)
    x,y=arrays(held);pixels=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5)
    return {'hidden_size':h,'method':config['name'],'training_rows':93,'development_rows':9,'epochs':20000,
            'training_mse_unweighted':training_mse,'development_mse_all9':float(np.mean((pixels-y)**2)),
            'development_black_mse':float(np.mean((pixels[6]-y[6])**2)),'development_gray_mse':float(np.mean((pixels[7]-y[7])**2)),
            'development_rgb9':pixels.astype(int).tolist(),'seconds':seconds,'reference':'DISTRIBUTED_NOTEBOOK_CMYK_DISPLAY_RECOVERED'}


def final(task):
    rows,h,config=task;output=HERE/f'models/h{h}';output.mkdir(parents=True,exist_ok=True)
    assert not (output/'model_parameters.h').exists()
    model,layers,mse,seconds=fit(rows,h,config,50000)
    data_hash=sha(HERE/'training102.csv');reference=HERE/'references/reference_image1_cmyk_3x3.ppm'
    header=output/'model_parameters.h';export_header(layers,h,data_hash,42,50000,header)
    header.write_text(header.read_text().replace('re-trained on the current 33 saved learning inputs.','trained on 102 user-measured inputs; distributed Notebook CMYK RGB targets; V7.'))
    np.savez(output/'model_arrays.npz',**{f'{kind}_{i}':v for i,(w,b) in enumerate(layers,1) for kind,v in [('weight',w),('bias',b)]})
    torch.save({'hidden_size':h,'epochs':50000,'state_dict':model.state_dict(),'config':config,'data_sha256':data_hash,'reference_sha256':sha(reference)},output/'checkpoint.pt')
    report={'hidden_size':h,'config':config,'training_rows':102,'distinct_reference_colors':33,'epochs':50000,'seed':42,
            'optimizer':'Adam LR0.003->0.00003 cosine; scratch initialization identical procedure for all H.','l1_weights_only':1e-6,
            'training_mse_unweighted_rgb8':mse,'training_mse_rgb8':mse,'wall_seconds':seconds,
            'training_data_sha256':data_hash,'reference_ppm_sha256':sha(reference),'model_header_sha256':sha(header),
            'scope':'After development condition selection, all102 used for final fitting. New V7 physical reading not received. No measured-only memorization finish phase.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    if (HERE/'training_protocol.json').exists():raise FileExistsError('Training already started; do not overwrite.')
    with (HERE/'training102.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==102
    x,y=arrays(rows);q,mode,widths=reliability(x,y)
    csv_write(HERE/'sample_reliability.csv',[{'sample_id':r['sample_id'],'group':r['group'],'chart_id':r['chart_id'],'color_id':r['color_id'],
               'reliability_weight':float(q[i]),'neighbour_mode_rgb':' '.join(str(int(v)) for v in mode[i]),'jitter_halfwidth':float(widths[i]),
               'interpretation':'Neighbour-label consistency; not proof of wrong label; all rows retained.'} for i,r in enumerate(rows)])
    (HERE/'training_protocol.json').write_text(json.dumps({
        'hidden_sizes':HIDDEN,'configs':CONFIGS,'seed':42,'development_epochs':20000,'final_epochs':50000,
        'development_split':'Latest V6 nine-color reading held out together; training93. Its labels are not used in parameter updates or reliability weights for that phase.',
        'selection':'Per H, lowest all-nine-color development RGB8 MSE among three declared learning conditions.',
        'development_limit':'Inspected reading used for design and selection; not independent final test. Final fitting includes all102 afterward.',
        'reliability':'Five nearest measured training inputs excluding self; mode of neighbour target RGB (tie: nearest distance then RGB tuple). Weight=max(0.1,1/(1+(RMS RGB difference/255/0.15)^2)). Training only, positive weights for every sample.',
        'robust_loss':'SmoothL1 beta0.05 on normalized RGB; evaluation remains ordinary unweighted RGB8 MSE.',
        'jitter':'Training only, uniform per-channel [-a,a], a=min(0.03,0.15*distance to nearest trusted (weight>=0.5) training point with RMS target RGB difference/255>0.15); clip0..1.',
        'sensor_and_normalization':'Unchanged. No geometry/photo prerequisite, no target/position-based prediction, no output palette snap.',
        'rows_removed':0,'provided_example_X_used':False,'new_physical_accuracy':'NOT_YET_MEASURED'},indent=2)+'\n')
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(development,(rows,h,c)) for h in HIDDEN for c in CONFIGS]):
            r=future.result();results.append(r);print('DEVELOPMENT',json.dumps(r),flush=True)
    (HERE/'development_results.json').write_text(json.dumps(results,indent=2)+'\n')
    csv_write(HERE/'development_results.csv',[{k:v for k,v in r.items() if k!='development_rgb9'} for r in sorted(results,key=lambda r:(r['hidden_size'],r['method']))])
    selected={}
    for h in HIDDEN:
        best=min([r for r in results if r['hidden_size']==h],key=lambda r:r['development_mse_all9'])
        selected[str(h)]={'config':next(c for c in CONFIGS if c['name']==best['method']),'development_result':best}
    (HERE/'selected_configs.json').write_text(json.dumps(selected,indent=2)+'\n')
    print('SELECTED',json.dumps({h:{'method':r['config']['name'],'development_mse':r['development_result']['development_mse_all9']} for h,r in selected.items()}),flush=True)
    finals=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(final,(rows,h,selected[str(h)]['config'])) for h in HIDDEN]):
            r=future.result();finals.append(r);print('FINAL',json.dumps(r),flush=True)
    csv_write(HERE/'training_summary.csv',[{'hidden_size':r['hidden_size'],'training_rows':102,'training_reference_colors':33,'training_mse_unweighted_rgb8':r['training_mse_rgb8'],'new_physical_mse':'PENDING','reference_ppm_sha256':r['reference_ppm_sha256']} for r in sorted(finals,key=lambda r:r['hidden_size'])])


if __name__=='__main__':main()
