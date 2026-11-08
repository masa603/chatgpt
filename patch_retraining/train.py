"""Compare labelled-patch fine tuning, keeping the V7 nine-color read out of updates."""

import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import csv
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'retraining'),str(HERE)]
from train_current33 import ColorNet,model_arrays,forward_fp32,export_header
from prepare import read_csv,write_csv,sha

HIDDEN=[40,60,80,100]
CONFIGS=[{'name':'labelled_mse','huber':False,'neutral_exposure':False},
         {'name':'labelled_huber','huber':True,'neutral_exposure':False},
         {'name':'labelled_huber_neutral_exposure','huber':True,'neutral_exposure':True}]


def arrays(rows):
    return np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32),np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in rows],np.float32)


def neighbour_weights(x,y):
    # Same five-neighbour label-consistency rule used in V7, without importing its training runner.
    distances=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)
    np.fill_diagonal(distances,np.inf)
    neighbours=np.argsort(distances,axis=1,kind='stable')[:,:5]
    result=[]
    for i,indices in enumerate(neighbours):
        unique,counts=np.unique(y[indices],axis=0,return_counts=True)
        choices=[(-int(count),min(float(distances[i,j]) for j in indices if np.array_equal(y[j],label)),tuple(label.tolist())) for label,count in zip(unique,counts)]
        mode=np.array(min(choices)[2],np.float32)
        error=float(np.sqrt(np.mean(((y[i]-mode)/255)**2)))
        result.append(max(.1,1/(1+(error/.15)**2)))
    return np.array(result,np.float32)


def row_weights(rows):
    x,y=arrays(rows);q=neighbour_weights(x,y)
    count=Counter((r['group'],r['chart_id'],r['color_id']) for r in rows)
    groups={color:{r['group'] for r in rows if r['chart_id']=='train_3x3' and r['color_id']==color}
            for color in {r['color_id'] for r in rows if r['chart_id']=='train_3x3'}}
    values=[]
    for i,r in enumerate(rows):
        if r['chart_id']=='train_3x3':
            # Equal contribution per target color, then per calibration group.
            values.append(1/len(groups[r['color_id']])/count[r['group'],r['chart_id'],r['color_id']])
        else:
            # Keep original sparse chart examples with positive, documented weight.
            values.append(.1*float(q[i]))
    return np.array(values,np.float32)


def fit(rows,h,config,epochs,initial=None):
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    parent=ROOT/f'distributed_training/models/h{h}/checkpoint.pt'
    state=torch.load(parent if initial is None else initial,map_location='cpu',weights_only=True)['state_dict']
    model=ColorNet(h).float();model.load_state_dict(state)
    x,y=arrays(rows);tx=torch.from_numpy(x);ty=torch.from_numpy(y/255)
    weights=torch.from_numpy(row_weights(rows))
    neutral=torch.tensor([r['chart_id']=='train_3x3' and r['color_id'] in ['r3c2','r3c3'] for r in rows])[:,None]
    optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.000003)
    start=time.monotonic()
    for _ in range(epochs):
        ax=tx
        if config['neutral_exposure']:
            multiplier=.5+.65*torch.rand((len(rows),1))
            apply=neutral&(torch.rand((len(rows),1))<.5)
            ax=torch.clamp(torch.where(apply,tx*multiplier,tx),0,1)
        optimizer.zero_grad();prediction=model(ax)
        losses=F.smooth_l1_loss(prediction,ty,beta=.05,reduction='none').mean(1) if config['huber'] else ((prediction-ty)**2).mean(1)
        loss=(losses*weights).sum()/weights.sum()+1e-6*sum(p.abs().sum() for name,p in model.named_parameters() if 'weight' in name)
        assert torch.isfinite(loss)
        loss.backward();optimizer.step();schedule.step()
    model.eval();layers=model_arrays(model)
    pixels=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5)
    return model,layers,float(np.mean((pixels-y)**2)),time.monotonic()-start


def development(task):
    rows,h,config=task
    training=[r for r in rows if r['group']!='v7_capture01'];held=[r for r in rows if r['group']=='v7_capture01']
    assert len(training)==117 and len(held)==9
    model,layers,training_mse,seconds=fit(training,h,config,15000)
    x,y=arrays(held);pixels=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5).astype(int)
    output=HERE/f'development_models/h{h}/{config["name"]}';output.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'hidden_size':h,'config':config},output/'checkpoint.pt')
    result={'hidden_size':h,'method':config['name'],'training_rows':117,'development_rows':9,'epochs':15000,
            'training_mse_unweighted_rgb8':training_mse,'development_mse_all9':float(np.mean((pixels-y)**2)),
            'development_rgb9':pixels.tolist(),'checkpoint':str((output/'checkpoint.pt').relative_to(HERE)),
            'scope':'Saved V7 read used for inspected development; not a V8 physical measurement or independent final evaluation.',
            'seconds':seconds}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result


def final(task):
    rows,h,selected=task;config=next(c for c in CONFIGS if c['name']==selected['method'])
    output=HERE/f'models/h{h}';output.mkdir(parents=True,exist_ok=True)
    assert not (output/'model_parameters.h').exists()
    model,layers,mse,seconds=fit(rows,h,config,25000,HERE/selected['checkpoint'])
    data=HERE/'training126.csv';reference=ROOT/'distributed_training/references/reference_image1_cmyk_3x3.ppm'
    header=output/'model_parameters.h';export_header(layers,h,sha(data),42,40000,header)
    header.write_text(header.read_text().replace('re-trained on the current 33 saved learning inputs.','V8: fine-tuned on126 measured inputs including15 labelled patch samples; distributed CMYK targets.'))
    np.savez(output/'model_arrays.npz',**{f'{kind}_{i}':v for i,(w,b) in enumerate(layers,1) for kind,v in [('weight',w),('bias',b)]})
    torch.save({'state_dict':model.state_dict(),'hidden_size':h,'config':config,'data_sha256':sha(data),'reference_sha256':sha(reference)},output/'checkpoint.pt')
    x,y=arrays(rows);pixels=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5).astype(int)
    replay={}
    for group in ['v7_capture01','patch_session01_cal1']:
        mask=np.array([r['group']==group for r in rows]);replay[group]=float(np.mean((pixels[mask]-y[mask])**2))
    result={'hidden_size':h,'config':config,'parent_v7_checkpoint_sha256':sha(ROOT/f'distributed_training/models/h{h}/checkpoint.pt'),
            'selected_development_checkpoint_sha256':sha(HERE/selected['checkpoint']),
            'development_epochs':15000,'final_epochs':25000,'training_rows':126,'seed':42,'l1_weights_only':1e-6,
            'training_mse_rgb8':mse,'training_replay_by_group':replay,'seconds':seconds,
            'training_data_sha256':sha(data),'reference_ppm_sha256':sha(reference),'model_header_sha256':sha(header),
            'scope':'All126 fitting after method selection. Group replay now uses training data. No new V8 physical result.'}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    write_csv(output/'training_predictions.csv',[{'sample_id':r['sample_id'],'group':r['group'],'color_id':r['color_id'],
              **{f'predicted_{c}8':int(pixels[i,j]) for j,c in enumerate('rgb')},
              **{f'target_{c}8':int(y[i,j]) for j,c in enumerate('rgb')},'role':'TRAINING_REPLAY_NOT_NEW_PHYSICAL_RESULT'} for i,r in enumerate(rows)])
    return result


def main():
    protocol=HERE/'training_protocol.json'
    if protocol.exists():raise FileExistsError('Do not overwrite completed or running training.')
    rows=read_csv(HERE/'training126.csv');assert len(rows)==126
    protocol.write_text(json.dumps({'hidden_sizes':HIDDEN,'configs':CONFIGS,'seed':42,
        'parent':'V7 checkpoint per H,102 fitting rows, never trained on held-out V7 physical capture.',
        'development_rows':9,'development_training_rows':117,'development_epochs':15000,'final_rows':126,'final_epochs':25000,
        'optimizer':'Adam cosine0.0003->0.000003; weights-onlyL1 coefficient1e-6; unchanged architecture.',
        'weights':'Each3x3 target has total weight1, split equally across calibration groups and their samples; original6x4 rows keep positive0.1 times V7 neighbour consistency weight. No measured rows removed.',
        'augmentation':'Only in third condition: on50% of gray/white3x3 examples, multiply all input channels by one uniform factor0.5..1.15, clip0..1. Training only; no inferred palette or image-position output.',
        'selection':'Lowest all9 ordinary RGB8 development MSE per H. Fixed candidates and epochs, no cherry-picking pixels.',
        'role_change':'V7 failed nine-color read inspected for design and method selection, then included in final fitting. Not independent final evaluation.',
        'physical_v8':'NOT_YET_MEASURED','patch_data':'All15 actual samples retained, including variable first purple batch. Repeats within a placement are not independent chart captures.'},indent=2)+'\n')
    for kind,subset in [('final',rows),('development',[r for r in rows if r['group']!='v7_capture01'])]:
        weights=row_weights(subset)
        write_csv(HERE/f'sample_weights_{kind}.csv',[{'sample_id':r['sample_id'],'group':r['group'],'weight':float(weights[i])} for i,r in enumerate(subset)])
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(development,(rows,h,c)) for h in HIDDEN for c in CONFIGS]):
            result=future.result();results.append(result);print('DEVELOPMENT',json.dumps(result),flush=True)
    (HERE/'development_results.json').write_text(json.dumps(results,indent=2)+'\n')
    write_csv(HERE/'development_results.csv',[{k:v for k,v in r.items() if k!='development_rgb9'} for r in sorted(results,key=lambda r:(r['hidden_size'],r['method']))])
    selected={str(h):min((r for r in results if r['hidden_size']==h),key=lambda r:r['development_mse_all9']) for h in HIDDEN}
    (HERE/'selected_configs.json').write_text(json.dumps(selected,indent=2)+'\n')
    print('SELECTED',json.dumps({h:{'method':r['method'],'mse':r['development_mse_all9']} for h,r in selected.items()}),flush=True)
    final_results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(final,(rows,h,selected[str(h)])) for h in HIDDEN]):
            result=future.result();final_results.append(result);print('FINAL',json.dumps(result),flush=True)
    write_csv(HERE/'training_summary.csv',[{'hidden_size':r['hidden_size'],'training_rows':126,
              'training_mse_unweighted_rgb8':r['training_mse_rgb8'],
              'old_v7_chart_training_replay_mse':r['training_replay_by_group']['v7_capture01'],
              'patch15_training_replay_mse':r['training_replay_by_group']['patch_session01_cal1'],
              'new_physical_v8_mse':'PENDING'} for r in sorted(final_results,key=lambda r:r['hidden_size'])])


if __name__=='__main__':main()
