"""One fixed reference-correction run; latest physical nine-color input remains held out."""

import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT/'retraining')]
from prepare import read_csv,write_csv,sha
from data_weights import arrays,row_weights
from train_current33 import ColorNet,model_arrays,forward_fp32,export_header

HIDDEN=[40,60,80,100]
EPOCHS=40000


def train(h):
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    rows=read_csv(HERE/'training135.csv');held=read_csv(HERE/'heldout9.csv')
    assert len(rows)==135 and len(held)==9 and all(r['group']!='v9_capture01' for r in rows)
    x,y=arrays(rows);hx,hy=arrays(held)
    model=ColorNet(h).float()
    parent=ROOT/f'calibration_retraining/models/h{h}/checkpoint.pt'
    model.load_state_dict(torch.load(parent,map_location='cpu',weights_only=True)['state_dict'])
    def rgb(layers,inputs):return np.floor(np.clip(forward_fp32(layers,inputs).astype(float),0,1)*255+.5).astype(int)
    before=rgb(model_arrays(model),hx);baseline=float(np.mean((before-hy)**2))
    tx=torch.from_numpy(x);ty=torch.from_numpy(y/255);weights=torch.from_numpy(row_weights(rows,True))
    neutral=torch.tensor([r['chart_id']=='train_3x3' and r['color_id'] in ['r3c2','r3c3'] for r in rows])[:,None]
    chart=torch.tensor([r['chart_id']=='train_3x3' for r in rows])[:,None]
    distance=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)
    contrast=np.sqrt(np.mean(((y[:,None,:]-y[None,:,:])/255)**2,axis=2))>.15
    width=np.minimum(.02,.1*np.where(contrast,distance,np.inf).min(1)).astype(np.float32)
    half=torch.from_numpy(width[:,None])
    optimizer=torch.optim.Adam(model.parameters(),lr=.0005)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=.000005)
    begin=time.monotonic()
    for _ in range(EPOCHS):
        scale=.5+.65*torch.rand((len(rows),1))
        apply=neutral&(torch.rand((len(rows),1))<.3)
        neutral_input=torch.clamp(torch.where(apply,tx*scale,tx),0,1)
        gain=.75+.55*torch.rand_like(tx);offset=(torch.rand_like(tx)*2-1)*half
        augment=chart&(torch.rand((len(rows),1))<.5)
        ax=torch.clamp(torch.where(augment,neutral_input*gain+offset,neutral_input),0,1)
        optimizer.zero_grad();prediction=model(ax)
        loss_rows=F.smooth_l1_loss(prediction,ty,beta=.05,reduction='none').mean(1)
        loss=(loss_rows*weights).sum()/weights.sum()+1e-6*sum(p.abs().sum() for name,p in model.named_parameters() if 'weight' in name)
        assert torch.isfinite(loss);loss.backward();optimizer.step();schedule.step()
    model.eval();layers=model_arrays(model);train_rgb=rgb(layers,x);held_rgb=rgb(layers,hx)
    output=HERE/f'models/h{h}';output.mkdir(parents=True,exist_ok=True)
    assert not (output/'model_parameters.h').exists()
    reference=HERE/'reference/result.ppm';data=HERE/'training135.csv'
    header=output/'model_parameters.h';export_header(layers,h,sha(data),42,EPOCHS,header)
    header.write_text(header.read_text().replace('re-trained on the current 33 saved learning inputs.','V10:135 measured training rows;3x3 targets fixed to teacher-required original result.ppm; latest9 held out.'))
    np.savez(output/'model_arrays.npz',**{f'{kind}_{i}':v for i,(w,b) in enumerate(layers,1) for kind,v in [('weight',w),('bias',b)]})
    torch.save({'state_dict':model.state_dict(),'hidden_size':h,'epochs':EPOCHS,'data_sha256':sha(data),'reference_sha256':sha(reference),'parent_v9_sha256':sha(parent)},output/'checkpoint.pt')
    ppm='P3\n3 3\n255\n'+'\n'.join(' '.join(map(str,row.flatten())) for row in held_rgb.reshape(3,3,3))+'\n'
    (output/'heldout9.ppm').write_text(ppm)
    report={'hidden_size':h,'training_rows':135,'heldout_rows':9,'epochs':EPOCHS,'seed':42,
        'method':'Fixed single condition;V9 parent, weighted SmoothL1 beta0.05,L1 weights-only1e-6, gain/offset training augmentation.',
        'teacher_reference_sha256':sha(reference),'reference_ppm_sha256':sha(reference),'reference_source':'TEACHER_REQUIRED_ORIGINAL_RESULT_PPM',
        'training_data_sha256':sha(data),'parent_checkpoint_sha256':sha(parent),'model_header_sha256':sha(header),
        'training_mse_rgb8':float(np.mean((train_rgb-y)**2)),
        'training3x3_mse_rgb8':float(np.mean((train_rgb[np.array([r['chart_id']=='train_3x3' for r in rows])]-y[np.array([r['chart_id']=='train_3x3' for r in rows])])**2)),
        'parent_heldout9_mse_same_teacher_reference':baseline,'heldout9_mse_teacher_reference':float(np.mean((held_rgb-hy)**2)),
        'heldout9_rgb':held_rgb.tolist(),'wall_seconds':time.monotonic()-begin,
        'scope':'Archived latest physical9 remain excluded from training135 and all augmentation/weight calculations. Retrospective inspected validation, no new V10 physical capture, not an independent final chart.'}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    write_csv(output/'training_predictions.csv',[{'sample_id':r['sample_id'],'group':r['group'],'chart_id':r['chart_id'],
        **{f'predicted_{c}8':int(train_rgb[i,j]) for j,c in enumerate('rgb')},
        **{f'target_{c}8':int(y[i,j]) for j,c in enumerate('rgb')},'role':'TRAINING_REPLAY'} for i,r in enumerate(rows)])
    return report


def main():
    protocol=HERE/'training_protocol.json'
    if protocol.exists():raise FileExistsError('Reference-correction run already started; do not overwrite.')
    rows=read_csv(HERE/'training135.csv');held=read_csv(HERE/'heldout9.csv')
    assert set(r['sample_id'] for r in rows).isdisjoint(r['sample_id'] for r in held)
    write_csv(HERE/'sample_weights.csv',[{'sample_id':r['sample_id'],'group':r['group'],'weight':float(w)} for r,w in zip(rows,row_weights(rows,True))])
    protocol.write_text(json.dumps({'reason':'Teacher mandates original result.ppm; previous CMYK recommendation corrected. Never change reference to lower reported MSE.',
        'reference_sha256':sha(HERE/'reference/result.ppm'),'hidden_sizes':HIDDEN,'training_rows':135,'heldout_rows':9,
        'one_fixed_condition_per_H':True,'epochs':EPOCHS,'seed':42,'parent':'Frozen V9 checkpoint for same H; never fitted latest V9 capture1.',
        'optimizer':'Adam0.0005->0.000005 cosine','loss':'Weighted SmoothL1 beta0.05+weights-onlyL1 coefficient1e-6',
        'weights':'Same positive group-balanced/older-consistency rule as V9, recomputed from corrected training135 labels only.',
        'augmentation':'Same as V9 robust gain/offset condition;50% independent-channel0.75..1.30 gains plus bounded offsets;30% gray/white common0.5..1.15 gain, training only.',
        'target_changes':'All111 training3x3 rows now exact teacher result.ppm.24 original6x4 rows keep their original supplied Notebook targets.',
        'heldout_policy':'Latest9 are never added to final fitting. Evaluate frozen final model once; no epoch/condition selection using their MSE.',
        'measurement_scope':'Archived real sensor inputs, retrospective validation. No new physical V10 or independent final chart evaluation yet.'},indent=2)+'\n')
    reports=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(train,h) for h in HIDDEN]):
            r=future.result();reports.append(r);print('TEACHER_REFERENCE_MODEL',json.dumps(r),flush=True)
    write_csv(HERE/'results.csv',[{k:r[k] for k in ['hidden_size','training_mse_rgb8','training3x3_mse_rgb8','parent_heldout9_mse_same_teacher_reference','heldout9_mse_teacher_reference']} for r in sorted(reports,key=lambda r:r['hidden_size'])])


if __name__=='__main__':main()
