"""Preserve augmented fits and finish all three models on measured rows only."""

import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'retraining'),str(ROOT/'learning_followup')]
from train_current33 import ColorNet,model_arrays,forward_fp32,export_header
from train_multiread import write_csv


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finish(h):
    output=HERE/f'models/h{h}';saved=HERE/f'pre_replay_models/h{h}'
    checkpoint=torch.load(saved/'checkpoint.pt',map_location='cpu',weights_only=True)
    old=json.loads((saved/'summary.json').read_text())
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42)
    model=ColorNet(h).float();model.load_state_dict(checkpoint['state_dict'])
    with (HERE/'training69.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32)
    y=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in rows],np.float32)
    same=(y[:,None,:]==y[None,:,:]).all(2)
    weights=1/same.sum(1).astype(np.float32)
    config=checkpoint['config']
    for i,r in enumerate(rows):
        if r['color_id']=='r3c2':weights[i]*=config['gray_weight']
        elif r['color_id']=='r3c3':weights[i]*=config['white_weight']
    tx,ty,tw=torch.from_numpy(x),torch.from_numpy(y/255),torch.from_numpy(weights)
    opt=torch.optim.Adam(model.parameters(),lr=.0001)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10000,eta_min=.000001)
    for epoch in range(10000):
        opt.zero_grad();prediction=model(tx)
        loss=((((prediction-ty)**2).mean(1)*tw).sum()/tw.sum()+1e-6*sum(p.abs().sum() for name,p in model.named_parameters() if 'weight' in name))
        assert torch.isfinite(loss)
        loss.backward();opt.step();schedule.step()
    model.eval();layers=model_arrays(model)
    pixel=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5)
    mse=float(np.mean((pixel-y)**2))
    header=output/'model_parameters.h'
    export_header(layers,h,sha(HERE/'training69.csv'),42,40000,header)
    header.write_text(header.read_text().replace('re-trained on the current 33 saved learning inputs.','V6: 30000-epoch adaptation plus 10000-epoch measured-only replay; 69 rows.'))
    np.savez(output/'model_arrays.npz',**{f'{kind}_{i}':v for i,(w,b) in enumerate(layers,1) for kind,v in [('weight',w),('bias',b)]})
    checkpoint.update(state_dict=model.state_dict(),epochs=40000,pre_replay_checkpoint_sha256=sha(saved/'checkpoint.pt'),measured_only_replay_epochs=10000)
    torch.save(checkpoint,output/'checkpoint.pt')
    report={**old,'epochs':40000,'training_mse_rgb8':mse,'model_header_sha256':sha(header),
            'pre_replay_checkpoint_sha256':sha(saved/'checkpoint.pt'),
            'pre_replay_training_mse_rgb8':old['training_mse_rgb8'],
            'measured_only_replay_epochs':10000,
            'optimizer':'Phase1: 30000 epochs Adam cosine LR0.0003->0.000003. Phase2: 10000 measured-only epochs Adam cosine LR0.0001->0.000001. Same gray/white weights and weights-only L1=1e-6.',
            'scope':'All69 final fitting. Pre-replay models preserved. Development numbers belong to earlier held-out phase. No new physical V6 result.'}
    report['phase1_wall_seconds']=report.pop('wall_seconds')
    report['phase2_wall_seconds']=None
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    if (HERE/'replay_finish.json').exists():raise FileExistsError('Replay finish already saved; do not overwrite.')
    previous={}
    for h in [40,60,100]:
        source=HERE/f'models/h{h}';destination=HERE/f'pre_replay_models/h{h}'
        assert not destination.exists()
        shutil.copytree(source,destination)
        previous[str(h)]=json.loads((source/'summary.json').read_text())['training_mse_rgb8']
    (HERE/'replay_finish.json').write_text(json.dumps({
        'reason':'Measured-row fit after neutral augmentation remained 477.76 MSE for H40. Add the same measured-only finish phase to all H, preserving every previous checkpoint.',
        'decision_timing':'Added after seeing final fitting metrics. No prospective V6 hardware result used.',
        'epochs':10000,'new_synthetic_rows':0,'noise':0,'augmentation':False,
        'previous_training_mses':previous,
        'development_scores':'Original four-condition development scores retained unchanged; this finish is not an independent development/test result.',
        'new_physical_accuracy':'NOT_YET_MEASURED'},indent=2)+'\n')
    reports=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(finish,h) for h in [40,60,100]]):
            r=future.result();reports.append(r);print('MEASURED_REPLAY_FINISH',json.dumps(r),flush=True)
    write_csv(HERE/'training_summary.csv',[{'hidden_size':r['hidden_size'],'training_rows':69,'training_mse_rgb8':r['training_mse_rgb8'],'new_physical_mse':'PENDING','reference_ppm_sha256':r['reference_ppm_sha256']} for r in sorted(reports,key=lambda r:r['hidden_size'])])


if __name__=='__main__':main()
