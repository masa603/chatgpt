"""Explore calibration on saved full readings; exclude reused reference patches from CV scores."""

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'current_input_adaptation'))
from train import fit


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def prepare():
    reference=ROOT/'learning_followup/reference_audit/provided/result.ppm'
    targets=np.array(reference.read_text().split()[4:],int).reshape(9,3)
    with (ROOT/'retraining/data/current33.csv').open(newline='') as f:
        initial=list(csv.DictReader(f))[:9]
    sources=[('original_3x3',ROOT/'retraining/data/current33.csv',initial)]
    for key,p in [('reread_01','experiment_today/fresh_readings/capture_01'),('reread_03','experiment_today/fresh_readings/capture_03'),('reread_latest','experiment_today/version_mismatch/old_firmware_retry_01'),('v3_capture01','learning_followup/physical_readings/capture_01'),('v4_capture01','reference_corrected/physical_readings/capture_01'),('v5_capture01','current_input_adaptation/physical_readings/capture_01')]:
        path=ROOT/p/'inputs.csv'
        with path.open(newline='') as f:rows=list(csv.DictReader(f))
        sources.append((key,path,rows))
    rows=[];evidence=[]
    for group,path,source in sources:
        assert len(source)==9
        raw=np.array([[r[f'raw_{c}'] for c in 'rgb'] for r in source],np.float32)
        old=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in source],np.float32)
        assert np.all(raw[8]>raw[6])
        paired=np.clip((raw-raw[6])/(raw[8]-raw[6]),0,1)
        evidence.append({'group':group,'source':str(path.relative_to(ROOT)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'black_patch_raw':raw[6].tolist(),'white_patch_raw':raw[8].tolist(),'gray_with_original_calibration':old[7].tolist(),'gray_with_same_read_black_white':paired[7].tolist()})
        for method,values in [('original_calibration',old),('same_read_references_proxy',paired)]:
            for i in range(9):
                rows.append({'sample_id':f'{group}_{i+1}','group':group,'method':method,'color_id':f'r{i//3+1}c{i%3+1}',**{f'input_{c}':float(values[i,j]) for j,c in enumerate('rgb')},**{f'target_{c}8':int(targets[i,j]) for j,c in enumerate('rgb')}})
    write_csv(HERE/'exploratory_inputs.csv',rows)
    (HERE/'provenance.json').write_text(json.dumps({'scope':'Exploratory offline input comparison. Reuses black/white pixels as calibration proxies; not a new physical accuracy result.','reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'sources':evidence,'primary_validation_metric':'MSE of seven non-reference colors only; excludes indices 6 black and 8 white in BOTH methods.','held_out_reference_role':'Black/white from the held-out reading are explicitly known calibration inputs, not scored test targets.','prospective_requirement':'A real revised protocol would need separate calibration measurements, with their collection timing tested. This proxy is not evidence of physical MSE <=300.'},indent=2)+'\n')
    return rows


def worker(task):
    rows,method,group=task
    train=[r for r in rows if r['method']==method and r['group']!=group]
    test=[r for r in rows if r['method']==method and r['group']==group and r['color_id'] not in ['r3c1','r3c3']]
    _,layers,mse,seconds,lo,hi=fit(train,60,{'name':'local_noise_limited','noise':.005},10000)
    from train_current33 import forward_fp32
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in test],np.float32)
    y=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in test],float)
    prediction=np.floor(np.clip(forward_fp32(layers,x).astype(float),0,1)*255+.5)
    return {'method':method,'held_out_group':group,'non_reference_colors_scored':7,'hidden_size':60,'epochs':10000,'mse_seven_colors':float(np.mean((prediction-y)**2)),'gray_patch_mse':float(np.mean((prediction[-1]-y[-1])**2)),'seconds':seconds}


def main():
    if (HERE/'results.csv').exists():raise FileExistsError('Saved exploratory results exist; do not overwrite.')
    rows=prepare();groups=list(dict.fromkeys(r['group'] for r in rows));results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        tasks=[(rows,method,g) for method in ['original_calibration','same_read_references_proxy'] for g in groups]
        for future in as_completed([pool.submit(worker,t) for t in tasks]):
            r=future.result();results.append(r);print(json.dumps(r),flush=True)
    write_csv(HERE/'results.csv',sorted(results,key=lambda r:(r['method'],r['held_out_group'])))
    scores=[{'method':method,'mean_mse_seven_colors':float(np.mean([r['mse_seven_colors'] for r in results if r['method']==method])),'max_fold_mse_seven_colors':max(r['mse_seven_colors'] for r in results if r['method']==method),'scope':'OFFLINE EXPLORATORY PROXY; NOT NEW PHYSICAL MSE'} for method in ['original_calibration','same_read_references_proxy']]
    (HERE/'summary.json').write_text(json.dumps(scores,indent=2)+'\n')
    print('SUMMARY',json.dumps(scores),flush=True)


if __name__=='__main__':main()
