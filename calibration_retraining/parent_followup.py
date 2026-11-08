"""Recorded extra parent-initialization comparison after H40 development remained above300."""

import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import shutil

import train as training
from prepare import read_csv,write_csv

HERE=Path(__file__).resolve().parent
CONFIG={'name':'v7_parent_calibration_jitter_older_consistency','huber':True,'calibration_jitter':True,'older_consistency':True,'parent_stage':7}


def main():
    protocol=HERE/'parent_followup.json'
    if protocol.exists():raise FileExistsError('Extra comparison already started; do not overwrite.')
    note={'reason':'Initial three conditions left H40 development MSE879.00. Compare earlier V7 initialization with the same best robust gain/offset method, on all four H. This is extra inspected development, not an independent test.',
          'condition':CONFIG,'hidden_sizes':[40,60,80,100],'epochs':15000,
          'selection':'Replace per-H selected candidate only when all9 development MSE is strictly smaller; preserve earlier final fit and selection. No pixel filtering.'}
    protocol.write_text(json.dumps(note,indent=2)+'\n')
    rows=read_csv(HERE/'training135.csv')
    original=json.loads((HERE/'development_results.json').read_text())
    selected=json.loads((HERE/'selected_configs.json').read_text())
    shutil.copyfile(HERE/'selected_configs.json',HERE/'initial_selected_configs.json')
    shutil.copyfile(HERE/'training_summary.csv',HERE/'initial_training_summary.csv')
    extras=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(training.development,(rows,h,CONFIG)) for h in [40,60,80,100]]):
            result=future.result();extras.append(result);print('EXTRA_DEVELOPMENT',json.dumps(result),flush=True)
    all_results=original+extras
    (HERE/'development_results.json').write_text(json.dumps(all_results,indent=2)+'\n')
    write_csv(HERE/'development_results.csv',[{k:v for k,v in r.items() if k!='development_rgb9'} for r in sorted(all_results,key=lambda r:(r['hidden_size'],r['method']))])
    changed=[]
    for r in extras:
        h=r['hidden_size']
        if r['development_mse_all9']<selected[str(h)]['development_mse_all9']:
            selected[str(h)]=r;changed.append(h)
    (HERE/'selected_configs.json').write_text(json.dumps(selected,indent=2)+'\n')
    training.CONFIGS.append(CONFIG)
    for h in changed:
        archive=HERE/f'initial_final_models/h{h}';archive.parent.mkdir(exist_ok=True)
        assert not archive.exists()
        shutil.move(str(HERE/f'models/h{h}'),str(archive))
        result=training.final((rows,h,selected[str(h)]));print('EXTRA_FINAL',json.dumps(result),flush=True)
    results=[json.loads((HERE/f'models/h{h}/summary.json').read_text()) for h in [40,60,80,100]]
    write_csv(HERE/'training_summary.csv',[{'hidden_size':r['hidden_size'],'training_rows':135,
        'training_mse_unweighted_rgb8':r['training_mse_rgb8'],
        'old_v8_chart_training_replay_mse':r['training_replay_by_group']['v8_capture01'],
        'patch15_training_replay_mse':r['training_replay_by_group']['patch_session01_cal1'],
        'new_physical_v9_mse':'PENDING'} for r in results])
    note['changed_h']=changed;note['results']=[{'H':r['hidden_size'],'development_mse':r['development_mse_all9']} for r in extras]
    protocol.write_text(json.dumps(note,indent=2)+'\n')
    print('FINAL_SELECTED',json.dumps({h:{'method':r['method'],'mse':r['development_mse_all9']} for h,r in selected.items()}),flush=True)


if __name__=='__main__':main()
