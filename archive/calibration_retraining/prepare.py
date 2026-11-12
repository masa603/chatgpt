"""Prepare V9 data while keeping the latest V8 physical reading as development first."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from patch_retraining.prepare import read_csv,write_csv,sha
from retraining.train_current33 import input_collision_bounds


def main():
    rows=read_csv(ROOT/'patch_retraining/training126.csv')
    actual=ROOT/'patch_retraining/physical_readings/capture_01'
    for r in read_csv(actual/'inputs.csv'):
        rows.append({'sample_id':f'v8_capture01_{r["color_id"]}','group':'v8_capture01','chart_id':'train_3x3',
            'color_id':r['color_id'],**{f'input_{c}':r[f'input_{c}'] for c in 'rgb'},
            **{f'target_{c}8':r[f'target_{c}8'] for c in 'rgb'},
            'source_role':'New V8 read; held out for V9 exploratory development, then included in final fitting; not a new V9 test.'})
    assert len(rows)==135 and len({r['sample_id'] for r in rows})==135
    write_csv(HERE/'training135.csv',rows)
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32)
    y=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in rows],np.float32)
    (HERE/'input_collision_bounds.json').write_text(json.dumps(input_collision_bounds(x,y),indent=2)+'\n')
    provenance={'sources':{'training126_sha256':sha(ROOT/'patch_retraining/training126.csv'),
        'v8_actual_log_sha256':sha(actual/'submitted_log.txt'),'v8_actual_inputs_sha256':sha(actual/'inputs.csv')},
        'training135_sha256':sha(HERE/'training135.csv'),'rows':135,'distinct_reference_colors':33,
        'calibration_groups':len({r['group'] for r in rows}),
        'reference_ppm_sha256':sha(ROOT/'distributed_training/references/reference_image1_cmyk_3x3.ppm'),
        'reference_source':'DISTRIBUTED_NOTEBOOK_CMYK_DISPLAY_RECOVERED',
        'development_split':'All9 latest V8 inputs excluded from parent weights, fine-tuning updates, weights and augmentation widths; training126.',
        'development_limit':'Latest failed read inspected to design and select methods; not an independent final test.',
        'final_role_change':'All9 latest V8 inputs included in final135 fitting after selection. New physical V9 reading pending.',
        'rows_removed':0,'provided_notebook_X_used':False,'inference_and_sensor_pipeline_changed':False}
    (HERE/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Prepared135 actual measured rows,33 colors,14 groups. Latest9 held out for development; all earlier126 retained.')


if __name__=='__main__':main()
