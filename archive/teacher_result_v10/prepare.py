"""Use the teacher's exact result.ppm as the sole 3x3 training/evaluation reference."""

import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    original=ROOT/'learning_followup/reference_audit/provided/result.ppm'
    reference=HERE/'reference/result.ppm';reference.parent.mkdir(exist_ok=True)
    reference.write_bytes(original.read_bytes())
    assert sha(reference)=='582b783b099db5bd6fae89ee0355a17f427fb4a4fc3f2750a8acb8398e5383e2'
    tokens=reference.read_text().split();assert tokens[:4]==['P3','3','3','255'] and len(tokens)==31
    values=list(map(int,tokens[4:]));targets={f'r{r}c{c}':values[((r-1)*3+c-1)*3:((r-1)*3+c)*3] for r in range(1,4) for c in range(1,4)}
    training=read_csv(ROOT/'calibration_retraining/training135.csv')
    changes=[]
    for row in training:
        old=[row[f'target_{c}8'] for c in 'rgb']
        for i,c in enumerate('rgb'):
            row[f'previous_target_{c}8']=old[i]
            if row['chart_id']=='train_3x3':row[f'target_{c}8']=str(targets[row['color_id']][i])
        row['target_source']='TEACHER_SPECIFIED_RESULT_PPM' if row['chart_id']=='train_3x3' else 'ORIGINAL_PROVIDED_NOTEBOOK_6X4_RETAINED'
        changes.append({'sample_id':row['sample_id'],'chart_id':row['chart_id'],'color_id':row['color_id'],
                        'previous_rgb':' '.join(old),'new_rgb':' '.join(row[f'target_{c}8'] for c in 'rgb'),'target_source':row['target_source']})
    assert len(training)==135 and all(r['group']!='v9_capture01' for r in training)
    held=[]
    for row in read_csv(ROOT/'calibration_retraining/physical_readings/capture_01/inputs.csv'):
        held.append({'sample_id':f'v9_capture01_{row["color_id"]}','group':'v9_capture01','chart_id':'train_3x3',
            'color_id':row['color_id'],**{f'input_{c}':row[f'input_{c}'] for c in 'rgb'},
            **{f'target_{c}8':targets[row['color_id']][i] for i,c in enumerate('rgb')},
            'source_role':'Archived physical V9 read, never fitted by V10; inspected retrospective validation, not a new capture or independent final chart.'})
    assert len(held)==9
    write_csv(HERE/'training135.csv',training);write_csv(HERE/'heldout9.csv',held);write_csv(HERE/'target_changes.csv',changes)
    (HERE/'provenance.json').write_text(json.dumps({'authority':'User clarified teacher mandates result.ppm; overrides earlier CMYK choice.',
        'reference_source':'TEACHER_SPECIFIED_RESULT_PPM','reference_sha256':sha(reference),'original_reference_bytes_unchanged':True,
        'training_source_sha256':sha(ROOT/'calibration_retraining/training135.csv'),'training135_sha256':sha(HERE/'training135.csv'),
        'heldout9_sha256':sha(HERE/'heldout9.csv'),'physical_source_log_sha256':sha(ROOT/'calibration_retraining/physical_readings/capture_01/submitted_log.txt'),
        'training_rows':135,'training3x3_rows':111,'training6x4_rows':24,'heldout_rows':9,
        'input_values_changed':False,'3x3_labels':'Every3x3 target matches exact teacher result.ppm. No CMYK target remains for3x3.',
        '6x4_labels':'Original supplied Notebook values retained; teacher result.ppm contains no6x4 targets. Not claimed independently verified against another teacher PPM.',
        'heldout_scope':'Latest9 inputs stay excluded from parent fitting, target correction updates, loss weights and augmentation widths. Their earlier inspection makes this retrospective validation, not an independent final test.',
        'no_source_rows_removed':True},indent=2)+'\n')
    print('PASS: teacher result.ppm exact bytes;135 measured rows retained with corrected3x3 labels; latest9 remain held out.')


if __name__=='__main__':main()
