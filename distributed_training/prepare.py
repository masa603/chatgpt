"""Recover distributed Notebook RGB targets and assemble only user-measured inputs."""

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'retraining'))
from train_current33 import input_collision_bounds


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def prepare(notebook,requirements):
    if (HERE/'provenance.json').exists():raise FileExistsError('Prepared dataset exists; do not overwrite.')
    materials=HERE/'materials';materials.mkdir(parents=True,exist_ok=True)
    refs=HERE/'references';refs.mkdir(exist_ok=True)
    nb=Path(notebook);req=Path(requirements)
    (materials/'train_L1_normalization.ipynb').write_bytes(nb.read_bytes())
    (materials/'requirements_provided.txt').write_bytes(req.read_bytes())
    content=json.loads(nb.read_text())
    encoded=[o['data']['image/png'] for c in content['cells'] for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
    assert len(encoded)==2
    prior=json.loads((ROOT/'retraining/data/provenance.json').read_text())
    targets={};reference_rows=[];evidence=[]
    for index,(data,extraction,shape) in enumerate(zip(encoded,prior['reference_extraction'],[(3,3),(6,4)]),1):
        if isinstance(data,list):data=''.join(data)
        display=refs/f'notebook_display_{index}.png';display.write_bytes(base64.b64decode(data))
        rgb=np.asarray(Image.open(display).convert('RGB'))
        colors=[]
        for patch in extraction['sample_centers']:
            x,y=patch['display_pixel'];value=rgb[y,x]
            assert np.all(rgb[y-5:y+6,x-5:x+6]==value)
            colors.append(value.tolist())
        width,height=shape;assert len(colors)==width*height
        chart=f'train_{width}x{height}'
        targets[chart]=np.array(colors,np.int32)
        path=refs/f'reference_image{index}_cmyk_{width}x{height}.ppm'
        path.write_text(f'P3\n{width} {height}\n255\n'+'\n'.join(' '.join(str(v) for pixel in colors[row*width:(row+1)*width] for v in pixel) for row in range(height))+'\n')
        evidence.append({'chart_id':chart,'display_png_sha256':sha(display),'reference_ppm':str(path.relative_to(ROOT)),'reference_ppm_sha256':sha(path),'sample_centers':extraction['sample_centers'],'check':'Every sampled 11x11 patch is uniform.'})
        reference_rows.extend({'chart_id':chart,'color_id':f'r{i//width+1}c{i%width+1}',**{f'target_{c}8':int(pixel[j]) for j,c in enumerate('rgb')}} for i,pixel in enumerate(colors))
    csv_write(refs/'target_rgb33.csv',reference_rows)
    ids=[f'r{r}c{c}' for r in range(1,4) for c in range(1,4)]
    with (ROOT/'data_learning_followup/training69.csv').open(newline='') as f:base=list(csv.DictReader(f))
    rows=[]

    def append(r,chart,group,sample_id,x,source):
        color=r['color_id'];width=3 if chart=='train_3x3' else 6
        index=(int(color[1])-1)*width+int(color[3])-1
        target=targets[chart][index]
        rows.append({'sample_id':sample_id,'group':group,'chart_id':chart,'color_id':color,
                     **{f'input_{c}':format(float(x[j]),'.9g') for j,c in enumerate('rgb')},
                     **{f'target_{c}8':int(target[j]) for j,c in enumerate('rgb')},
                     **{f'previous_target_{c}8':r[f'target_{c}8'] for c in 'rgb'},'source_role':source})

    for r in base:append(r,'train_3x3',r['group'],r['sample_id'],[r[f'input_{c}'] for c in 'rgb'],'Prior measured inputs; targets now distributed CMYK.')
    recent=ROOT/'data_learning_followup/physical_readings/capture_01'
    with (recent/'inputs.csv').open(newline='') as f:latest=list(csv.DictReader(f))
    summary=json.loads((recent/'summary.json').read_text())
    raw=np.array([[r[f'raw_{c}'] for c in 'rgb'] for r in latest],np.float32)
    b,w=[np.array(summary[f'{k}_rgb'],np.float32) for k in ['black','white']]
    values=np.clip((raw-b)/(w-b),0,1)
    assert len(latest)==9 and [r['color_id'] for r in latest]==ids
    for i,r in enumerate(latest):append(r,'train_3x3','v6_capture01',f'v6_capture01_{r["color_id"]}',values[i],'Latest V6 physical reading; development then fitting.')
    with (ROOT/'retraining/data/current33.csv').open(newline='') as f:original=list(csv.DictReader(f))
    for r in original[9:]:
        item={**r,'color_id':f'r{int(r["row"])+1}c{int(r["column"])+1}'}
        append(item,'train_6x4','original_6x4',f'original_6x4_{r["sample_id"]}',[r[f'input_{c}'] for c in 'rgb'],'Actual original 6x4 measurement; not template X.')
    assert len(rows)==102 and len({r['group'] for r in rows})==11
    assert len({(r['chart_id'],r['color_id']) for r in rows})==33
    csv_write(HERE/'training102.csv',rows)
    x=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in rows],np.float32)
    y=np.array([[r[f'target_{c}8'] for c in 'rgb'] for r in rows],float)
    collisions=input_collision_bounds(x,y)
    (HERE/'input_collision_bounds.json').write_text(json.dumps(collisions,indent=2)+'\n')
    source_code='\n'.join(''.join(c.get('source',[])) for c in content['cells'])
    assert 'reference_image1_cmyk_large.png' in source_code and 'reference_image2_cmyk_large.png' in source_code
    (HERE/'provenance.json').write_text(json.dumps({
        'user_selected_reference':'配布Notebookの画像に合わせる（推奨）',
        'requested_hidden_sizes':[40,60,80,100],
        'user_learning_priority':'Data/model changes; no geometry/photo prerequisite.',
        'notebook_source_sha256':sha(nb),'requirements_source_sha256':sha(req),
        'reference_extraction':evidence,'original_reference_png_bytes_verified':False,
        'reference_note':'RGB recovered from uniform interior pixels of the supplied Notebook saved displays; exact raw original PNG files were not uploaded as local files.',
        'template_X_used_as_measured_input':False,
        'actual_input_sources':[{'file':'data_learning_followup/training69.csv','sha256':sha(ROOT/'data_learning_followup/training69.csv')},{'file':str((recent/'submitted_log.txt').relative_to(ROOT)),'sha256':sha(recent/'submitted_log.txt')},{'file':'retraining/data/current33.csv','sha256':sha(ROOT/'retraining/data/current33.csv')}],
        'rows':102,'three_by_three_rows':78,'six_by_four_rows':24,'distinct_reference_colors':33,
        'previous_rgb_reference':'Retained with all previous logs. Previous pure-RGB result.ppm MSEs must not be called CMYK-reference results.',
        'role_change':'Latest V6 nine-color read is development for condition selection then final fitting; not an independent V7 result.',
        'development_group':'v6_capture01','development_training_rows':93,'development_held_out_rows':9,
        'loss_note':'Known conflicting identical inputs retained and documented, not asserted to be perfectly recoverable.',
        'requirements_note':'Provided requirements pins numpy2.2.4, OpenCV4.11.0.86, matplotlib3.10.1 but omits torch despite importing it. It was inspected, not blindly installed.'},indent=2)+'\n')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--notebook',type=Path,required=True);parser.add_argument('--requirements',type=Path,required=True)
    args=parser.parse_args();prepare(args.notebook,args.requirements)
