"""Verify actual user-reported V10 sensor logs and summarize only received captures."""

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT/'pruning')]
from prepare import sha,write_csv
from record_device_run import csv_section,frame,near
from threshold_tools import load_model,prune
from export_csr_arduino import dense_layer_references


def record(path,received_date,boot_id):
    path=Path(path).resolve();directory=path.parent
    # Preserve the submitted bytes; remove only chat escapes/entities for parsing.
    text='\n'.join(html.unescape(line.strip().strip('\\').strip()).strip() for line in path.read_text().splitlines())+'\n'
    block=frame(text,'TEACHER FRESH CAPTURE');lines=block.splitlines()
    required=['FIRMWARE=TEACHER_FRESH_V10','INPUT_SOURCE=NEW_PHYSICAL_SENSOR_READINGS',
        'COMPARISON=ALL_EIGHT_CONDITIONS_USE_THE_SAME_NEW_INPUTS','CHART_ROLE=LEARNING_3X3_REPEAT; NOT_INDEPENDENT_FINAL_CHART',
        'REFERENCE_SOURCE=TEACHER_REQUIRED_ORIGINAL_RESULT_PPM','TRAINING_ROWS=135; MODEL_GENERATION=FROZEN_V10',
        'SAMPLE_COUNT=10; RGB_FLASH_DELAY_MS=10; LED_DRIVE=ANALOG_WRITE_255_0','# size,3,3']
    assert all(line in lines for line in required)
    reference=HERE/'reference/result.ppm';training=HERE/'training135.csv'
    assert f'REFERENCE_FILE_SHA256={sha(reference)}' in lines and f'TRAINING_DATA_SHA256={sha(training)}' in lines
    capture=int(re.search(r'^CAPTURE=(\d+)$',block,re.M)[1])
    completed=int(re.search(r'^COMPLETED_CAPTURES_THIS_BOOT=(\d+)$',block,re.M)[1])
    assert 1<=completed<=capture
    ids=[f'r{r}c{c}' for r in range(1,4) for c in range(1,4)]
    target=np.array(reference.read_text().split()[4:],int).reshape(9,3)
    raw_rows=csv_section(block,'color_id,sample_completed_ms,raw_r,raw_g,raw_b',9)
    input_rows=csv_section(block,'color_id,input_r,input_g,input_b',9)
    assert [r['color_id'] for r in raw_rows]==ids==[r['color_id'] for r in input_rows]
    raw=np.array([[r[f'raw_{c}'] for c in 'rgb'] for r in raw_rows],np.float32)
    logged=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in input_rows],np.float32)
    black,white=[np.array(next(line for line in lines if line.startswith(f'# {kind}_rgb,')).split(',')[1:],np.float32) for kind in ['black','white']]
    assert black.shape==white.shape==(3,) and all(np.isfinite(v).all() for v in [raw,logged,black,white])
    assert np.all(white>black) and np.all((logged>=0)&(logged<=1))
    inputs=np.clip((raw-black)/(white-black),0,1)
    assert np.allclose(inputs,logged,atol=1e-7,rtol=1e-6)
    times=[int(r['sample_completed_ms']) for r in raw_rows];assert all(b>a for a,b in zip(times,times[1:]))
    metrics=csv_section(block,'mse_dataset,H,format,colors,mse',8)
    goals=csv_section(block,'quality_goal,H,format,max_mse,met_for_this_read',8)
    diffs=csv_section(block,'H,dense_mse,csr_mse,csr_minus_dense,change_percent_if_dense_nonzero',4)
    images=re.findall(r'H=(\d+)\nFORMAT=(DENSE|CSR)\nMODEL_SOURCE_SHA256=([^\n]+)\nTHRESHOLD=([^\n]+)\n---------- PPM START ----------\n(.*?)----------- PPM END -----------',block,re.S)
    assert len(images)==8
    verified=[];patch_errors=[];all_pixels=[];max_rgb_difference=0
    for i,(image,row,goal) in enumerate(zip(images,metrics,goals)):
        h,fmt,model_hash,threshold,ppm=image;h=int(h)
        assert h==[40,60,80,100][i%4] and fmt==('DENSE' if i<4 else 'CSR')
        header=HERE/f'models/h{h}/model_parameters.h';assert model_hash==sha(header)
        chosen=json.loads((HERE/f'selection/h{h}/summary.json').read_text())['selected']
        near(threshold,0 if fmt=='DENSE' else chosen['threshold'],1e-15)
        tokens=ppm.split();assert tokens[:4]==['P3','3','3','255'] and len(tokens)==31
        pixels=np.array(tokens[4:],int).reshape(9,3);assert np.all((pixels>=0)&(pixels<=255))
        mse=float(np.mean((pixels-target)**2))
        assert row['mse_dataset']=='new_learning_chart_read' and int(row['H'])==h and row['format']==fmt and int(row['colors'])==9
        near(row['mse'],mse,5.1e-7)
        assert int(goal['H'])==h and goal['format']==fmt and int(goal['max_mse'])==300
        assert goal['met_for_this_read']==('YES' if mse<=300 else 'NO')
        layers=load_model(header)
        if fmt=='CSR':layers=prune(layers,chosen['threshold'])
        prediction=dense_layer_references(layers,inputs)[-1]
        expected=np.floor(np.clip(prediction.astype(float),0,1)*255+.5).astype(int)
        difference=int(np.abs(expected-pixels).max());assert difference<=1,(h,fmt,difference)
        max_rgb_difference=max(max_rgb_difference,difference)
        destination=directory/f'h{h}_{fmt.lower()}.ppm'
        destination.write_text('P3\n3 3\n255\n'+'\n'.join(' '.join(map(str,r.flatten())) for r in pixels.reshape(3,3,3))+'\n')
        site=subprocess.run(['node',str(ROOT/'learning_followup/reference_audit/run_teacher_calculator.cjs'),str(HERE/'reference/mse_calculator.html'),str(reference),str(destination)],check=True,capture_output=True,text=True)
        website=json.loads(site.stdout);assert website['valid_full_image_comparison'];near(website['mse_displayed'],mse,.005001)
        verified.append({'capture_key':directory.name,'boot_id':boot_id,'capture_counter':capture,'H':h,'format':fmt,'colors':9,
            'mse_device':float(row['mse']),'mse_from_received_ppm':mse,'teacher_html_mse':website['mse_displayed'],
            'goal300_this_read':mse<=300,'pc_max_rgb8_difference':difference,'model_header_sha256':model_hash})
        all_pixels.append(pixels)
        for p in range(9):
            patch_errors.append({'H':h,'format':fmt,'color_id':ids[p],**{f'predicted_{c}8':int(pixels[p,j]) for j,c in enumerate('rgb')},
                **{f'target_{c}8':int(target[p,j]) for j,c in enumerate('rgb')},'patch_mse':float(np.mean((pixels[p]-target[p])**2))})
    for i,row in enumerate(diffs):
        a,b=verified[i]['mse_from_received_ppm'],verified[i+4]['mse_from_received_ppm']
        assert int(row['H'])==verified[i]['H']
        for key,value in [('dense_mse',a),('csr_mse',b),('csr_minus_dense',b-a)]:near(row[key],value,5.1e-7)
        if a>0:near(row['change_percent_if_dense_nonzero'],(b/a-1)*100,5.1e-7)
        else:assert row['change_percent_if_dense_nonzero']=='NA'
    write_csv(directory/'inputs.csv',[{**r,**{k:v for k,v in n.items() if k!='color_id'},
        **{f'target_{c}8':int(target[i,j]) for j,c in enumerate('rgb')}} for i,(r,n) in enumerate(zip(raw_rows,input_rows))])
    write_csv(directory/'mse.csv',verified);write_csv(directory/'patch_errors.csv',patch_errors)
    write_csv(directory/'rgb9.csv',[{'color_id':ids[p],**{f'h{[40,60,80,100][i%4]}_{"dense" if i<4 else "csr"}_rgb':' '.join(map(str,pixels[p])) for i,pixels in enumerate(all_pixels)}} for p in range(9)])
    (directory/'normalized_log.txt').write_text(text)
    for i,pixels in enumerate(all_pixels):
        name=f'h{[40,60,80,100][i%4]}_{"dense" if i<4 else "csr"}'
        Image.fromarray(pixels.astype(np.uint8).reshape(3,3,3)).resize((360,360),Image.Resampling.NEAREST).save(directory/f'{name}.png')
    summary={'received_date_japan':received_date,'firmware':'TEACHER_FRESH_V10','boot_id':boot_id,
        'capture_counter':capture,'completed_captures_this_boot':completed,'physical_captures_in_this_record':1,
        'input_source':'NEW_PHYSICAL_SENSOR_READINGS','role':'Same learning3x3 chart physically read again after V10 freeze; not independent unseen-chart final evaluation.',
        'fitting_or_threshold_selection':'NOT_USED; models and thresholds remain fixed during repeated physical measurement.',
        'submitted_log_sha256':sha(path),'reference_sha256':sha(reference),'training_data_sha256':sha(training),
        'parameter_header_sha256':sha(HERE/'teacher_fresh_v10/teacher_parameters_v10.h'),
        'actual_execution_date_time':'Not supplied; sample_completed_ms is uptime,not wall-clock time.',
        'sample_first_completed_ms':times[0],'sample_last_completed_ms':times[-1],
        'black_rgb':black.tolist(),'white_rgb':white.tolist(),'raw_and_normalized_inputs':'PASS',
        'normalization_max_abs_difference':float(np.max(np.abs(inputs-logged))),
        'all8_model_hashes_thresholds_mse_and_flags':'PASS','all8_original_teacher_HTML_mses':'PASS',
        'max_rgb8_difference_from_frozen_PC':max_rgb_difference,
        'pc_match_status':'EXACT' if max_rgb_difference==0 else 'WITHIN_ONE_RGB8_STEP; received pixels and MSE retained unchanged.',
        'startup_self_check_log':'NOT_INCLUDED_IN_THIS_SUBMISSION','input_fingerprint_sha256':hashlib.sha256(raw.tobytes()+black.tobytes()+white.tobytes()).hexdigest(),
        'results':verified}
    note=directory/'operation_note.json'
    if note.exists():summary['reported_operation_note']=json.loads(note.read_text())
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'PASS: {directory.name}, actual CAPTURE={capture};9RAW/normalized inputs,8 model/threshold hashes,MSE/PPMs,teacher HTML; maximum PC RGB8 difference={max_rgb_difference}.')
    return summary


def summarize():
    base=HERE/'fresh_readings';sources=sorted(base.glob('capture_*/summary.json'))
    summaries=[json.loads(p.read_text()) for p in sources]
    assert summaries
    assert len({(s['boot_id'],s['capture_counter']) for s in summaries})==len(summaries),'Repeated boot/capture ID; check duplicate submission or restart.'
    assert len({s['submitted_log_sha256'] for s in summaries})==len(summaries),'Duplicate submitted log is not another measurement.'
    for boot in {s['boot_id'] for s in summaries}:
        runs=sorted([s for s in summaries if s['boot_id']==boot],key=lambda s:s['capture_counter'])
        for a,b in zip(runs,runs[1:]):
            assert b['sample_first_completed_ms']>a['sample_last_completed_ms'],'Uptime restarted/overlapped; record a separate boot ID.'
            assert b['completed_captures_this_boot']>a['completed_captures_this_boot']
    rows=[r for s in summaries for r in s['results']];n=len(summaries)
    assert len(rows)==8*n
    grouped=[];paired=[]
    for h in [40,60,80,100]:
        arrays={}
        for fmt in ['DENSE','CSR']:
            values=np.array([next(r['mse_from_received_ppm'] for r in s['results'] if r['H']==h and r['format']==fmt) for s in summaries])
            arrays[fmt]=values
            grouped.append({'H':h,'format':fmt,'physical_captures':n,'mean_mse':float(values.mean()),
                'sample_stddev_mse':float(values.std(ddof=1)) if n>=2 else None,
                'min_mse':float(values.min()),'max_mse':float(values.max()),'reads_with_mse_le300':int(np.sum(values<=300)),
                'target10_collected':n>=10,'scope':'ACTUAL_RECEIVED_PHYSICAL_REPETITIONS_ONLY; SAME_LEARNING_CHART'})
        delta=arrays['CSR']-arrays['DENSE'];a=float(arrays['DENSE'].mean());b=float(arrays['CSR'].mean())
        paired.append({'H':h,'physical_captures':n,'dense_mean_mse':a,'csr_mean_mse':b,'mean_paired_difference':float(delta.mean()),
            'sample_stddev_paired_difference':float(delta.std(ddof=1)) if n>=2 else None,
            'relative_change_of_mean_mse_percent':(b/a-1)*100 if a>0 else None})
    write_csv(base/'mse_results.csv',rows);write_csv(base/'mse_summary.csv',grouped);write_csv(base/'paired_comparisons.csv',paired)
    progress={'firmware':'TEACHER_FRESH_V10','target_physical_captures':10,'actual_verified_physical_captures':n,
        'physical_captures_remaining':max(0,10-n),'per_model_received_MSE_count':n,'MSE_records_total':8*n,
        'ten_physical_captures_complete':n>=10,'input_sets_shared_by_all8_models':True,
        'same_chart_repetition':'LEARNING_3X3_REPEAT; NOT_INDEPENDENT_FINAL_CHART',
        'all_received_captures_included':True,'model_or_threshold_updates_after_start':False,
        'sample_SD':'Unavailable for one physical repetition; never substitute0 or treat9 colors/8models/10 internal sensor samples as independent chart repetitions.',
        'captures':[{'capture_key':p.parent.name,'boot_id':s['boot_id'],'counter':s['capture_counter'],'source_sha256':s['submitted_log_sha256']} for p,s in zip(sources,summaries)],
        'excluded_captures':[],'duplicate_input_fingerprints_flagged':len({s['input_fingerprint_sha256'] for s in summaries})!=n}
    (base/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    protocol_path=HERE/'fresh_protocol.json';protocol=json.loads(protocol_path.read_text())
    protocol.update({'physical_captures_received_for_this_protocol':n,'current_status':'TEN_REAL_CAPTURES_RECEIVED_AND_VERIFIED' if n>=10 else f'ACTUAL_PHYSICAL_CAPTURE_COLLECTION_IN_PROGRESS_{n}_OF_10',
        'actual_fresh_firmware_build':'Execution confirmed by received user sensor log; no local Arduino cross-build claimed.',
        'progress_record':'fresh_readings/progress.json','mean_and_sample_SD':'fresh_readings/mse_summary.csv'})
    protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    goal_path=ROOT/'learning_followup/goal_update.json'
    goal=json.loads(goal_path.read_text())
    goal.update({'current_status':'V10_TEN_ACTUAL_PHYSICAL_CAPTURES_VERIFIED' if n>=10 else f'V10_ACTUAL_PHYSICAL_CAPTURE_COLLECTION_{n}_OF_10',
        'real10_new_physical_captures_received':n,'real10_actual_build':'Execution confirmed by received fresh firmware logs.',
        'real10_progress':'../teacher_result_v10/fresh_readings/progress.json',
        'real10_MSE_summary':'../teacher_result_v10/fresh_readings/mse_summary.csv',
        'real10_next_action':'REPORT_ACTUAL10_RESULTS' if n>=10 else f'COLLECT_REMAINING_{10-n}_PHYSICAL_CHART_READS',
        'next_step':'Summarize all actual10 readings and preserve failed conditions.' if n>=10 else 'Continue n,black b,white w,all9 colors each r; save each complete fresh block. Keep frozen models/thresholds and include high-MSE reads.',
        'last_updated_date_japan':max(s['received_date_japan'] for s in summaries)})
    goal_path.write_text(json.dumps(goal,ensure_ascii=False,indent=2)+'\n')
    change_path=ROOT/'experiment_today/protocol_change.json'
    changes=json.loads(change_path.read_text())
    current=changes['v10_real10_physical_capture_protocol']
    current.update({'actual_new_physical_captures_received':n,'progress':'../teacher_result_v10/fresh_readings/progress.json',
        'state':'ACTUAL10_COMPLETE' if n>=10 else 'COLLECTING_REAL_MEASUREMENTS','all_received_completed_captures_included':True,
        'each_received_capture_verified':'RAW normalization,frozen model/threshold hashes,8 receivedPPMs/MSEs,and original teacher HTML.'})
    change_path.write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
    table='\n'.join('| '+str(h)+' | '+' | '.join(f"{next(r['mean_mse'] for r in grouped if r['H']==h and r['format']==fmt):.2f}" for fmt in ['DENSE','CSR'])+' |' for h in [40,60,80,100])
    (base/'README.md').write_text(f'''# V10の新規物理読取：{n}／10回受領

紙の9色を実際に読み取ったログだけを集計しています。各回の入力を40・60・80・100次元のDense／枝刈り＋CSRで共有し、学習済み重みと閾値は変更していません。
正解は先生指定の元のresult.ppmです。独立した未見チャートの最終評価とは区別します。

| H | 現在のDense平均MSE | 現在のCSR平均MSE |
| --- | ---: | ---: |
{table}

現在の実測回数は **{n}回**、各モデルのMSEは **{n}個**です。全モデル合計{8*n}個のMSEを{8*n}回の紙の測定とは数えません。
10回完了は **{'済み' if n>=10 else '未完了'}** です。1回しかないときはMSEの標本標準偏差を求められないため空欄にしています。
10回が揃うまで、現在の平均を10回の最終結果として報告しません。

[各回のMSE](mse_results.csv)、[現在の平均・標本SD・最小／最大](mse_summary.csv)、[対応する圧縮前後の差](paired_comparisons.csv)、[進捗と出典](progress.json) を保存しています。
各captureフォルダに受領ログ、正規化確認、入力9組、8PPM／PNG、先生HTMLによる検算結果を残します。高いMSEの回も含めて集計します。
過去の保存入力再生と速度・容量は [別の実機記録](../hardware/run_01/README.md) に保持し、今回の新規物理読取数へ加算しません。

{'10回の新規物理読取が完了しました。追加のn/r操作は不要です。[報告書・コード・データ一式](../../../experiment_report_v10/README.md)' if n>=10 else '次の回は **n → 黒b → 白w → 赤・緑・青・黄・水色・紫・黒・灰・白を各r** です。[測定手順](../teacher_fresh_v10/README.md)'}
''')
    print(f'Summarized {n}/10 actual received physical captures;8 results per capture; no generated repetitions.')
    return progress


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path);parser.add_argument('--received-date',required=True);parser.add_argument('--boot-id',default='boot_01')
    args=parser.parse_args();record(args.log,args.received_date,args.boot_id);summarize()
