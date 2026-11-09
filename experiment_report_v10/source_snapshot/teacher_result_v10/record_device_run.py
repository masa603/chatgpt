"""Validate and archive the user's V10 device replay, real timing and IDE memory log."""

import csv
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
from prepare import read_csv,write_csv,sha
from threshold_tools import load_model,prune
from export_csr_arduino import dense_layer_references


def csv_section(text,heading,count):
    lines=text.splitlines();i=lines.index(heading)
    result=list(csv.DictReader(io.StringIO('\n'.join(lines[i:i+count+1]))))
    assert len(result)==count
    return result


def frame(text,name):
    matches=re.findall(r'========== '+name+r' START ==========\n(.*?)=========== '+name+r' END ===========',text,re.S)
    assert len(matches)==1,(name,len(matches))
    return matches[0]


def near(actual,expected,tolerance):
    assert np.isfinite(float(actual)) and abs(float(actual)-expected)<=tolerance,(actual,expected,tolerance)


def record():
    run=HERE/'hardware/run_01'
    submitted=(run/'submitted_log.txt').read_text()
    # Backslashes are chat formatting, not actual serial data. Preserve submitted_log.txt.
    text='\n'.join(line.strip().strip('\\').strip() for line in submitted.splitlines())+'\n'
    reference=HERE/'reference/result.ppm';data=HERE/'training135.csv';held=HERE/'heldout9.csv'
    physical=ROOT/'calibration_retraining/physical_readings/capture_01/submitted_log.txt'
    target=np.array(reference.read_text().split()[4:],int).reshape(9,3)
    inputs=np.array([[r[f'input_{c}'] for c in 'rgb'] for r in read_csv(held)],np.float32)
    assert sha(data) in text and sha(reference) in text and sha(held) in text and sha(physical) in text
    assert 'ALL MODELS CHECK: PASS' in text and 'ALL EIGHT MODELS CHECK: PASS' in text
    dense_checks=re.findall(r'H=(\d+)\nMODEL_SOURCE_SHA256=([^\n]+)\nMODEL OUTPUT CHECK: PASS\nMAX ABS ERROR=([^\n]+)',text)
    csr_checks=re.findall(r'CSR H=(\d+) INDEX_AND_OUTPUT_CHECK=PASS MAX_ABS_ERROR=([^\n]+)',text)
    assert [int(r[0]) for r in dense_checks]==[40,60,80,100]
    assert [int(r[0]) for r in csr_checks]==[40,60,80,100]
    assert all(0<=float(r[-1])<1e-5 for r in dense_checks+csr_checks)
    comparison=frame(text,'TEACHER RESULT COMPARISON');bench=frame(text,'FOUR PAIR BENCHMARK')
    assert 'FIRMWARE=TEACHER_RESULT_COMPARE_V10' in comparison and 'FROZEN_TEACHER_RESULT_V10' in bench
    assert 'TEACHER_REQUIRED_ORIGINAL_RESULT_PPM' in comparison and 'NO_NEW_SENSOR_CAPTURE' in comparison
    assert 'HELDOUT_ARCHIVED_REAL9; NO_NEW_SENSOR_CAPTURE' in bench
    assert 'INPUT_COUNT=9; WARMUP_CALLS=10; CALLS_PER_GROUP=100; GROUPS=5' in bench
    assert 'F_CPU_HZ=48000000' in bench and 'SENSOR_SERIAL_MSE_EXCLUDED' in bench
    compiler=re.search(r'^COMPILER=(.+)$',bench,re.M).group(1)
    mses=csv_section(comparison,'mse_dataset,H,format,colors,mse',8)
    goals=csv_section(comparison,'quality_goal,H,format,max_mse,met_on_saved_input',8)
    differences=csv_section(comparison,'H,dense_mse,csr_mse,csr_minus_dense,change_percent_if_dense_nonzero',4)
    images=re.findall(r'H=(\d+)\nFORMAT=(DENSE|CSR)\nMODEL_SOURCE_SHA256=([^\n]+)\nTHRESHOLD=([^\n]+)\n---------- PPM START ----------\n(.*?)----------- PPM END -----------',comparison,re.S)
    sections=re.findall(r'H=(\d+)\n(.*?)(?=\nH=|\Z)',bench,re.S)
    assert len(images)==8 and len(sections)==4
    dense=[];sparse=[];selection=[]
    for index,h in enumerate([40,60,80,100]):
        source=HERE/f'models/h{h}/model_parameters.h'
        assert dense_checks[index][1]==sha(source)
        layers=load_model(source);dense.append(layers)
        selected=json.loads((HERE/f'selection/h{h}/summary.json').read_text())['selected']
        selection.append(selected);sparse.append(prune(layers,selected['threshold']))
    models=dense+sparse;results=[];pixels_all=[];rgb_rows=[];max_checksum_error=0
    for i,(image,model,row,goal) in enumerate(zip(images,models,mses,goals)):
        h,fmt,model_sha,threshold,ppm=image;h=int(h)
        assert h==[40,60,80,100][i%4] and fmt==('DENSE' if i<4 else 'CSR')
        assert model_sha==sha(HERE/f'models/h{h}/model_parameters.h')
        near(threshold,0 if i<4 else selection[i%4]['threshold'],1e-15)
        tokens=ppm.split();assert tokens[:4]==['P3','3','3','255'] and len(tokens)==31
        pixels=np.array(tokens[4:],int).reshape(9,3);assert np.all((pixels>=0)&(pixels<=255))
        predicted=dense_layer_references(model,inputs)[-1]
        expected=np.floor(np.clip(predicted.astype(float),0,1)*255+.5).astype(int)
        assert np.array_equal(pixels,expected)
        saved=np.array((HERE/f'site_outputs/h{h}_{fmt.lower()}.ppm').read_text().split()[4:],int).reshape(9,3)
        assert np.array_equal(pixels,saved)
        mse=float(np.mean((pixels-target)**2));near(row['mse'],mse,5.1e-7)
        assert int(row['H'])==h and row['format']==fmt and int(row['colors'])==9
        assert int(goal['H'])==h and goal['format']==fmt and int(goal['max_mse'])==300
        assert goal['met_on_saved_input']==('YES' if mse<=300 else 'NO')
        output=run/f'h{h}_{fmt.lower()}.ppm'
        output.write_text('P3\n3 3\n255\n'+'\n'.join(' '.join(map(str,r.flatten())) for r in pixels.reshape(3,3,3))+'\n')
        site=subprocess.run(['node',str(ROOT/'learning_followup/reference_audit/run_teacher_calculator.cjs'),str(HERE/'reference/mse_calculator.html'),str(reference),str(output)],check=True,capture_output=True,text=True)
        website=json.loads(site.stdout);assert website['valid_full_image_comparison'];near(website['mse_displayed'],mse,.005001)
        pixels_all.append(pixels)
        results.append({'H':h,'format':fmt,'colors':9,'device_mse':float(row['mse']),'recalculated_mse':mse,
                        'teacher_html_mse':website['mse_displayed'],'goal300':mse<=300,
                        'scope':'DEVICE_REPLAY_OF_ARCHIVED_PHYSICAL_INPUTS; NOT_A_NEW_CAPTURE'})
        for point in range(9):
            rgb_rows.append({'H':h,'format':fmt,'color_id':f'r{point//3+1}c{point%3+1}',
                **{f'predicted_{c}8':int(pixels[point,j]) for j,c in enumerate('rgb')},
                **{f'target_{c}8':int(target[point,j]) for j,c in enumerate('rgb')}})
    timing_rows=[];pair_rows=[]
    for index,(h,section) in enumerate(sections):
        h=int(h);chosen=selection[index];assert h==[40,60,80,100][index] and 'TIMING VALID: YES' in section
        assert sha(HERE/f'models/h{h}/model_parameters.h') in section
        for key,expected in [('DENSE_MODEL_BYTES',chosen['dense_bytes']),('CSR_MODEL_BYTES',chosen['csr_bytes'])]:
            assert int(re.search(r'^'+key+r'=(\d+)$',section,re.M).group(1))==expected
        near(re.search(r'^THRESHOLD=(.+)$',section,re.M).group(1),chosen['threshold'],1e-15)
        groups=csv_section(section,'group,first_model,dense_total_us,csr_total_us,dense_us_per_inference,csr_us_per_inference,dense_checksum,csr_checksum',5)
        for group,row in enumerate(groups):
            assert int(row['group'])==group+1 and row['first_model']==('Dense' if group%2==0 else 'CSR')
            x=inputs[np.arange(group*100,group*100+100)%9]
            for name,model in [('dense',models[index]),('csr',models[index+4])]:
                total=int(row[f'{name}_total_us']);assert total>0
                near(row[f'{name}_us_per_inference'],total/100,5.1e-5)
                out=dense_layer_references(model,x)[-1]
                checksum=float((out[:,0]+out[:,1]+out[:,2]).astype(float).sum())
                delta=abs(float(row[f'{name}_checksum'])-checksum);max_checksum_error=max(max_checksum_error,delta)
                # ARM soft-float and host may differ below the RGB8 resolution.
                near(row[f'{name}_checksum'],checksum,2e-4+1e-5*abs(checksum))
            timing_rows.append({'H':h,**row})
        stats={}
        for name in ['dense','csr']:
            times=np.array([int(row[f'{name}_total_us'])/100 for row in groups])
            mean=float(times.mean());std=float(times.std(ddof=1));stats[name]=(mean,std)
            reported=re.search(r'^'+name.upper()+r' mean_us=([^ ]+) sample_stddev_us=([^\n]+)',section,re.M)
            assert reported;near(reported[1],mean,5.1e-5);near(reported[2],std,5.1e-5)
        time_change=(stats['csr'][0]/stats['dense'][0]-1)*100
        near(re.search(r'^CSR time_change_percent=(.+)$',section,re.M)[1],time_change,5.1e-5)
        warm=np.arange(10)%9;warm_expected=0
        for model in [models[index],models[index+4]]:
            out=dense_layer_references(model,inputs[warm])[-1]
            warm_expected+=float((out[:,0]+out[:,1]+out[:,2]).astype(float).sum())
        near(re.search(r'^WARMUP_CHECKSUM=(.+)$',section,re.M)[1],warm_expected,2e-4)
        dense_mse=results[index]['recalculated_mse'];csr_mse=results[index+4]['recalculated_mse']
        diff=differences[index];assert int(diff['H'])==h
        for key,value in [('dense_mse',dense_mse),('csr_mse',csr_mse),('csr_minus_dense',csr_mse-dense_mse),('change_percent_if_dense_nonzero',(csr_mse/dense_mse-1)*100)]:near(diff[key],value,5.1e-7)
        pair_rows.append({'H':h,'dense_mse':dense_mse,'csr_mse':csr_mse,'csr_mse_minus_dense':csr_mse-dense_mse,
            'csr_mse_change_percent':(csr_mse/dense_mse-1)*100,
            'threshold':chosen['threshold'],'dense_model_bytes':chosen['dense_bytes'],'csr_model_bytes':chosen['csr_bytes'],
            'model_payload_reduction_percent':(1-chosen['csr_bytes']/chosen['dense_bytes'])*100,
            'dense_mean_us':stats['dense'][0],'dense_group_sample_stddev_us':stats['dense'][1],
            'csr_mean_us':stats['csr'][0],'csr_group_sample_stddev_us':stats['csr'][1],
            'csr_time_change_percent':time_change,'device_timing_groups':5,'calls_per_group':100})
    compile_text=(run/'compile_submitted.txt').read_text()
    flash=re.search(r'最大(\d+)バイトのフラッシュメモリのうち、スケッチが(\d+)バイト（(\d+)%）',compile_text)
    ram=re.search(r'最大(\d+)バイトのRAMのうち、グローバル変数が(\d+)バイト（(\d+)%）を使っていて、ローカル変数で(\d+)バイト',compile_text)
    assert flash and ram
    flash_max,flash_used,flash_pct=map(int,flash.groups());ram_max,ram_used,ram_pct,headroom=map(int,ram.groups())
    assert (flash_max,ram_max)==(262144,32768) and ram_max-ram_used==headroom
    assert int(flash_used*100/flash_max)==flash_pct and int(ram_used*100/ram_max)==ram_pct
    memory={'firmware':'TEACHER_RESULT_COMPARE_V10','scope':'WHOLE_8_MODEL_COMPARISON_FIRMWARE; NOT_PER_CONDITION',
        'board':'Arduino UNO R4 WiFi','flash_capacity_bytes':flash_max,'sketch_flash_bytes':flash_used,'ide_flash_percent':flash_pct,
        'ram_capacity_bytes':ram_max,'static_global_ram_bytes':ram_used,'ide_static_ram_percent':ram_pct,
        'remaining_ram_after_static_bytes':headroom,'actual_stack_usage':'NOT_MEASURED',
        'uploader_write_bytes':int(re.search(r'Write (\d+) bytes to flash',compile_text)[1]),
        'uploader_write_bytes_not_sketch_flash':True,'source_sha256':sha(run/'compile_submitted.txt')}
    (run/'normalized_log.txt').write_text(text)
    write_csv(run/'mse.csv',results);write_csv(run/'rgb9.csv',rgb_rows);write_csv(run/'timing_groups.csv',timing_rows)
    write_csv(HERE/'device_results.csv',pair_rows)
    (run/'memory.json').write_text(json.dumps(memory,indent=2)+'\n')
    summary={'received_date_japan':'2026-11-09','firmware':'TEACHER_RESULT_COMPARE_V10',
        'record_type':'USER_REPORTED_ARDUINO_EXECUTION; not a host mock',
        'serial_submission_sha256':sha(run/'submitted_log.txt'),'compile_submission_sha256':sha(run/'compile_submitted.txt'),
        'teacher_reference_sha256':sha(reference),'training_data_sha256':sha(data),'heldout_input_sha256':sha(held),
        'source_physical_log_sha256':sha(physical),'model_parameters_sha256':sha(HERE/'teacher_compare_v10/teacher_parameters_v10.h'),
        'compiler':compiler,'f_cpu_hz':48000000,'all8_self_checks':'PASS','all8_rgb_match_frozen_PC':'EXACT',
        'all8_mses_match_teacher_original_html':'PASS','all4_timing_statistics_recalculated':'PASS',
        'all4_alternating_orders_and_checksums':'PASS','max_checksum_absolute_difference_from_PC':max_checksum_error,
        'scope':'MSE: actual device replay of9 archived real sensor inputs. Timing: actual user-reported board measurement,100 calls per group x5. Not a new physical capture.',
        'same_inputs_for_all8':True,'latest9_used_for_fitting_or_threshold_choice':False,
        'new_sensor_capture_count':0,'hardware_timing_runs_received':1,'independent_final_chart':'NOT_PERFORMED',
        'ten_physical_repetitions':'NOT_PERFORMED','csr_quality300_failed_H':[80,100],
        'csr_saved9_relative_mse5pct_failed_H':[40,60,80,100],
        'quality_note':'Separate threshold-selection validation passed its rule; saved9 replay does not preserve MSE within5%. Do not claim general accuracy preservation.',
        'mse_results':results,'pairs':pair_rows,'memory':memory}
    (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for i,pixels in enumerate(pixels_all):
        h=[40,60,80,100][i%4];fmt='dense' if i<4 else 'csr'
        Image.fromarray(pixels.astype(np.uint8).reshape(3,3,3)).resize((360,360),Image.Resampling.NEAREST).save(run/f'h{h}_{fmt}.png')
    print(json.dumps({'checks':'PASS','new_physical_captures':0,'real_device_timing_runs':1,'max_checksum_error':max_checksum_error,'memory':memory,'pairs':pair_rows},indent=2))


if __name__=='__main__':record()
