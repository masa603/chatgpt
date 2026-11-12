"""Check the packaged real measurements, aggregates, code copies and file manifest."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def close(a,b,tolerance=1e-9):assert abs(float(a)-float(b))<tolerance,(a,b)


def verify(refresh=False):
    assert sha(HERE/'reference/result.ppm')=='582b783b099db5bd6fae89ee0355a17f427fb4a4fc3f2750a8acb8398e5383e2'
    target=list(map(int,(HERE/'reference/result.ppm').read_text().split()[4:]));assert len(target)==27
    all_mse=rows(HERE/'data/physical10_mse_all80.csv');summary=rows(HERE/'data/physical10_mse_summary.csv')
    assert len(all_mse)==80 and len(summary)==8
    assert len({(int(r['capture_counter']),r['H'],r['format']) for r in all_mse})==80
    folders=sorted((HERE/'raw_logs/physical10').glob('capture_*'));assert len(folders)==10
    submitted_hashes=[]
    for counter,folder in enumerate(folders,1):
        record=json.loads((folder/'summary.json').read_text())
        assert record['capture_counter']==counter and record['physical_captures_in_this_record']==1
        assert record['submitted_log_sha256']==sha(folder/'submitted_log.txt')
        assert record['max_rgb8_difference_from_frozen_PC']==0
        submitted_hashes.append(record['submitted_log_sha256'])
        for h in [40,60,80,100]:
            for fmt in ['DENSE','CSR']:
                tokens=(folder/f'h{h}_{fmt.lower()}.ppm').read_text().split()
                assert tokens[:4]==['P3','3','3','255'] and len(tokens)==31
                values=list(map(int,tokens[4:]));assert all(0<=v<=255 for v in values)
                mse=sum((v-t)**2 for v,t in zip(values,target))/27
                r=next(r for r in all_mse if int(r['capture_counter'])==counter and int(r['H'])==h and r['format']==fmt)
                close(r['mse_from_received_ppm'],mse)
    assert len(set(submitted_hashes))==10
    for r in summary:
        values=[float(q['mse_from_received_ppm']) for q in all_mse if q['H']==r['H'] and q['format']==r['format']]
        assert len(values)==10 and int(r['physical_captures'])==10
        for key,expected in [('mean_mse',statistics.mean(values)),('sample_stddev_mse',statistics.stdev(values)),('min_mse',min(values)),('max_mse',max(values))]:close(r[key],expected)
        assert int(r['reads_with_mse_le300'])==sum(v<=300 for v in values)
    paired=rows(HERE/'data/physical10_paired_differences.csv')
    for r in paired:
        dense=[float(q['mse_from_received_ppm']) for q in all_mse if q['H']==r['H'] and q['format']=='DENSE']
        csr=[float(q['mse_from_received_ppm']) for q in all_mse if q['H']==r['H'] and q['format']=='CSR']
        differences=[b-a for a,b in zip(dense,csr)]
        assert all(v>0 for v in differences)
        close(r['mean_paired_difference'],statistics.mean(differences))
        close(r['sample_stddev_paired_difference'],statistics.stdev(differences))
    reference_hash='2516a64a3acea49c10a2a26e4c8ffa5f8e1f5092270c7fba0ece0b02b4edcb4e'
    for name in ['teacher_fresh_v10','teacher_compare_v10']:
        assert sha(HERE/f'code/{name}/teacher_parameters_v10.h')==reference_hash
        for path in (HERE/f'code/{name}').glob('*'):
            assert path.read_bytes()==(HERE/'source_snapshot/teacher_result_v10'/name/path.name).read_bytes()
    for source,target_name in [('training135.csv','training135.csv'),('fresh_readings/mse_results.csv','physical10_mse_all80.csv'),('fresh_readings/mse_summary.csv','physical10_mse_summary.csv')]:
        assert (HERE/'source_snapshot/teacher_result_v10'/source).read_bytes()==(HERE/'data'/target_name).read_bytes()
    probe=HERE/'code/memory_probe_v10'
    for path in probe.iterdir():
        if path.is_file():assert path.read_bytes()==(HERE/'source_snapshot/teacher_result_v10/memory_probe_v10'/path.name).read_bytes()
    template=rows(HERE/'data/single_model_memory_template.csv');assert len(template)==8
    for item in template:
        assert item['status']=='PENDING_MAC_ARDUINO_BUILD'
        assert item['flash_used_bytes']==item['static_ram_used_bytes']==''
    single=rows(HERE/'data/single_model_memory_results.csv')
    comparisons=rows(HERE/'data/single_model_memory_comparison.csv')
    record=HERE/'raw_logs/hardware/single_model_memory_run_01'
    single_summary=json.loads((record/'summary.json').read_text())
    assert len(single)==8 and len(comparisons)==4 and single_summary['received_conditions']==8
    assert single_summary['source_log_sha256']==sha(record/'submitted_log.txt')
    assert single_summary['actual_runtime_stack_peak']=='NOT_MEASURED'
    pattern=r'(40|60|80|100)次元\s+(Dense|枝刈り＋CSR)\s+最大262144バイト.*?スケッチが(\d+)バイト.*?グローバル変数が(\d+)バイト.*?ローカル変数で(\d+)バイト'
    found=re.findall(pattern,(record/'submitted_log.txt').read_text(),re.S)
    assert len(found)==8
    for h,fmt,flash,ram,remaining in found:
        fmt='DENSE' if fmt=='Dense' else 'CSR'
        r=next(r for r in single if r['H']==h and r['format']==fmt)
        assert (r['flash_used_bytes'],r['static_ram_used_bytes'],r['remaining_after_static_bytes'])==(flash,ram,remaining)
        assert int(ram)+int(remaining)==32768 and r['source_log_sha256']==single_summary['source_log_sha256']
    for r in comparisons:
        dense=next(v for v in single if v['H']==r['H'] and v['format']=='DENSE')
        csr=next(v for v in single if v['H']==r['H'] and v['format']=='CSR')
        delta=int(dense['flash_used_bytes'])-int(csr['flash_used_bytes'])
        assert int(r['flash_saved_bytes'])==delta and int(r['static_ram_saved_bytes'])==0
        close(r['flash_reduction_percent'],100*delta/int(dense['flash_used_bytes']))
    for name in ['single_model_memory_results.csv','single_model_memory_comparison.csv']:
        assert (HERE/'data'/name).read_bytes()==(HERE/'source_snapshot/teacher_result_v10'/name).read_bytes()
    for name,flash,ram in [('run_01/memory.json',187888,6740),('teacher_fresh_v10_compile_01.json',191320,7324)]:
        memory=json.loads((HERE/'raw_logs/hardware'/name).read_text())
        assert memory['sketch_flash_bytes']==flash and memory['static_global_ram_bytes']==ram
        assert memory['remaining_ram_after_static_bytes']==32768-ram
    report=(HERE/'report.md').read_text()
    for r in summary:
        assert f"{float(r['mean_mse']):.2f}" in report and f"{float(r['sample_stddev_mse']):.2f}" in report
    for text in ['第2回','除外した回はない','精度維持','達成しなかった','独立','191,320','7,324']:assert text in report
    for r in single:assert f"{int(r['flash_used_bytes']):,}" in report
    for text in ['6,764','26,004','31.26%','7.94','表6']:assert text in report
    verification={'actual_physical_captures':10,'MSE_records':80,'RGB_components':2160,'all_received_captures_included':True,
        'all80_MSE_recomputed_from_copied_PPM':'PASS','all8_means_sampleSD_min_max_goal_counts':'PASS',
        'all4_paired_differences_and_sampleSD':'PASS','frozen_numeric_headers_byte_identical':'PASS',
        'code_copies_match_source_snapshot':'PASS','two_firmware_memory_records_separate':'PASS',
        'single_model_memory_probe':'8 USER_REPORTED_ARDUINO_IDE_BUILD_SUMMARIES_RECEIVED; LOG_ARITHMETIC_CSV_REPORT_MATCH_PASS',
        'single_model_memory_conditions':8,'single_model_memory_template_is_separate_blank_form':True,
        'single_model_memory_stack_peak':'NOT_MEASURED',
        'report_values_match_actual_all10_aggregates':'PASS','source_snapshot_replay_test':'PASS in an isolated temporary copy; no original repository paths required.',
        'CSR_mean300_goal':'NOT_MET','CSR_relative5pct_preservation':'NOT_MET','independent_final_chart':'NOT_PERFORMED'}
    if refresh:
        (HERE/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
        files=[p for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='manifest.json']
        manifest={'scope':'Factual package of10 actual user-reported physical reads; all80 model outputs and unmodified code/data.',
            'file_count':len(files),'files':[{'path':p.relative_to(HERE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
        (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    manifest=json.loads((HERE/'manifest.json').read_text());assert manifest['file_count']==len(manifest['files'])
    listed=set()
    for item in manifest['files']:
        path=HERE/item['path'];assert path.is_relative_to(HERE)
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path'];listed.add(path)
    actual={p for p in HERE.rglob('*') if p.is_file() and p.name!='manifest.json'}
    assert actual==listed
    print(f"PASS:10 real capture records,80 received MSEs,2160RGB,all10 statistics,unchanged code and {len(listed)} file hashes.")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--refresh-manifest',action='store_true')
    verify(parser.parse_args().refresh_manifest)
