"""Package the actual ten physical reads, frozen code and a factual experiment report."""

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/teacher-report-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
OUT=(ROOT.parent if (ROOT/'moved_paths.json').is_file() else ROOT)/'experiment_report_v10'


def load_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_path(source,destination):
    source=Path(source);destination=Path(destination)
    if source.is_dir():
        shutil.copytree(source,destination,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    else:
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)


def cpp_function(source,signature):
    start=source.index(signature);brace=source.index('{',start);depth=1;end=brace+1
    while depth:
        if source[end]=='{':depth+=1
        elif source[end]=='}':depth-=1
        end+=1
    return source[start:end]


def figures(reads,device,single_memory):
    folder=OUT/'figures';folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':13.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(9,3));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    boxes=[(.02,.30,.24,.38,'同じ9色の入力\nセンサで読んだRGB','#edf2f7'),
           (.36,.61,.31,.29,'通常方式（Dense）\n全接続を計算','#e8f0fa'),
           (.36,.11,.31,.29,'圧縮方式：枝刈り＋CSR\n残った接続だけ計算','#fff0df'),
           (.77,.30,.22,.38,'正解RGBと比べる\nMSE（色の誤差）','#edf4eb')]
    for x0,y0,w,h,label,color in boxes:
        ax.add_patch(FancyBboxPatch((x0,y0),w,h,boxstyle='round,pad=0.01',facecolor=color,edgecolor='#6a7480'))
        ax.text(x0+w/2,y0+h/2,label,ha='center',va='center',fontsize=11)
    for start,end in [((.27,.49),(.35,.75)),((.27,.49),(.35,.25)),((.68,.75),(.76,.49)),((.68,.25),(.76,.49))]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':1.5,'color':'#4d5865'})
    ax.text(.5,.015,'H＝40・60・80・100で比較／モデルと閾値は測定中に固定',ha='center',va='bottom',fontsize=10)
    fig.tight_layout(pad=.6)
    for ext in ['png','pdf']:fig.savefig(folder/f'comparison_flow.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11.2,7.0),sharex=True,sharey=True)
    for ax,h in zip(axes.flat,[40,60,80,100]):
        for fmt,color in [('DENSE','#3569a5'),('CSR','#d58439')]:
            rows=[r for r in reads if int(r['H'])==h and r['format']==fmt]
            ax.plot([int(r['capture_counter']) for r in rows],[float(r['mse_from_received_ppm']) for r in rows],marker='o',ms=4,label='通常方式（Dense）' if fmt=='DENSE' else '圧縮方式（枝刈り＋CSR）',color=color,lw=1.4)
        ax.axhline(300,ls='--',color='#ac3345',lw=1,label='目標：300')
        ax.set_title(f'次元 H = {h}');ax.set_yscale('log');ax.set_ylim(.8,4500);ax.set_xticks(range(1,11))
        ax.grid(alpha=.2,which='both');ax.set_ylabel('MSE（対数目盛）');ax.set_xlabel('紙の読取回数')
    axes[0,0].legend(fontsize=12.5,loc='upper right')
    fig.suptitle('紙の新規10回読取：全ての回を含む比較',y=.99)
    fig.tight_layout(rect=(0,0,1,.96))
    for ext in ['png','pdf']:fig.savefig(folder/f'physical_mse10.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8));x=np.arange(4);width=.35
    for fmt,color,offset in [('dense','#3569a5',-width/2),('csr','#d58439',width/2)]:
        axes[0].bar(x+offset,[r[f'{fmt}_model_bytes']/1024 for r in device],width,label='通常方式（Dense）' if fmt=='dense' else '圧縮方式（枝刈り＋CSR）',color=color)
        axes[1].bar(x+offset,[r[f'{fmt}_mean_us']/1000 for r in device],width,label='通常方式（Dense）' if fmt=='dense' else '圧縮方式（枝刈り＋CSR）',color=color,
                    yerr=[r[f'{fmt}_group_sample_stddev_us']/1000 for r in device],capsize=3)
    for ax in axes:
        ax.set_xticks(x,['40','60','80','100']);ax.set_xlabel('隠れ層の大きさ H');ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=12.5)
    axes[0].set_ylabel('モデル配列容量（KiB）');axes[0].set_title('重み・バイアス・位置情報を含む容量')
    axes[1].set_ylabel('1色の計算時間（ms）');axes[1].set_title('Arduinoで保存入力を使って計時')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/f'capacity_and_time.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,3.5))
    ax.bar(x-width/2,[r['dense_flash_bytes']/1024 for r in single_memory['pairs']],width,label='通常方式（Dense）',color='#3569a5')
    ax.bar(x+width/2,[r['csr_flash_bytes']/1024 for r in single_memory['pairs']],width,label='圧縮方式（枝刈り＋CSR）',color='#d58439')
    ax.set_xticks(x,['40','60','80','100']);ax.set_xlabel('隠れ層の大きさ H');ax.set_ylabel('プログラム全体のFlash（KiB）')
    ax.set_title('モデルを1個ずつ入れた場合の保存容量');ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2);ax.legend()
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/f'single_model_flash.{ext}',dpi=180)
    plt.close(fig)


def build():
    progress=json.loads((HERE/'fresh_readings/progress.json').read_text())
    assert progress['actual_verified_physical_captures']==10 and progress['ten_physical_captures_complete']
    assert not progress['excluded_captures']
    stats=load_csv(HERE/'fresh_readings/mse_summary.csv');reads=load_csv(HERE/'fresh_readings/mse_results.csv')
    paired=load_csv(HERE/'fresh_readings/paired_comparisons.csv')
    assert len(stats)==8 and len(reads)==80 and all(int(r['physical_captures'])==10 for r in stats)
    run=json.loads((HERE/'hardware/run_01/summary.json').read_text());device=run['pairs']
    fresh_memory_path=HERE/'hardware/teacher_fresh_v10_compile_01.json'
    fresh_memory=json.loads(fresh_memory_path.read_text()) if fresh_memory_path.exists() else None
    single_memory=json.loads((HERE/'hardware/single_model_memory_run_01/summary.json').read_text())
    assert single_memory['received_conditions']==8
    OUT.mkdir(exist_ok=True)
    figures(reads,device,single_memory)
    single_rows='\n'.join(f"| {r['H']} | {r['dense_flash_bytes']:,} | {r['csr_flash_bytes']:,} | {r['flash_saved_bytes']:,} | {r['flash_reduction_percent']:.2f}% | {r['dense_static_ram_bytes']:,} |" for r in single_memory['pairs'])
    memory_components=[]
    summary_rows='\n'.join(f"| {r['H']} | {r['format']} | {float(r['mean_mse']):.2f} | {float(r['sample_stddev_mse']):.2f} | {float(r['min_mse']):.2f} | {float(r['max_mse']):.2f} | {r['reads_with_mse_le300']}/10 |" for r in stats)
    payload_rows=[]
    validation_rows=[]
    for r in device:
        s=json.loads((HERE/f"selection/h{r['H']}/summary.json").read_text());c=s['selected']
        payload_rows.append(f"| {r['H']} | {c['threshold']:.8f} | {c['nonzero']} | {c['sparsity_pct']:.2f}% | {r['dense_model_bytes']:,} | {r['csr_model_bytes']:,} | {r['model_payload_reduction_percent']:.2f}% |")
        validation_rows.append(f"| {r['H']} | {s['dense_validation_mse']:.2f} | {c['validation_mse']:.2f} | {c['mse_increase_pct']:+.2f}% | {c['training9_mse']:.2f} |")
        h=r['H'];k=c['nonzero'];parts=[4*k,k,2*(2*h+6),4*(2*h+3)]
        assert sum(parts)==r['csr_model_bytes'] and 4*(h*h+8*h+3)==r['dense_model_bytes']
        memory_components.append('| '+str(h)+' | '+' | '.join(f'{n:,}' for n in parts+[sum(parts)])+' |')
    validation=load_csv(ROOT/'pruning/validation.csv')
    zeros=[r for r in validation if all(float(r[f'input_{c}'])==0 for c in 'rgb')]
    assert len(zeros)==6 and len({tuple(r[f'target_{c}8'] for c in 'rgb') for r in zeros})==6
    implementation=(HERE/'teacher_compare_v10/teacher_compare_v10.ino').read_text()
    snippets={'CSR_LAYER_SOURCE':cpp_function(implementation,'void sparseLayer('),
              'TIMER_SOURCE':cpp_function(implementation,'uint32_t timeModel('),
              'MEMORY_COMPONENT_ROWS':'\n'.join(memory_components)}
    extra_sections={}
    for key,name in [('IMPLEMENTATION','implementation'),('PLAN_CHANGES','plan_changes'),('HAND_CALCULATION','hand_calculation'),('QUANTITATIVE','quantitative_discussion'),('FUTURE_WORK','future_work')]:
        section=(HERE/f'report_sections/{name}.md').read_text()
        for token,value in snippets.items():section=section.replace('@@'+token+'@@',value)
        extra_sections[key]=section.strip()
    timing_rows='\n'.join(f"| {r['H']} | {r['dense_mean_us']:.3f} ± {r['dense_group_sample_stddev_us']:.4f} | {r['csr_mean_us']:.3f} ± {r['csr_group_sample_stddev_us']:.4f} | {-r['csr_time_change_percent']:.2f}% |" for r in device)
    paired_rows='\n'.join(f"| {r['H']} | {float(r['mean_paired_difference']):.2f} | {float(r['sample_stddev_paired_difference']):.2f} | {float(r['relative_change_of_mean_mse_percent']):+.2f}% |" for r in paired)
    def raw_table(fmt):
        return '\n'.join('| '+str(i)+' | '+' | '.join(f"{float(next(r['mse_from_received_ppm'] for r in reads if int(r['capture_counter'])==i and int(r['H'])==h and r['format']==fmt)):.6f}" for h in [40,60,80,100])+' |' for i in range(1,11))
    training_rows='\n'.join(f"| {r['hidden_size']} | {float(r['training_mse_rgb8']):.2f} | {float(r['training3x3_mse_rgb8']):.2f} |" for r in load_csv(HERE/'results.csv'))
    plan=(ROOT/'experiment_plan.tex').read_text()
    student_id=re.search(r'\\newcommand\{\\StudentID\}\{([^}]+)\}',plan)[1]
    student_name=re.search(r'\\newcommand\{\\StudentName\}\{([^}]+)\}',plan)[1]
    memory_row=(f"| 新規読取版・全8条件 | {fresh_memory['sketch_flash_bytes']:,} | {fresh_memory['static_global_ram_bytes']:,} | {fresh_memory['remaining_ram_after_static_bytes']:,} |" if fresh_memory else '| 新規読取版・全8条件 | IDE表示未受領 | IDE表示未受領 | IDE表示未受領 |')
    report=(HERE/'report_sections/report_body.md').read_text()
    substitutions={'ID':student_id,'NAME':student_name,'SUMMARY':summary_rows,'PAIRED':paired_rows,
        'PAYLOAD':'\n'.join(payload_rows),'VALIDATION':'\n'.join(validation_rows),'TIMING':timing_rows,'MEMORY':memory_row,
        'RAW_DENSE':raw_table('DENSE'),'RAW_CSR':raw_table('CSR'),'TRAINING':training_rows,'SINGLE_MEMORY':single_rows,
        'TIMER_SOURCE':snippets['TIMER_SOURCE'],**extra_sections}
    for key,value in substitutions.items():report=report.replace('@@'+key+'@@',value)
    assert '@@' not in report
    (OUT/'report.md').write_text(report)
    header=r'''\usepackage{setspace}
\setstretch{1.15}
\usepackage{needspace,etoolbox}
\pretocmd{\section}{\Needspace{8\baselineskip}}{}{}
\pretocmd{\subsection}{\Needspace{6\baselineskip}}{}{}
\XeTeXlinebreaklocale "ja"
\XeTeXlinebreakskip=0pt plus 1pt
\setlength{\parindent}{1em}
\setlength{\parskip}{0.35em}
\renewcommand{\arraystretch}{1.15}
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}}
'''
    (OUT/'header.tex').write_text(header)
    # Stable portable copies, while leaving original files untouched.
    for name in ['teacher_fresh_v10','teacher_compare_v10','memory_probe_v10']:copy_path(HERE/name,OUT/'code'/name)
    copy_path(HERE/'single_model_memory_template.csv',OUT/'data/single_model_memory_template.csv')
    for name in ['single_model_memory_results.csv','single_model_memory_comparison.csv']:copy_path(HERE/name,OUT/'data'/name)
    copy_path(HERE/'report_requirements',OUT/'requirements')
    for source,name in [(HERE/'fresh_readings/mse_results.csv','physical10_mse_all80.csv'),
        (HERE/'fresh_readings/mse_summary.csv','physical10_mse_summary.csv'),(HERE/'fresh_readings/paired_comparisons.csv','physical10_paired_differences.csv'),
        (HERE/'training135.csv','training135.csv'),(HERE/'sample_weights.csv','training_sample_weights.csv'),
        (HERE/'heldout9.csv','old_saved_input9.csv'),(ROOT/'pruning/validation.csv','threshold_validation128.csv'),
        (HERE/'device_results.csv','fixed_input_device_timing_capacity.csv')]:copy_path(source,OUT/'data'/name)
    copy_path(HERE/'reference',OUT/'reference')
    for source in sorted((HERE/'fresh_readings').glob('capture_*')):copy_path(source,OUT/'raw_logs'/'physical10'/source.name)
    copy_path(HERE/'hardware',OUT/'raw_logs'/'hardware')
    subprocess.run([sys.executable,str(HERE/'build_color_images.py'),'--package',str(OUT)],check=True)
    sources=['teacher_result_v10','retraining/train_current33.py','retraining/data/current33.csv','pruning/threshold_tools.py',
        'pruning/export_csr_arduino.py','pruning/validation.csv','pruning/validation','hdjsk',
        'calibration_retraining/models','calibration_retraining/training135.csv','calibration_retraining/physical_readings/capture_01',
        'calibration_retraining/color_retrain_v9','calibration_retraining/compare_all_v9','calibration_retraining/comparison_kernels.cpp',
        'calibration_retraining/comparison_benchmark.cpp','experiment_today/build_fresh_scanner.py','experiment_today/verify_fresh_scanner.py','experiment_today/verify_host.py',
        'learning_followup/reference_audit/run_teacher_calculator.cjs','learning_followup/reference_audit/provided/result.ppm',
        'distributed_training/materials/train_L1_normalization.ipynb','distributed_training/materials/requirements_provided.txt',
        'learning_followup/goal_update.json','experiment_today/protocol_change.json','experiment_plan.tex']
    for source in sources:copy_path(ROOT/source,OUT/'source_snapshot'/source)
    (OUT/'requirements.txt').write_text('numpy==2.3.5\ntorch==2.8.0\nPillow==12.3.0\nmatplotlib==3.10.8\n')
    environment={'report_created_date_japan':'2026-11-09','report_updated_date_japan':'2026-11-13','python':sys.version,'platform':platform.platform(),
        'numpy':np.__version__,'matplotlib':matplotlib.__version__,'training_torch_recorded':'2.8.0+cpu',
        'source_base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'snapshot_scope':'Current workspace including actualcapture10 and8 user-reported single-model memory builds; original frozen models preserved.',
        'physical_captures':10,'excluded_captures':0,'new_training_after_physical10':False,
        'quality_result':'CSR meanMSE>300 and increase>5% for allH; capacity and fixed-input time reduced.',
        'actual_new_firmware_memory_received':fresh_memory is not None,'single_model_memory_builds_received':8,
        'single_model_memory_source':'User-pasted Arduino IDE usage summaries;IDE/core versions and build binaries not supplied.'}
    environment['report_requirements']='Week6 supplied PDF pages6-7; requirement mapping in requirements/coverage.md. Lecture slides excluded from bibliography by user request.'
    environment['editorial_update']='Removed glossary and repetitive tutorial explanations; retained experiment conditions, equations, code, results and required discussion; Japanese prose punctuation unified to fullwidth period and comma; measurements and frozen models unchanged.'
    (OUT/'environment_used.json').write_text(json.dumps(environment,indent=2)+'\n')
    common=['pandoc','report.md','--from=markdown-implicit_figures','--standalone','--number-sections','--resource-path=.',
        '-V','mainfont=Noto Serif CJK JP','-V','sansfont=Noto Sans CJK JP','-V','monofont=Noto Sans Mono CJK JP',
        '-V','fontsize=10pt','-V','papersize=a4','-V','geometry:margin=22mm','--include-in-header=header.tex']
    word=['--reference-doc=word_reference.docx'] if (OUT/'word_reference.docx').exists() else []
    for extra in [['--output=report.tex'],['--pdf-engine=xelatex','--output=report.pdf'],word+['--output=report.docx']]:
        result=subprocess.run(common+extra,cwd=OUT,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
        (OUT/('build_'+extra[-1].split('.')[-1]+'.log')).write_text(result.stderr)
    (OUT/'README.md').write_text('''# 実際の10回測定に基づく報告書・コード・データ一式

[リポジトリの案内へ戻る](../README.md)

[報告書PDF](report.pdf)、[編集できるWord版](report.docx)、[Markdown本文](report.md)、[LaTeX本文](report.tex) を用意しました。
内容を理解するための資料は [本人用の解説PDF（8ページ）](explanation/guide.pdf) です。実験の目的、式、コード、MSE、容量、速度を順に説明しています。提出用はreport.pdfです。
学籍番号・氏名は元の計画書にある情報を引き継いでいます。
用語表と重複した解説を削り，実験条件・必要な式とコード・結果・考察を中心に整理しました．本文の句読点は「．」「，」へ統一し，測定値と結論は保持しています．
提出用PDFはLaTeX（XeLaTeX）から生成しています。[第6週資料と本文の対応](requirements/coverage.md) も別ファイルへ記録しました。講義資料は参考文献一覧へ入れていません。

実際の紙の測定10回を全て含めています。第2回は操作ミスに気づいていないとの回答を残し、削除していません。
容量・固定入力での推論速度は改善しました。CSRの10回平均MSE300以下・精度維持は達成していない結果として記載しています。
各条件単独のFlash・静的RAMは8条件を受領・集計済みです。独立した最終チャート、H50、搭載限界と最大スタック使用量は未実施です。
この資料は実施した内容をまとめたもので、当初計画の全項目完了や採点結果を保証するものではありません。

## 必要なファイル

- 紙を読んだコード：[teacher_fresh_v10.ino](code/teacher_fresh_v10/teacher_fresh_v10.ino)、[数値ヘッダ](code/teacher_fresh_v10/teacher_parameters_v10.h)
- 各次元の色画像：[MSE付きの画像一覧](color_images/README.md)。最後の10回目の8条件と、全10回の個別PNG／PPMを掲載。画像ごとのMSEと10回平均を分けています。
- 速度を計ったコード：[teacher_compare_v10.ino](code/teacher_compare_v10/teacher_compare_v10.ino)、[数値ヘッダ](code/teacher_compare_v10/teacher_parameters_v10.h)
- 各モデル単独のFlash・静的RAM：[8条件の結果](data/single_model_memory_results.csv)、[4組の削減率](data/single_model_memory_comparison.csv)、[受領文・検算](raw_logs/hardware/single_model_memory_run_01/README.md)、[測定手順・コード](code/memory_probe_v10/README.md)。記入用template.csvは実験結果とは別の空欄ひな形です。
- 全10回・8条件（MSE80個）：[MSE一覧](data/physical10_mse_all80.csv)、[平均・標本SD](data/physical10_mse_summary.csv)、[同じ回の圧縮前後差](data/physical10_paired_differences.csv)
- 学習データ：[training135.csv](data/training135.csv)、[行重み](data/training_sample_weights.csv)
- 閾値選択の別入力：[threshold_validation128.csv](data/threshold_validation128.csv)
- 速度・容量：[集計CSV](data/fixed_input_device_timing_capacity.csv)
- 正解：[元のresult.ppm](reference/result.ppm)、[元の先生HTML](reference/mse_calculator.html)
- 各回の元ログ・80枚のPPM/PNG：[raw_logs/physical10](raw_logs/physical10)
- 速度の5組とIDE使用量の元ログ：[raw_logs/hardware](raw_logs/hardware)
- 図：[比較の流れ](figures/comparison_flow.png)、[10回のMSE](figures/physical_mse10.png)、[容量・速度](figures/capacity_and_time.png)、[単独モデルのFlash](figures/single_model_flash.png)。同名PDFも同梱。

コードと学習済み数値は別ファイルです。Arduinoでは使う版の2ファイルを同名フォルダに置きます。
元のデータは変更せず複製し、manifest.jsonへ各ファイルのSHA256を保存しています。
このフォルダで `python verify_package.py` を実行すると、全80MSE・10回の平均と標本SD・コードの対応・ファイルハッシュを再確認できます。

## 資料指定の提出日

2026年11月13日：作成中の実験報告書（書けている部分まで）、12月4日：完成原稿、12月11日：最終版。
資料ではTAへ提出して添削を受け、その指摘を反映する工程も含みます。提出後の指摘はこの原稿へ反映します。
提出用はreport.pdf、LaTeXソースはreport.texです。Word版は編集補助として同梱しています。

## 保存ログの再計算

Python3.12、NumPy・PyTorch・Pillow・Matplotlib、Node.js、C++11対応g++を使います。
requirements.txtはクラウドで使った主要Python版です。OS・CPUが違う場合は、凍結済みヘッダと受領した実機PPMを結果の照合基準にします。

source_snapshotを作業ディレクトリとして、次を実行できます。

```sh
python teacher_result_v10/record_fresh_capture.py teacher_result_v10/fresh_readings/capture_10/submitted_log.txt --received-date 2026-11-09
```

既に各回の検算済みsummaryと出力も同梱しています。上のコマンドは最後の回を再検算し、全10回を集計し直します。
数値ヘッダのモデル検査は `python teacher_result_v10/verify.py`、新規読取プログラムのホスト検査は `python teacher_result_v10/verify_fresh.py` です。
後者の模擬入力は実験の測定回数に含めません。

## 学習・枝刈りの使用ソース

source_snapshot内のteacher_result_v10にprepare.py、train.py、data_weights.py、select_compression.py、finalize_compression.py、build_program.py、build_fresh.pyを同梱しています。
学習を開始した親V9のチェックポイントと、最終V10のチェックポイント・NPZ・Denseヘッダ・CSRヘッダ・全候補CSVも含みます。
凍結済み成果物の誤上書きを避けるガードがあるため、そのままtrain.pyや選択処理を再実行する用途には使わず、再学習時は別の作業コピーを用意します。
記録済み結果の検算には再学習は不要です。
''')
    subprocess.run([sys.executable,str(HERE/'build_explanation_guide.py'),'--package',str(OUT)],check=True)
    for path in OUT.rglob('*'):
        if path.is_file():path.chmod(0o644)
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='manifest.json']
    manifest={'scope':'Factual package of10 actual user-reported paper readings,all80 measured model outputs,real saved-input timing and original code/data.',
        'file_count':len(files),'files':[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Built readable report PDF/DOCX/MD/TeX,4 figures,actual10 logs,8 memory build summaries and source snapshot: {len(files)} files.')


if __name__=='__main__':build()
