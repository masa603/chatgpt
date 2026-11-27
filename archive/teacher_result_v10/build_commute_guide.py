"""Build an offline A5 study guide without changing the experiment report."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import textwrap
from build_explanation_guide import function, mse_text, sha, read_rows

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PACKAGE=(ROOT.parent if (ROOT/'moved_paths.json').is_file() else ROOT)/'experiment_report_v10'
if ROOT.name=='source_snapshot':PACKAGE=ROOT.parent


def normalized(code):return re.sub(r'\s+','',code)


def build(package=PACKAGE):
    package=Path(package).resolve()
    unchanged={n:sha(package/n) for n in ['report.pdf','report.docx','report.md','report.tex','explanation/guide.pdf','data/training135.csv']}
    stats={(int(r['H']),r['format']):r for r in read_rows(package/'data/physical10_mse_summary.csv')}
    reads={(int(r['capture_counter']),int(r['H']),r['format']):r for r in read_rows(package/'data/physical10_mse_all80.csv')}
    device={int(r['H']):r for r in read_rows(package/'data/fixed_input_device_timing_capacity.csv')}
    memory={int(r['H']):r for r in read_rows(package/'data/single_model_memory_comparison.csv')}
    assert len(stats)==8 and len(reads)==80
    chosen=json.loads((package/'source_snapshot/teacher_result_v10/selection/h60/summary.json').read_text())['selected']
    assert chosen['nonzero']==1141 and chosen['dense_bytes']==16332 and chosen['csr_bytes']==6449
    source_paths=[package/'code/teacher_compare_v10/teacher_compare_v10.ino',package/'code/teacher_fresh_v10/teacher_fresh_v10.ino',package/'source_snapshot/teacher_result_v10/train.py']
    source,scanner,trainer=[p.read_text() for p in source_paths]
    a=source.index('    float value=m.b1[r];')
    dense=textwrap.dedent(source[a:source.index('\n  }',a)])
    normalize='''float range =
  whiteValue - blackValue;
float result =
  (value - blackValue)
  / range;'''
    smooth='''loss_rows = F.smooth_l1_loss(
    prediction, ty,
    beta=.05, reduction='none'
).mean(1)'''
    train='\n'.join(line.strip() for line in trainer.splitlines() if line.strip().startswith(('optimizer.zero_grad();','loss_rows=','loss=','assert torch.isfinite(loss);')))
    assert len(train.splitlines())==4
    assert normalized(smooth)==normalized(next(line for line in trainer.splitlines() if line.strip().startswith('loss_rows=')))
    for statement in normalize.split(';'):
        if statement.strip():assert normalized(statement+';') in normalized(function(scanner,'float normalizeInput('))
    csr='''for (int r=0; r<layer.rows; r++) {
  float value=layer.bias[r];
  for (uint16_t k=layer.row[r];
       k<layer.row[r+1]; k++)
    value += layer.values[k]
           * input[layer.col[k]];
  output[r]=(relu && value<0)
          ? 0 : value;
}'''
    timer='''float output[3]; checksum=0;
uint32_t begin=static_cast<uint32_t>(micros());
for (int call=0; call<100; call++) {
  int input=(group*100+call)
    % comparison_models::BENCHMARK_INPUT_COUNT;
  comparison_models::predict(
    model,
    comparison_models::BENCHMARK_INPUTS[input],
    output);
  checksum+=output[0]+output[1]+output[2];
}
return static_cast<uint32_t>(micros())-begin;'''
    for snippet,signature in [(csr,'void sparseLayer('),(timer,'uint32_t timeModel(')]:
        actual=function(source,signature).split('{',1)[1].rsplit('}',1)[0]
        assert normalized(snippet)==normalized(actual)
    mse_code='''int e=static_cast<int>(
  pixels[model][point][c])
  -trained_models::REFERENCE_RGB[point][c];
mse[model]+=static_cast<double>(e)*e/27.0;'''
    for statement in mse_code.split(';'):
        if statement.strip():assert normalized(statement+';') in normalized(source)
    d=device[60];m=memory[60]
    substitutions={
        'MEAN_TABLE':'\n'.join(f"| {h} | {mse_text(stats[h,'DENSE']['mean_mse'])} | {mse_text(stats[h,'CSR']['mean_mse'])} |" for h in [40,60,80,100]),
        'H60_MEAN':mse_text(stats[60,'DENSE']['mean_mse']),
        'H60_LAST':mse_text(reads[10,60,'DENSE']['mse_from_received_ppm']),
        'H60_SECOND':mse_text(reads[2,60,'DENSE']['mse_from_received_ppm']),
        'H60_BYTES_D':f"{int(d['dense_model_bytes']):,}",'H60_BYTES_C':f"{int(d['csr_model_bytes']):,}",
        'H60_BYTES_REDUCTION':f"{float(d['model_payload_reduction_percent']):.2f}",
        'H60_TIME_D':f"{float(d['dense_mean_us']):.3f}",'H60_TIME_C':f"{float(d['csr_mean_us']):.3f}",
        'H60_TIME_REDUCTION':f"{-float(d['csr_time_change_percent']):.2f}",
        'H60_FLASH_D':f"{int(m['dense_flash_bytes']):,}",'H60_FLASH_C':f"{int(m['csr_flash_bytes']):,}",
        'H60_FLASH_REDUCTION':f"{float(m['flash_reduction_percent']):.2f}",'H60_RAM':f"{int(m['dense_static_ram_bytes']):,}",
        'TAU60':f"{chosen['threshold']:.8f}",'NORMALIZE_CODE':normalize,'DENSE_CODE':dense,'SMOOTH_CODE':smooth,
        'TRAIN_CODE':train,'CSR_BODY':csr,'MSE_CODE':mse_code,'TIMER_BODY':timer}
    body=(HERE/'report_sections/commute_guide.md').read_text()
    for key,value in substitutions.items():body=body.replace('@@'+key+'@@',value)
    assert '@@' not in body and len(re.findall(r'^# \d+　',body,re.M))==22
    assert '、' not in body and '。' not in body
    out=package/'explanation';out.mkdir(exist_ok=True)
    (out/'commute_guide.md').write_text(body)
    header=r'''\usepackage{setspace,needspace,etoolbox}
\setstretch{1.12}
\XeTeXlinebreaklocale "ja"
\XeTeXlinebreakskip=0pt plus 1pt
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.24em}
\setlength{\footskip}{18pt}
\renewcommand{\arraystretch}{1.08}
\pretocmd{\section}{\Needspace{5\baselineskip}}{}{}
\usepackage{fvextra}
\newcommand{\commutecodefont}{\fontsize{10}{12}\selectfont}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,fontsize=\commutecodefont,commandchars=\\\{\}}
\renewcommand{\CommentTok}[1]{\textcolor[rgb]{0.25,0.25,0.25}{#1}}
\makeatletter
\renewcommand{\maketitle}{\begin{center}\vspace*{-5mm}{\LARGE\bfseries\@title\par}\vspace{3mm}{\normalsize\@date\par}\end{center}}
\makeatother
'''
    (out/'commute_header.tex').write_text(header)
    common=['pandoc','commute_guide.md','--from=markdown-implicit_figures','--standalone','--resource-path=.',
        '-V','documentclass=extarticle','-V','fontsize=14pt','-V','papersize=a5','-V','geometry:margin=12mm',
        '-V','mainfont=Noto Sans CJK JP','-V','sansfont=Noto Sans CJK JP','-V','monofont=Noto Sans Mono CJK JP',
        '--include-in-header=commute_header.tex']
    env=dict(os.environ,XDG_CACHE_HOME=os.environ.get('XDG_CACHE_HOME','/tmp/commute-guide-cache'))
    for extra in [['--output=commute_guide.tex'],['--pdf-engine=xelatex','--output=commute_guide.pdf']]:
        result=subprocess.run(common+extra,cwd=out,env=env,capture_output=True,text=True)
        if result.returncode:raise RuntimeError(result.stderr)
        if extra[-1].endswith('.pdf'):(out/'commute_build.log').write_text(result.stderr)
    assert unchanged=={name:sha(package/name) for name in unchanged}
    data_paths=[package/'data'/name for name in ['physical10_mse_summary.csv','physical10_mse_all80.csv','fixed_input_device_timing_capacity.csv','single_model_memory_comparison.csv']]
    data_paths.extend([package/'figures/circuit_two_buttons.pdf', package/'figures/circuit_two_buttons.json'])
    record={'created_date_japan':'2026-11-27','role':'OFFLINE_A5_STUDY_GUIDE','format':'A5 portrait;14pt text;10pt code;22 topics',
        'original_files_unchanged':unchanged,'physical_capture_count':10,'MSE_records':80,
        'MSE_display':'TRUNCATE_THIRD_AND_LATER_DECIMALS','actual_code_excerpts_checked_ignoring_whitespace':True,
        'illustrative_examples_labelled':True,'new_measurement':False,'retraining':False,
        'sources':{str(path.relative_to(package)):sha(path) for path in data_paths+source_paths}}
    (out/'commute_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('Built offline A5 study guide;22 topics;8 means, circuit and real code excerpts checked;report/data unchanged.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,default=PACKAGE)
    build(parser.parse_args().package)
