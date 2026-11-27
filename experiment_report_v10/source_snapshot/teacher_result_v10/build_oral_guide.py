"""Build an A5 oral-explanation guide from the frozen final report and measurements."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import textwrap

import fitz

from build_explanation_guide import function, mse_text, read_rows, sha, PACKAGE

HERE = Path(__file__).resolve().parent


def code_tokens(code):
    return re.sub(r'\s+', '', re.sub(r'//[^\n]*', '', code))


def build(package=PACKAGE):
    package = Path(package).resolve()
    protected = [package / n for n in ['report.pdf', 'report.md', 'report.tex',
                 'report.docx', 'equations.json', 'reference/circuit_original.png']]
    for name in ['data', 'code', 'raw_logs']:
        protected += [p for p in (package / name).rglob('*') if p.is_file()]
    before = {str(p.relative_to(package)): sha(p) for p in protected}

    stats = {(int(r['H']), r['format']): r for r in read_rows(package / 'data/physical10_mse_summary.csv')}
    reads = {(int(r['capture_counter']), int(r['H']), r['format']): r
             for r in read_rows(package / 'data/physical10_mse_all80.csv')}
    pairs = {int(r['H']): r for r in read_rows(package / 'data/physical10_paired_differences.csv')}
    device = {int(r['H']): r for r in read_rows(package / 'data/fixed_input_device_timing_capacity.csv')}
    memory = {int(r['H']): r for r in read_rows(package / 'data/single_model_memory_comparison.csv')}
    selection = {h: json.loads((package / f'source_snapshot/teacher_result_v10/selection/h{h}/summary.json').read_text())
                 for h in [40, 60, 80, 100]}
    assert len(stats) == 8 and len(reads) == 80
    assert all(int(row['physical_captures']) == 10 for row in stats.values())

    paths = {
        'compare': package / 'code/teacher_compare_v10/teacher_compare_v10.ino',
        'fresh': package / 'code/teacher_fresh_v10/teacher_fresh_v10.ino',
        'train': package / 'source_snapshot/teacher_result_v10/train.py',
        'weights': package / 'source_snapshot/teacher_result_v10/data_weights.py',
    }
    source, scanner, trainer = [paths[n].read_text() for n in ['compare', 'fresh', 'train']]
    a = source.index('    float value=m.b1[r];')
    dense = textwrap.dedent(source[a:source.index('\n  }', a)])
    normalize = '''float range = whiteValue - blackValue;
float result =
  (value - blackValue) / range;'''
    for statement in normalize.split(';'):
        if statement.strip():
            assert code_tokens(statement + ';') in code_tokens(function(scanner, 'float normalizeInput('))

    flow = '''optimizer.zero_grad(); prediction=model(ax)
# この間に次ページの損失を計算する
assert torch.isfinite(loss)
loss.backward(); optimizer.step(); schedule.step()'''
    assert code_tokens(flow.splitlines()[0]) in code_tokens(trainer)
    actual_end = next(line.strip() for line in trainer.splitlines() if line.strip().startswith('assert torch.isfinite(loss);'))
    assert code_tokens(actual_end) == code_tokens('assert torch.isfinite(loss);loss.backward();optimizer.step();schedule.step()')
    loss_rows = '''loss_rows = F.smooth_l1_loss(
    prediction, ty,
    beta=.05, reduction='none'
).mean(1)'''
    actual_loss = next(line.strip() for line in trainer.splitlines() if line.strip().startswith('loss_rows='))
    assert code_tokens(loss_rows) == code_tokens(actual_loss)
    full_loss = next(line.strip() for line in trainer.splitlines() if line.strip().startswith('loss='))

    csr = '''// 出力の各行を順番に計算する
for (int r=0; r<layer.rows; r++) {
  // バイアスから始める
  float value=layer.bias[r];
  // この行の非ゼロ重みだけを使う
  for (uint16_t k=layer.row[r];
       k<layer.row[r+1]; k++)
    value += layer.values[k]
           * input[layer.col[k]];
  // 隠れ層だけReLUを使う
  output[r]=(relu && value<0)
          ? 0 : value;
}'''
    timer = '''float output[3]; checksum=0;
// 開始時刻
uint32_t begin=static_cast<uint32_t>(micros());
for (int call=0; call<100; call++) {
  // 保存済み9入力から選ぶ
  int input=(group*100+call)
    % comparison_models::BENCHMARK_INPUT_COUNT;
  comparison_models::predict(
    model,
    comparison_models::BENCHMARK_INPUTS[input],
    output);
  // 出力を使用する
  checksum+=output[0]+output[1]+output[2];
}
// 終了時刻との差
return static_cast<uint32_t>(micros())-begin;'''
    for snippet, signature in [(csr, 'void sparseLayer('), (timer, 'uint32_t timeModel(')]:
        actual = function(source, signature).split('{', 1)[1].rsplit('}', 1)[0]
        assert code_tokens(snippet) == code_tokens(actual), signature
    mse = '''int e=static_cast<int>(
  pixels[model][point][c])
  -trained_models::REFERENCE_RGB[point][c];
mse[model]+=static_cast<double>(e)*e/27.0;'''
    for statement in mse.split(';'):
        if statement.strip():
            assert code_tokens(statement + ';') in code_tokens(source)

    substitutions = {
        'NORMALIZE_CODE': normalize, 'DENSE_CODE': dense,
        'TRAIN_FLOW_CODE': flow, 'LOSS_ROWS_CODE': loss_rows, 'FULL_LOSS_CODE': full_loss,
        'CSR_BODY': csr, 'TIMER_BODY': timer, 'MSE_CODE': mse,
        'TAU60': f"{selection[60]['selected']['threshold']:.8f}",
        'H60_MEAN': mse_text(stats[60, 'DENSE']['mean_mse']),
        'H60_MEAN_C': mse_text(stats[60, 'CSR']['mean_mse']),
        'H60_SD': mse_text(stats[60, 'DENSE']['sample_stddev_mse']),
        'H60_FIRST': mse_text(reads[1, 60, 'DENSE']['mse_from_received_ppm']),
        'H60_SECOND': mse_text(reads[2, 60, 'DENSE']['mse_from_received_ppm']),
        'H60_MSE_CHANGE': f"{float(pairs[60]['relative_change_of_mean_mse_percent']):.2f}",
        'MEAN_TABLE': '\n'.join(f"| {h} | {mse_text(stats[h, 'DENSE']['mean_mse'])} | {mse_text(stats[h, 'CSR']['mean_mse'])} |" for h in selection),
        'PASS_COUNT_TABLE': '\n'.join(f"| {h} | {stats[h, 'DENSE']['reads_with_mse_le300']}/10 | {stats[h, 'CSR']['reads_with_mse_le300']}/10 |" for h in selection),
        'VALIDATION_TABLE': '\n'.join(f"| {h} | {mse_text(s['dense_validation_mse'])} | {mse_text(s['selected']['validation_mse'])} |" for h, s in selection.items()),
        'CAPACITY_TABLE': '\n'.join(f"| {h} | {int(r['dense_model_bytes']):,} | {int(r['csr_model_bytes']):,} |" for h, r in device.items()),
        'TIME_TABLE': '\n'.join(f"| {h} | {float(r['dense_mean_us']):.3f} | {float(r['csr_mean_us']):.3f} | {-float(r['csr_time_change_percent']):.2f}% |" for h, r in device.items()),
        'FLASH_TABLE': '\n'.join(f"| {h} | {int(r['dense_flash_bytes']):,} | {int(r['csr_flash_bytes']):,} |" for h, r in memory.items()),
    }
    body = (HERE / 'report_sections/oral_guide.md').read_text()
    for key, value in substitutions.items():
        body = body.replace('@@' + key + '@@', value)
    assert '@@' not in body
    assert '。' not in body and '、' not in body
    assert len(re.findall(r'^# \d+　', body, re.M)) == 32
    assert set(range(1, 18)).issubset({int(n) for n in re.findall(r'\((\d+)\)', body)})
    assert json.loads((package / 'equations.json').read_text())['count'] == 17

    output = package / 'explanation'
    output.mkdir(exist_ok=True)
    (output / 'oral_guide.md').write_text(body)
    header = r'''\usepackage{setspace,needspace,etoolbox}
\setstretch{1.08}
\XeTeXlinebreaklocale "ja"
\XeTeXlinebreakskip=0pt plus 1pt
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.22em}
\setlength{\footskip}{18pt}
\renewcommand{\arraystretch}{1.08}
\pretocmd{\section}{\Needspace{5\baselineskip}}{}{}
\usepackage{fvextra}
\newcommand{\oralcodefont}{\fontsize{10}{12}\selectfont}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,fontsize=\oralcodefont,commandchars=\\\{\}}
\renewcommand{\CommentTok}[1]{\textcolor[rgb]{0.25,0.25,0.25}{#1}}
\makeatletter
\renewcommand{\maketitle}{\begin{center}\vspace*{-4mm}{\LARGE\bfseries\@title\par}\vspace{3mm}{\normalsize\@date\par}\end{center}}
\makeatother
'''
    (output / 'oral_header.tex').write_text(header)
    command = ['pandoc', 'oral_guide.md', '--from=markdown-implicit_figures', '--standalone',
               '--resource-path=.', '-V', 'documentclass=extarticle', '-V', 'fontsize=14pt',
               '-V', 'papersize=a5', '-V', 'geometry:margin=12mm',
               '-V', 'mainfont=Noto Sans CJK JP', '-V', 'sansfont=Noto Sans CJK JP',
               '-V', 'monofont=Noto Sans Mono CJK JP', '--include-in-header=oral_header.tex']
    env = dict(os.environ, XDG_CACHE_HOME=os.environ.get('XDG_CACHE_HOME', '/tmp/oral-guide-cache'))
    for options in [['--output=oral_guide.tex'], ['--pdf-engine=xelatex', '--output=oral_guide.pdf']]:
        result = subprocess.run(command + options, cwd=output, env=env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        if options[-1].endswith('.pdf'):
            (output / 'oral_build.log').write_text(result.stderr)

    assert before == {str(p.relative_to(package)): sha(p) for p in protected}
    assert (output / 'oral_guide.pdf').read_bytes().startswith(b'%PDF-')
    with fitz.open(output / 'oral_guide.pdf') as pdf:
        assert len(pdf) == 33, 'Expected cover plus one page per topic.'
        assert [entry[2] for entry in pdf.get_toc()] == list(range(2, 34))
        assert all(len(page.get_text()) > 100 for page in pdf)
    record = {
        'date_japan': '2026-11-27', 'role': 'ORAL_EXPLANATION_OF_COMPLETED_REPORT',
        'format': 'A5 portrait,14pt text,10pt code,32 topics',
        'pages': 33, 'one_topic_per_page_verified': True,
        'source_report_pdf_sha256': before['report.pdf'],
        'source_report_md_sha256': before['report.md'],
        'report_equations_covered': list(range(1, 18)),
        'measured_inputs': 90, 'physical_chart_reads': 10, 'mse_results': 80,
        'all8_mean_MSE_and_counts_from_measurement_CSV': True,
        'illustrative_values_labelled': True,
        'code_excerpts_checked_against_frozen_source': True,
        'training_control_excerpt_has_explicit_loss_omission_comment': True,
        'MSE_display': 'TRUNCATE_THIRD_AND_LATER_DECIMALS',
        'protected_report_measurement_code_files_unchanged': before,
        'source_code_sha256': {str(p.relative_to(package)): sha(p) for p in paths.values()},
        'new_measurement': False, 'retraining': False,
    }
    (output / 'oral_verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(f'Built oral guide:32 topics,17 report equations,80 results;{len(before)} final report/data/code files unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=PACKAGE)
    build(parser.parse_args().package)
