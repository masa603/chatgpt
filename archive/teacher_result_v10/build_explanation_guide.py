"""Build a personal explanation PDF from frozen report data, without editing the report."""

import argparse
import csv
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
import os
from pathlib import Path
import subprocess
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE = (ROOT.parent if (ROOT / 'moved_paths.json').is_file() else ROOT) / 'experiment_report_v10'
if ROOT.name == 'source_snapshot':
    PACKAGE = ROOT.parent


def read_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mse_text(value):
    return format(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_DOWN), '.2f')


def function(source, signature):
    start = source.index(signature); end = source.index('{', start) + 1; depth = 1
    while depth:
        if source[end] == '{':
            depth += 1
        elif source[end] == '}':
            depth -= 1
        end += 1
    return source[start:end]


def build(package=PACKAGE):
    package = Path(package).resolve()
    report_sha = sha(package / 'report.pdf')
    means = {(int(r['H']), r['format']): float(r['mean_mse']) for r in read_rows(package / 'data/physical10_mse_summary.csv')}
    metrics = {(int(r['capture_counter']), int(r['H']), r['format']): float(r['mse_from_received_ppm'])
               for r in read_rows(package / 'data/physical10_mse_all80.csv')}
    memory = {int(r['H']): r for r in read_rows(package / 'data/single_model_memory_comparison.csv')}
    device = {int(r['H']): r for r in read_rows(package / 'data/fixed_input_device_timing_capacity.csv')}
    assert len(means) == 8 and len(metrics) == 80 and len(memory) == len(device) == 4
    chosen = json.loads((package / 'source_snapshot/teacher_result_v10/selection/h60/summary.json').read_text())['selected']
    assert chosen['nonzero'] == 1141 and chosen['dense_bytes'] == 16332 and chosen['csr_bytes'] == 6449
    code = (package / 'code/teacher_compare_v10/teacher_compare_v10.ino').read_text()
    start = code.index('    float value=m.b1[r];'); end = code.index('\n  }', start)
    dense = textwrap.dedent(code[start:end])
    assert dense.split() == code[start:end].split()
    csr = function(code, 'void sparseLayer(')
    timer = function(code, 'uint32_t timeModel(')
    substitution = {
        'DENSE_CODE': dense, 'CSR_CODE': csr, 'TIMER_CODE': timer,
        'TAU60': f"{chosen['threshold']:.4f}",
        'H60_MEAN': mse_text(means[60, 'DENSE']), 'H60_MEAN_C': mse_text(means[60, 'CSR']),
        'H60_LAST': mse_text(metrics[10, 60, 'DENSE']), 'H60_SECOND': mse_text(metrics[2, 60, 'DENSE']),
        'H60_TIME_D': f"{float(device[60]['dense_mean_us']):.3f}",
        'H60_TIME_C': f"{float(device[60]['csr_mean_us']):.3f}",
        'MEAN_TABLE': '\n'.join(f"| {h} | {mse_text(means[h, 'DENSE'])} | {mse_text(means[h, 'CSR'])} |" for h in (40,60,80,100)),
        'FLASH_TABLE': '\n'.join(f"| {h} | {int(memory[h]['dense_flash_bytes']):,} | {int(memory[h]['csr_flash_bytes']):,} | {float(memory[h]['flash_reduction_percent']):.2f}% |" for h in (40,60,80,100)),
        'TIME_TABLE': '\n'.join(f"| {h} | {float(device[h]['dense_mean_us']):.3f} | {float(device[h]['csr_mean_us']):.3f} | {-float(device[h]['csr_time_change_percent']):.2f}% |" for h in (40,60,80,100))}
    body = (HERE / 'report_sections/explanation_guide.md').read_text()
    for key, value in substitution.items():
        body = body.replace('@@' + key + '@@', value)
    assert '@@' not in body
    for value in means.values():
        assert mse_text(value) in body
    assert dense in body and csr in body and timer in body
    output = package / 'explanation'; output.mkdir(exist_ok=True)
    (output / 'guide.md').write_text(body)
    header = r'''\usepackage{setspace,needspace,etoolbox}
\setstretch{1.10}
\XeTeXlinebreaklocale "ja"
\XeTeXlinebreakskip=0pt plus 1pt
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.4em}
\renewcommand{\arraystretch}{1.12}
\pretocmd{\section}{\Needspace{7\baselineskip}}{}{}
\pretocmd{\subsection}{\Needspace{5\baselineskip}}{}{}
\usepackage{fvextra}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,fontsize=\small,commandchars=\\\{\}}
'''
    (output / 'header.tex').write_text(header)
    common = ['pandoc','guide.md','--from=markdown-implicit_figures','--standalone','--resource-path=.',
              '-V','mainfont=Noto Sans CJK JP','-V','sansfont=Noto Sans CJK JP',
              '-V','monofont=Noto Sans Mono CJK JP','-V','fontsize=11pt','-V','papersize=a4',
              '-V','geometry:margin=20mm','--include-in-header=header.tex']
    env = dict(os.environ, XDG_CACHE_HOME=os.environ.get('XDG_CACHE_HOME','/tmp/explanation-guide-cache'))
    for options in [['--output=guide.tex'],['--pdf-engine=xelatex','--output=guide.pdf']]:
        result = subprocess.run(common + options, cwd=output, env=env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        if options[-1].endswith('.pdf'):
            (output / 'build.log').write_text(result.stderr)
    assert sha(package / 'report.pdf') == report_sha
    record = dict(created_date_japan='2026-11-13', role='PERSONAL_EXPLANATION_GUIDE; submission report remains report.pdf',
                  source_report_pdf_sha256=report_sha, source_mean_MSE_csv_sha256=sha(package / 'data/physical10_mse_summary.csv'),
                  all8_reported_mean_MSE_match_data=True, last_read_MSE_separate_from_ten_read_means=True,
                  MSE_display='TWO_DECIMAL_PLACES_TRUNCATE_THIRD_AND_LATER',
                  Arduino_code_excerpts_match_frozen_source=True, dense_excerpt_indentation_normalized=True,
                  illustrative_CSR_matrix_not_measured_weights=True,
                  original_report_unchanged=True, new_measurement=False, retraining=False)
    (output / 'verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    (output / 'README.md').write_text('''# レポートを理解するための解説

[解説PDFを開く](guide.pdf)

本人が式・コード・実験内容を理解するための資料です。提出用の本文は [report.pdf](../report.pdf) にあります。
学習と実測、枝刈りとCSR、各画像のMSEと10回平均、モデル配列とFlash・静的RAMを順番に説明しています。
コードは使用済みプログラムから抜粋し、平均MSEは元の実測CSVから入れています。

説明の原稿は [guide.md](guide.md)、LaTeXソースは [guide.tex](guide.tex)、照合記録は [verification.json](verification.json) です。
再作成はsource_snapshotを作業フォルダとして `python teacher_result_v10/build_explanation_guide.py` で行えます。
pandoc、XeLaTeXとNoto CJKフォントが必要です。測定や再学習は行いません。
''')
    print('Built personal explanation guide PDF/LaTeX;8 mean MSEs and code excerpts checked; submission report unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=PACKAGE)
    build(parser.parse_args().package)
