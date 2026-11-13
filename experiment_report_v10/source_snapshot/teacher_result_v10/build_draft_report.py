"""Create an explicitly labelled review excerpt without changing the complete report."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from report_layout import finish_docx, write_equation_record

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACKAGE = (ROOT.parent if (ROOT / 'moved_paths.json').is_file() else ROOT) / 'experiment_report_v10'
if ROOT.name == 'source_snapshot':
    PACKAGE = ROOT.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(package=PACKAGE):
    package = Path(package).resolve()
    source = package / 'report.md'
    complete_hashes = {name: sha(package / name) for name in ['report.md', 'report.pdf', 'report.tex', 'report.docx']}
    text = source.read_text()
    metadata = text[:text.index('\n# 概要')]
    metadata = metadata.replace('アジャイルワーク2　実験報告書', 'アジャイルワーク2　実験報告書草稿（確認用抜粋）')
    methods = text.split('# 目的\n', 1)[1].split('\n# 結果\n', 1)[0]
    results = text.split('# 結果\n', 1)[1].split('\n\\Needspace{10\\baselineskip}', 1)[0]
    bibliography = text.split('# 参考文献\n', 1)[1].split('\n# 付録', 1)[0]
    appendix_code = text.split('## コメント付きコード\n', 1)[1].replace('\n### ', '\n## ')
    draft = metadata + '''
# 本稿の範囲

本稿は完成版から一部を省略した確認用草稿である．実施済みの実験に基づき，目的・理論・実験方法と，新規10回のMSE集計までを掲載する．

# 目的
''' + methods + '\n\n# 結果\n' + results + '''

# 本稿で省略した内容

対応差，モデル配列容量，推論時間，Flash・RAMの結果，考察，結論および各回の実測値は，完成版に掲載している．本稿はその一部を確認するための抜粋である．使用関数の役割を本文で説明し，コメント付きコードを付録に掲載する．

# 参考文献
''' + bibliography + '\n\n# 付録：コメント付きコード\n' + appendix_code
    assert '完成版から一部を省略した確認用草稿' in draft
    assert '表1　全10回のMSE集計' in draft and '表2　' not in draft
    assert '# 考察\n' not in draft and '# 結論\n' not in draft
    assert '、' not in draft and '。' not in draft
    output = package / 'draft'
    output.mkdir(exist_ok=True)
    (output / 'report_draft.md').write_text(draft)
    equation_count=write_equation_record(output/'equations.json',draft)
    common = ['pandoc', 'report_draft.md', '--from=markdown-implicit_figures', '--standalone',
              '--number-sections', '--resource-path=..', '--lua-filter=' + str(HERE / 'caption_layout.lua'),
              '--lua-filter=' + str(HERE / 'report_layout.lua'),
              '-V', 'mainfont=Noto Serif CJK JP', '-V', 'sansfont=Noto Sans CJK JP',
              '-V', 'monofont=Noto Sans Mono CJK JP', '-V', 'fontsize=10pt',
              '-V', 'papersize=a4', '-V', 'geometry:margin=22mm', '--include-in-header=../header.tex']
    env = dict(os.environ, XDG_CACHE_HOME=os.environ.get('XDG_CACHE_HOME', '/tmp/draft-report-cache'))
    reference = ['--reference-doc=../word_reference.docx'] if (package / 'word_reference.docx').is_file() else []
    for extra in [['--output=report_draft.tex'], ['--pdf-engine=xelatex', '--output=report_draft.pdf'],
                  reference + ['--output=report_draft.docx']]:
        result = subprocess.run(common + extra, cwd=output, env=env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        extension = extra[-1].rsplit('.', 1)[-1]
        (output / ('build_' + extension + '.log')).write_text(result.stderr)
    finish_docx(output/'report_draft.docx',equation_count)
    assert complete_hashes == {name: sha(package / name) for name in complete_hashes}
    record = dict(created_date_japan='2026-11-13', role='EXPLICIT_REVIEW_EXCERPT_FROM_COMPLETE_REPORT',
                  included='PURPOSE_THEORY_METHODS_ALL10_MSE_SUMMARY_AND_ANNOTATED_CODE_APPENDIX',
                  omitted='PAIRED_DIFFERENCES_CAPACITY_TIMING_FLASH_RAM_DISCUSSION_CONCLUSION_RAW_DATA_APPENDIX',
                  complete_report_files_unchanged=complete_hashes, new_measurements=False,
                  retraining=False, source_measurements_unchanged=True)
    record['cover_and_numbered_equations']=equation_count
    (output / 'verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    (output / 'README.md').write_text('''# 確認用の途中稿（完成版からの抜粋）

[途中稿PDF](report_draft.pdf) ／ [Word](report_draft.docx) ／ [LaTeX](report_draft.tex)

完成版から一部を省略した確認用草稿です．目的・理論・実験方法と，全10回のMSE集計までを掲載しています．
関数の役割は本文で説明し，行ごとの説明は付録のコメント付きコードにまとめています．
元の完成版は [report.pdf](../report.pdf) です．測定値や実施状況，作成日を変更した記録ではありません．
元データは同じものを使用し，図表キャプションの中央揃え・句読点なし，MSEの小数第3位以降の切り捨ても引き継いでいます．

再作成は `source_snapshot` を作業ディレクトリとして `python teacher_result_v10/build_draft_report.py` で実行できます．
完成版4形式の変更がないことと出典ハッシュは [verification.json](verification.json) に記録しています．
''')
    print('Built explicitly labelled review excerpt PDF/DOCX/TeX; complete report and measurements unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=PACKAGE)
    build(parser.parse_args().package)
