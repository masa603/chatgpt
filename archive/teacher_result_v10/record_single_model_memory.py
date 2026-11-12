"""Record eight user-reported standalone Arduino build summaries and calculate reductions."""

import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, records):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def parse_summaries(text):
    normalized = re.sub(r'\\[ \t]*(?=\n|$)', '', text)
    headings = list(re.finditer(r'(?m)^(40|60|80|100)次元[ \t]+(Dense|枝刈り[＋+]CSR)[ \t]*$', normalized))
    if len(headings) != 8:
        raise ValueError('Exactly eight H/format headings are required')
    records = []
    seen = set()
    for i, heading in enumerate(headings):
        h = int(heading[1]); fmt = 'DENSE' if heading[2] == 'Dense' else 'CSR'
        if (h, fmt) in seen:
            raise ValueError('Duplicate condition')
        seen.add((h, fmt))
        block = normalized[heading.end():headings[i + 1].start() if i + 1 < len(headings) else len(normalized)]
        flash = re.findall(r'最大(\d+)バイトのフラッシュメモリのうち、スケッチが(\d+)バイト（(\d+)%）', block)
        ram = re.findall(r'最大(\d+)バイトのRAMのうち、グローバル変数が(\d+)バイト（(\d+)%）.*?ローカル変数で(\d+)バイト', block)
        if len(flash) != 1 or len(ram) != 1:
            raise ValueError(f'Missing or duplicate memory summary for H{h} {fmt}')
        flash_max, flash_used, flash_pct = map(int, flash[0])
        ram_max, ram_used, ram_pct, remaining = map(int, ram[0])
        if (flash_max, ram_max) != (262144, 32768):
            raise ValueError('Unexpected board capacity')
        if not (0 < flash_used <= flash_max and 0 <= ram_used <= ram_max):
            raise ValueError('Invalid usage')
        if ram_used + remaining != ram_max:
            raise ValueError('Remaining RAM is inconsistent with static RAM')
        if (flash_pct, ram_pct) != (flash_used * 100 // flash_max, ram_used * 100 // ram_max):
            raise ValueError('IDE percentages are inconsistent with byte totals')
        records.append(dict(H=h, format=fmt, flash_used_bytes=flash_used, flash_max_bytes=flash_max,
                            ide_flash_percent=flash_pct, static_ram_used_bytes=ram_used,
                            ram_max_bytes=ram_max, ide_static_ram_percent=ram_pct,
                            remaining_after_static_bytes=remaining))
    assert seen == {(h, fmt) for h in (40, 60, 80, 100) for fmt in ('DENSE', 'CSR')}
    return normalized, sorted(records, key=lambda r: (r['H'], r['format'] == 'CSR'))


def record(source, received_date):
    date.fromisoformat(received_date)
    source = Path(source).resolve()
    folder = HERE / 'hardware/single_model_memory_run_01'
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / 'submitted_log.txt'
    raw = source.read_bytes()
    if destination.exists() and destination.read_bytes() != raw:
        raise ValueError('Do not overwrite a different submitted log; use a new run record')
    if not destination.exists():
        destination.write_bytes(raw)
    normalized, rows = parse_summaries(raw.decode('utf-8'))
    (folder / 'normalized_log.txt').write_text(normalized)
    conditions = json.loads((HERE / 'memory_probe_v10/conditions.json').read_text())['conditions']
    for row in rows:
        condition = next(c for c in conditions if c['H'] == row['H'] and c['format'] == row['format'])
        row.update(status='RECEIVED_USER_REPORTED_ARDUINO_IDE_BUILD',
                   model_array_bytes=condition['model_array_bytes'], threshold=condition['threshold'],
                   intended_model_source_sha256=condition['model_source_sha256'],
                   source_log_sha256=sha(destination), received_date_japan=received_date)
    pairs = []
    for h in (40, 60, 80, 100):
        dense = next(r for r in rows if r['H'] == h and r['format'] == 'DENSE')
        csr = next(r for r in rows if r['H'] == h and r['format'] == 'CSR')
        saved = dense['flash_used_bytes'] - csr['flash_used_bytes']
        pairs.append(dict(H=h, dense_flash_bytes=dense['flash_used_bytes'], csr_flash_bytes=csr['flash_used_bytes'],
                          flash_saved_bytes=saved, flash_reduction_percent=100 * saved / dense['flash_used_bytes'],
                          dense_static_ram_bytes=dense['static_ram_used_bytes'], csr_static_ram_bytes=csr['static_ram_used_bytes'],
                          static_ram_saved_bytes=dense['static_ram_used_bytes'] - csr['static_ram_used_bytes']))
    write_csv(folder / 'conditions.csv', rows)
    write_csv(folder / 'paired_comparison.csv', pairs)
    write_csv(HERE / 'single_model_memory_results.csv', rows)
    write_csv(HERE / 'single_model_memory_comparison.csv', pairs)
    summary = dict(received_date_japan=received_date, evidence='User-pasted Arduino IDE build summaries with condition labels',
        scope='Whole single-model inference probe firmware per build; includes Arduino core and common wrapper. Not numeric arrays alone.',
        board='Arduino UNO R4 WiFi', received_conditions=8, source_log_sha256=sha(destination),
        intended_sketch_sha256=sha(HERE / 'memory_probe_v10/memory_probe_v10.ino'),
        intended_numeric_header_sha256=sha(HERE / 'memory_probe_v10/memory_model_v10.h'),
        compiled_source_identity='Intended source recorded; no binary or build metadata supplied for independent identity verification.',
        arduino_ide_version='NOT_REPORTED', arduino_core_version='NOT_REPORTED',
        source_log_arithmetic_and_percent_checks='PASS', conditions=rows, pairs=pairs,
        actual_runtime_stack_peak='NOT_MEASURED', new_sensor_captures=0,
        firmware_uploaded_or_runtime_tested_in_this_record='NOT_REPORTED; compile summaries only',
        original_physical10_mse_and_device_timing_unchanged=True)
    (folder / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    table = '\n'.join(f"| {r['H']} | {r['dense_flash_bytes']:,} | {r['csr_flash_bytes']:,} | {r['flash_saved_bytes']:,} | {r['flash_reduction_percent']:.2f}% | {r['dense_static_ram_bytes']:,} | {r['csr_static_ram_bytes']:,} |" for r in pairs)
    (folder / 'README.md').write_text(f'''# 各モデル単独のFlash・静的RAM：8条件の受領結果

{received_date}、日本時間で利用者から受領したArduino IDEの条件名付き2行表示を記録しました。
同じ測定用プログラムに40/60/80/100次元のDense／枝刈り＋CSRを1条件ずつ含めたビルドの使用量です。

| H | Dense Flash byte | CSR Flash byte | Flash削減 byte | Flash削減率 | Dense静的RAM byte | CSR静的RAM byte |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

全8条件で静的RAMは6,764 byte、静的割当後の残りは26,004 byteです。
Flashはプログラム全体で7.94〜31.26%減りました。静的RAMの削減は0 byteです。
数値配列の60.51〜69.70%削減とは対象が異なります。共通処理とArduinoコアを含むため、Flash全体の削減率は小さくなります。
推論の中間配列は関数内のローカル変数で、IDEのグローバル変数使用量には入りません。26,004 byteはスタックの実使用量ではありません。

[受領文](submitted_log.txt)、[表示用バックスラッシュ除去版](normalized_log.txt)、[8条件CSV](conditions.csv)、[4組の比較CSV](paired_comparison.csv)、[出典・検算JSON](summary.json) を保存しました。
公称容量、表示百分率、静的RAM＋残りRAMを検算しました。IDE・ボードコアの版、ビルドバイナリは未受領です。
この記録を新たな紙の読取や速度測定、実行時スタック測定として数えません。
''')
    print(json.dumps({'received_conditions': 8, 'flash_reduction_percent': {r['H']: r['flash_reduction_percent'] for r in pairs},
                      'static_ram_bytes_each': rows[0]['static_ram_used_bytes']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--received-date', required=True)
    args = parser.parse_args()
    record(args.source, args.received_date)
