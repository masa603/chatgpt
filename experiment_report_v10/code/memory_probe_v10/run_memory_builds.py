"""Build all eight frozen V10 single-model conditions using an installed Arduino CLI.

Python standard library only. Does not install packages, upload firmware, or read sensors.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
EXPECTED_HEADER_SHA256 = '705a272d6ae147b76a716f0d0215432f2dfd346f93926ef451ee427ddad4d684'
CAPACITY = (262144, 32768)
MODEL_BYTES = {40: (7692, 2854), 60: (16332, 6449), 80: (28172, 9229), 100: (43212, 13094)}
FIELDS = ['H', 'format', 'status', 'model_array_bytes', 'flash_used_bytes', 'flash_max_bytes',
          'static_ram_used_bytes', 'ram_max_bytes', 'remaining_after_static_bytes',
          'fqbn', 'cli_version', 'core_version', 'generated_sketch_sha256', 'header_sha256', 'log_file']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def parse_json(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def parse_memory(stdout, stderr=''):
    """Use CLI section totals, with English/Japanese summary fallback; never ELF file size."""
    combined = stdout + '\n' + stderr
    parsed = parse_json(stdout)
    sections = {}
    for item in walk(parsed):
        if isinstance(item, dict) and str(item.get('name', '')).lstrip('.') in ('text', 'data'):
            if 'size' in item and 'max_size' in item:
                name = str(item['name']).lstrip('.')
                value = (int(item['size']), int(item['max_size']))
                if name in sections and sections[name] != value:
                    raise ValueError('CLIの容量情報が複数あり一致しません。元ログを確認してください。')
                sections[name] = value
    if set(sections) == {'text', 'data'}:
        flash, flash_max = sections['text']
        ram, ram_max = sections['data']
    else:
        # CLI JSON sometimes puts the human-readable compiler summary inside a string.
        combined += '\n' + '\n'.join(x for x in walk(parsed) if isinstance(x, str))
        clean = combined.replace(',', '')
        flash_match = re.search(r'Sketch uses\s+(\d+)\s+bytes.*?Maximum is\s+(\d+)\s+bytes', clean, re.I | re.S)
        ram_match = re.search(r'Global variables use\s+(\d+)\s+bytes.*?Maximum is\s+(\d+)\s+bytes', clean, re.I | re.S)
        if flash_match and ram_match:
            flash, flash_max = map(int, flash_match.groups())
            ram, ram_max = map(int, ram_match.groups())
        else:
            flash_match = re.search(r'最大\s*(\d+)\s*バイトのフラッシュメモリ.*?スケッチが\s*(\d+)\s*バイト', clean, re.S)
            ram_match = re.search(r'最大\s*(\d+)\s*バイトのRAM.*?グローバル変数が\s*(\d+)\s*バイト', clean, re.S | re.I)
            if not (flash_match and ram_match):
                raise ValueError('Flash・静的RAMの容量を読み取れませんでした。元ログは保存しました。')
            flash_max, flash = map(int, flash_match.groups())
            ram_max, ram = map(int, ram_match.groups())
    if (flash_max, ram_max) != CAPACITY:
        raise ValueError(f'UNO R4 WiFiの容量と一致しません: Flash={flash_max}, RAM={ram_max}')
    if not (0 < flash <= flash_max and 0 <= ram <= ram_max):
        raise ValueError(f'容量の値が不正です: Flash={flash}, RAM={ram}')
    return dict(flash_used_bytes=flash, flash_max_bytes=flash_max, static_ram_used_bytes=ram,
                ram_max_bytes=ram_max, remaining_after_static_bytes=ram_max - ram)


def find_cli(explicit=None):
    if explicit:
        value = shutil.which(explicit) or explicit
        if Path(value).is_file():
            return str(Path(value).resolve())
        raise ValueError(f'Arduino CLIが見つかりません: {explicit}')
    if shutil.which('arduino-cli'):
        return shutil.which('arduino-cli')
    for apps in (Path('/Applications'), Path.home() / 'Applications'):
        for name in ('Arduino IDE.app', 'Arduino IDE 2.app'):
            resources = apps / name / 'Contents/Resources'
            direct = resources / 'app/lib/backend/resources/arduino-cli'
            if direct.is_file():
                return str(direct)
            if resources.is_dir():
                found = next((p for p in resources.rglob('arduino-cli') if p.is_file() and os.access(p, os.X_OK)), None)
                if found:
                    return str(found)
    raise ValueError('Arduino CLIが見つかりません。Arduino IDEの「検証」で手動測定できます。README.mdを開いてください。')


def selected_sketch(source, h, mode):
    selected, n1 = re.subn(r'(?m)^#define PROBE_H \d+', f'#define PROBE_H {h}', source)
    selected, n2 = re.subn(r'(?m)^#define PROBE_CSR \d+', f'#define PROBE_CSR {mode}', selected)
    if (n1, n2) != (1, 1):
        raise ValueError('スケッチの条件指定行が見つかりません。元ファイルを確認してください。')
    return selected


def execute(base, arguments):
    command = [*base, *arguments]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return {'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', help='arduino-cliのパス（通常は自動検出）')
    parser.add_argument('--config', help='Arduino CLIの設定ファイル（通常はIDEの設定を自動使用）')
    parser.add_argument('--fqbn', default='arduino:renesas_uno:unor4wifi')
    parser.add_argument('--output', type=Path, help='新しく作る結果フォルダ。既存フォルダは上書きしません。')
    args = parser.parse_args(argv)
    try:
        header, sketch = HERE / 'memory_model_v10.h', HERE / 'memory_probe_v10.ino'
        if not header.is_file() or not sketch.is_file():
            raise ValueError('run_memory_builds.pyと同じフォルダにmemory_probe_v10.inoとmemory_model_v10.hを置いてください。')
        if sha(header) != EXPECTED_HEADER_SHA256:
            raise ValueError('数値ヘッダが凍結済みV10と一致しません。配布ファイルをそのまま置いてください。')
        base = [find_cli(args.cli)]
        config = Path(args.config).expanduser() if args.config else Path.home() / '.arduinoIDE/arduino-cli.yaml'
        if args.config and not config.is_file():
            raise ValueError(f'設定ファイルが見つかりません: {config}')
        if config.is_file():
            base += ['--config-file', str(config)]
        version = execute(base, ['version', '--json'])
        cores = execute(base, ['core', 'list', '--json'])
        if version['returncode'] or cores['returncode']:
            raise ValueError('Arduino CLIの起動または既存ボード情報の取得に失敗しました。Arduino IDEで手動検証してください。\n'
                             + version['stderr'] + cores['stderr'])
        core_json = parse_json(cores['stdout'])
        installed = [x for x in walk(core_json) if isinstance(x, dict) and x.get('id') == 'arduino:renesas_uno']
        if not installed:
            raise ValueError('インストール済みのArduino UNO R4ボードパッケージを確認できません。いつも使っているArduino IDEで手動検証してください。')
        core = installed[0]
        core_version = str(core.get('installed_version') or core.get('installed') or core.get('version') or 'UNKNOWN_SEE_ENVIRONMENT_JSON')
        version_json = parse_json(version['stdout']) or {}
        cli_version = str(version_json.get('VersionString') or version_json.get('version') or version['stdout'].strip())
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        output = args.output.expanduser().resolve() if args.output else HERE / 'compile_results' / stamp
        output.mkdir(parents=True, exist_ok=False)
        source = sketch.read_text()
        (output / 'environment.json').write_text(json.dumps({
            'scope': 'Arduino CLI compile only; no upload and no physical sensor readings.',
            'started_utc': stamp, 'fqbn': args.fqbn, 'version': version, 'installed_cores': cores,
            'original_sketch_sha256': sha(sketch), 'header_sha256': sha(header),
            'declared_work_array_bytes': 800, 'stack_peak': 'NOT_MEASURED'}, ensure_ascii=False, indent=2) + '\n')
        rows = []
        with tempfile.TemporaryDirectory(prefix='v10-memory-build-') as name:
            temp = Path(name)
            for h in (40, 60, 80, 100):
                for mode, fmt in ((0, 'DENSE'), (1, 'CSR')):
                    condition = f'h{h}_{fmt.lower()}'
                    work = temp / condition
                    work.mkdir()
                    ino = work / f'{condition}.ino'
                    ino.write_text(selected_sketch(source, h, mode))
                    shutil.copyfile(header, work / header.name)
                    shutil.copyfile(ino, output / f'{condition}.ino.txt')
                    row = dict(H=h, format=fmt, status='PENDING', model_array_bytes=MODEL_BYTES[h][mode],
                               fqbn=args.fqbn, cli_version=cli_version, core_version=core_version,
                               generated_sketch_sha256=sha(ino), header_sha256=sha(header), log_file=f'{condition}.json')
                    print(f'{h}次元 {fmt}: コンパイル中…', flush=True)
                    record = execute(base, ['compile', '--json', '--fqbn', args.fqbn, '--clean',
                                           '--build-path', str(temp / f'build_{condition}'), str(work)])
                    if record['returncode']:
                        row['status'] = 'COMPILE_FAILED'
                    else:
                        try:
                            row.update(parse_memory(record['stdout'], record['stderr']))
                            row['status'] = 'PASS'
                        except (ValueError, TypeError) as error:
                            row['status'] = 'MEMORY_PARSE_FAILED'
                            record['parse_error'] = str(error)
                    record['condition'] = row.copy()
                    (output / row['log_file']).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
                    rows.append(row)
                    # Persist progress after every completed build; failed rows have blank numbers.
                    with (output / 'single_model_memory.csv').open('w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
                        writer.writeheader()
                        writer.writerows(rows)
                    print(f"  {row['status']}: Flash={row.get('flash_used_bytes', '未取得')} / 静的RAM={row.get('static_ram_used_bytes', '未取得')}", flush=True)
        assert sha(sketch) == json.loads((output / 'environment.json').read_text())['original_sketch_sha256']
        passed = sum(row['status'] == 'PASS' for row in rows)
        print(f'記録完了: {passed}/8条件。結果フォルダ: {output}')
        print('single_model_memory.csv と environment.json を送ってください。失敗した条件は同名jsonも送ってください。')
        return 0 if passed == 8 else 1
    except (ValueError, OSError) as error:
        print(f'測定を完了できませんでした: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
