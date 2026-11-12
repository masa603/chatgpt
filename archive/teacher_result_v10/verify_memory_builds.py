"""Test the memory-log parser and automation using temporary mocks only."""

from contextlib import redirect_stdout, redirect_stderr
import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / 'memory_probe_v10/run_memory_builds.py'
spec = importlib.util.spec_from_file_location('memory_builds', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def verify():
    # All numeric examples in this function are parser fixtures, not measured results.
    expected = dict(flash_used_bytes=20000, flash_max_bytes=262144, static_ram_used_bytes=2000,
                    ram_max_bytes=32768, remaining_after_static_bytes=30768)
    sections = [{'name': 'text', 'size': 20000, 'max_size': 262144},
                {'name': 'data', 'size': 2000, 'max_size': 32768}]
    english = ('Sketch uses 20,000 bytes (7%) of program storage space. Maximum is 262,144 bytes.\n'
               'Global variables use 2,000 bytes (6%) of dynamic memory, leaving 30,768 bytes for local variables. Maximum is 32,768 bytes.')
    japanese = ('最大262144バイトのフラッシュメモリのうち、スケッチが20000バイト（7%）を使っています。\n'
                '最大32768バイトのRAMのうち、グローバル変数が2000バイト（6%）を使っていて、ローカル変数で30768バイト使うことができます。')
    fixtures = [json.dumps({'builder_result': {'executable_sections_size': sections}}),
                json.dumps({'build_result': {'executable_sections_size': sections}}),
                json.dumps({'compiler_out': english}), json.dumps({'compiler_out': japanese}), english, japanese]
    for text in fixtures:
        assert module.parse_memory(text) == expected
    assert module.parse_memory('', english) == expected
    for invalid in ['', 'Compilation error', english.replace('262,144', '32,256'),
                    english.replace('20,000', '300,000'), english.replace('20,000', '0')]:
        try:
            module.parse_memory(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid summary was accepted')
    sketch = (HERE / 'memory_probe_v10/memory_probe_v10.ino').read_text()
    for failed_condition in (None, 'h60_csr'):
        with tempfile.TemporaryDirectory(prefix='mock-only-v10-cli-') as name:
            temp = Path(name)
            for file in ['memory_model_v10.h', 'memory_probe_v10.ino']:
                shutil.copyfile(HERE / 'memory_probe_v10' / file, temp / file)
            calls = []

            def fake_execute(base, arguments):
                calls.append(arguments)
                result = dict(command=base + arguments, returncode=0, stdout='', stderr='')
                if arguments[0] == 'version':
                    result['stdout'] = json.dumps({'VersionString': 'MOCK_ONLY'})
                elif arguments[:2] == ['core', 'list']:
                    result['stdout'] = json.dumps({'platforms': [{'id': 'arduino:renesas_uno', 'installed_version': 'MOCK_ONLY'}]})
                else:
                    assert arguments[0] == 'compile' and '--clean' in arguments
                    work = Path(arguments[-1])
                    h = int(work.name.split('_')[0][1:])
                    mode = int(work.name.endswith('csr'))
                    actual = (work / f'{work.name}.ino').read_text()
                    assert actual == module.selected_sketch(sketch, h, mode)
                    assert module.sha(work / 'memory_model_v10.h') == module.EXPECTED_HEADER_SHA256
                    result['stdout'] = json.dumps({'builder_result': {'executable_sections_size': sections}})
                    if work.name == failed_condition:
                        result.update(returncode=1, stdout='', stderr='MOCK_FAILURE_ONLY')
                return result

            with patch.object(module, 'HERE', temp), patch.object(module, 'execute', fake_execute), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                code = module.main(['--cli', shutil.which('true'), '--output', str(temp / 'test_results')])
                assert code == int(failed_condition is not None)
                # An existing result folder is refused rather than overwritten.
                assert module.main(['--cli', shutil.which('true'), '--output', str(temp / 'test_results')]) == 1
            with (temp / 'test_results/single_model_memory.csv').open(newline='') as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 8 and len([x for x in calls if x[0] == 'compile']) == 8
            assert all(x[0] in ('version', 'core', 'compile') for x in calls)
            assert (temp / 'memory_probe_v10.ino').read_text() == sketch
            for row in rows:
                failed = f"h{row['H']}_{row['format'].lower()}" == failed_condition
                assert row['status'] == ('COMPILE_FAILED' if failed else 'PASS')
                assert row['flash_used_bytes'] == ('' if failed else '20000')
                assert row['static_ram_used_bytes'] == ('' if failed else '2000')
                assert (temp / 'test_results' / row['log_file']).is_file()
    report = {'scope': 'Parser fixtures and temporary mock CLI calls only; all temporary results deleted. No Arduino measurements.',
              'json_english_japanese_memory_parsers': 'PASS', 'invalid_or_wrong_board_sizes_rejected': 'PASS',
              'all8_condition_source_selection': 'PASS', 'compile_failure_leaves_numbers_blank': 'PASS',
              'never_uploads_or_installs': True, 'original_files_unchanged': True,
              'existing_results_not_overwritten': True}
    (HERE / 'memory_builds_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: memory parsers and8-condition automation with temporary mocks; no Arduino measurements claimed.')


if __name__ == '__main__':
    verify()
