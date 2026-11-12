"""Publish the recorded color charts and labeled data figures; never generate new readings."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_PACKAGE = (ROOT.parent if (ROOT / 'moved_paths.json').is_file() else ROOT) / 'experiment_report_v10'
if ROOT.name == 'source_snapshot':
    DEFAULT_PACKAGE = ROOT.parent
SIZES = (40, 60, 80, 100)
FORMATS = ('DENSE', 'CSR')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def rgb9(path):
    tokens = path.read_text().split()
    assert tokens[:4] == ['P3', '3', '3', '255'] and len(tokens) == 31, path
    numbers = list(map(int, tokens[4:]))
    assert all(0 <= n <= 255 for n in numbers)
    return [tuple(numbers[i:i + 3]) for i in range(0, 27, 3)]


def assert_chart_pixels(path, pixels):
    with Image.open(path) as image:
        assert image.mode == 'RGB' and image.size == (360, 360), path
        for i, color in enumerate(pixels):
            r, c = divmod(i, 3)
            region = image.crop((c * 120, r * 120, (c + 1) * 120, (r + 1) * 120))
            assert region.getextrema() == tuple((v, v) for v in color), (path, i)


def chart(draw, pixels, x, y, size):
    """Render numeric RGB measurements as nine solid blocks, without smoothing."""
    assert size % 3 == 0
    unit = size // 3
    for i, color in enumerate(pixels):
        r, c = divmod(i, 3)
        draw.rectangle((x + c * unit, y + r * unit,
                        x + (c + 1) * unit - 1, y + (r + 1) * unit - 1), fill=color)
    draw.rectangle((x - 1, y - 1, x + size, y + size), outline='#7b8490', width=1)


def font(size):
    candidates = [os.environ.get('COLOR_IMAGE_FONT', ''),
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                  '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
                  '/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc']
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    raise FileNotFoundError('Japanese font not found. Set COLOR_IMAGE_FONT to a Japanese TTF/TTC font path.')


def build_images(package=DEFAULT_PACKAGE):
    package = Path(package).resolve()
    output = package / 'color_images'
    output.mkdir(exist_ok=True)
    target = rgb9(package / 'reference/result.ppm')
    recorded = rows(package / 'data/physical10_mse_all80.csv')
    means = rows(package / 'data/physical10_mse_summary.csv')
    assert len(recorded) == 80 and len(means) == 8
    index = {}; metadata = []
    for capture in range(1, 11):
        name = f'capture_{capture:02d}'
        source = package / 'raw_logs/physical10' / name
        destination = output / name
        destination.mkdir(exist_ok=True)
        for h in SIZES:
            for fmt in FORMATS:
                stem = f'h{h}_{fmt.lower()}'
                ppm = source / f'{stem}.ppm'; png = source / f'{stem}.png'
                pixels = rgb9(ppm)
                mse = sum((v - t) ** 2 for color, correct in zip(pixels, target) for v, t in zip(color, correct)) / 27
                record = next(r for r in recorded if int(r['capture_counter']) == capture and int(r['H']) == h and r['format'] == fmt)
                assert abs(mse - float(record['mse_from_received_ppm'])) < 1e-9
                assert_chart_pixels(png, pixels)
                for path in (ppm, png):
                    shutil.copyfile(path, destination / path.name)
                    assert sha(path) == sha(destination / path.name)
                item = dict(capture=capture, H=h, format=fmt, mse_of_this_image=mse,
                            png=f'{name}/{stem}.png', ppm=f'{name}/{stem}.ppm',
                            source_ppm=ppm.relative_to(package).as_posix(), source_ppm_sha256=sha(ppm),
                            source_png_sha256=sha(png))
                metadata.append(item); index[capture, h, fmt] = dict(item, pixels=pixels)
    for row in means:
        values = [index[i, int(row['H']), row['format']]['mse_of_this_image'] for i in range(1, 11)]
        assert abs(statistics.mean(values) - float(row['mean_mse'])) < 1e-9

    reference = Image.new('RGB', (360, 360), 'white')
    # Reference-only chart has no labels or border inside its RGB pixels.
    draw = ImageDraw.Draw(reference)
    for i, color in enumerate(target):
        r, c = divmod(i, 3)
        draw.rectangle((c * 120, r * 120, (c + 1) * 120 - 1, (r + 1) * 120 - 1), fill=color)
    reference.save(output / 'reference.png')
    assert_chart_pixels(output / 'reference.png', target)

    canvas = Image.new('RGB', (1180, 1790), 'white'); draw = ImageDraw.Draw(canvas)
    draw.text((590, 25), '10回目の実測：次元別の補正画像', font=font(35), fill='#182433', anchor='mt')
    draw.text((590, 78), '表示MSEは各画像の値です（10回平均とは別）', font=font(23), fill='#394b60', anchor='mt')
    for x, label in [(290, '先生指定の正解'), (640, '通常方式\n（Dense）'), (990, '圧縮方式\n（枝刈り＋CSR）')]:
        draw.multiline_text((x, 123), label, font=font(25), fill='#182433', anchor='ma', align='center', spacing=4)
    for i, h in enumerate(SIZES):
        y = 215 + i * 380
        draw.text((70, y + 145), f'H={h}', font=font(29), fill='#182433', anchor='mm')
        chart(draw, target, 140, y, 300)
        draw.text((290, y + 314), '比較の基準', font=font(24), fill='#394b60', anchor='mt')
        for fmt, x in [('DENSE', 490), ('CSR', 840)]:
            item = index[10, h, fmt]
            chart(draw, item['pixels'], x, y, 300)
            draw.text((x + 150, y + 314), f"MSE = {item['mse_of_this_image']:.2f}", font=font(26), fill='#182433', anchor='mt')
    draw.text((590, 1740), '同じ9色の新規読取入力を全8条件で共有／正解は元のresult.ppm', font=font(23), fill='#394b60', anchor='mt')
    canvas.save(output / 'comparison_capture_10.png')

    for h in SIZES:
        canvas = Image.new('RGB', (1900, 650), 'white'); draw = ImageDraw.Draw(canvas)
        draw.text((950, 20), f'H={h}：全10回の色画像', font=font(32), fill='#182433', anchor='mt')
        for capture in range(1, 11):
            x = 180 + (capture - 1) * 170
            draw.text((x + 75, 90), f'{capture}回目', font=font(22), fill='#394b60', anchor='mt')
        for fmt, y, label in [('DENSE', 145, '通常方式\n（Dense）'), ('CSR', 385, '圧縮方式\n（CSR）')]:
            draw.multiline_text((85, y + 48), label, font=font(23), fill='#182433', anchor='ma', align='center')
            for capture in range(1, 11):
                item = index[capture, h, fmt]; x = 180 + (capture - 1) * 170
                chart(draw, item['pixels'], x, y, 150)
                draw.text((x + 75, y + 164), f"MSE {item['mse_of_this_image']:.2f}", font=font(20), fill='#182433', anchor='mt')
        draw.text((950, 602), '各列は同じ回の入力。各画像のMSEを表示し、10回平均はREADMEに別記。全10回を掲載。', font=font(22), fill='#394b60', anchor='mt')
        canvas.save(output / f'h{h}_all10.png')

    with (output / 'images.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(metadata)
    latest = []
    average = []
    for h in SIZES:
        d, c = index[10, h, 'DENSE'], index[10, h, 'CSR']
        latest.append(f"| {h} | {d['mse_of_this_image']:.2f}：[PNG]({d['png']})／[PPM]({d['ppm']}) | {c['mse_of_this_image']:.2f}：[PNG]({c['png']})／[PPM]({c['ppm']}) |")
        d = next(r for r in means if int(r['H']) == h and r['format'] == 'DENSE')
        c = next(r for r in means if int(r['H']) == h and r['format'] == 'CSR')
        average.append(f"| {h} | {float(d['mean_mse']):.2f} | {float(c['mean_mse']):.2f} | [全10回の画像](h{h}_all10.png) |")
    (output / 'README.md').write_text('''# 各次元の色画像とMSE

[最新版の案内へ戻る](../../README.md)

実際に紙を読んだ10回の出力画像です。元のPPMとPNGをそのまま複製し、色を変更していません。
4種類の次元×通常方式／枝刈り＋CSR×10回＝80枚を収録しています。

## 最後に測った10回目：8条件の比較

10回目は最後の測定回として掲載しています。最もよい回を選んだものではありません。
以下のMSEは**各画像1枚の値**で、10回平均とは異なります。
PNGは色を見る用途、PPMは先生の計算サイトで元のresult.ppmと比べる用途に使えます。

![10回目の比較](comparison_capture_10.png)

| 次元H | 通常方式（Dense） | 枝刈り＋CSR |
| --- | --- | --- |
''' + '\n'.join(latest) + '''

正解は [先生指定のresult.ppm](../reference/result.ppm) です。[正解のPNG](reference.png) も用意しました。
9色のRGB・計27成分の二乗誤差を平均し、元の受領PPMから全80個を再計算して確認しました。

## 全10回の画像と、10回平均MSE

次の数値は各条件の**10回平均**です。平均値に対応する1枚の実測画像を作ったわけではありません。
画像のRGBを平均しても、各回のMSEの平均と同じ意味にはならないため、全回の画像を掲載しています。

| 次元H | Dense平均MSE | CSR平均MSE | 各回の画像とMSE |
| --- | ---: | ---: | --- |
''' + '\n'.join(average) + '''

## 1枚ずつ開く

''' + '\n'.join(f'- [{i}回目](capture_{i:02d})' for i in range(1, 11)) + '''

各フォルダのh40_dense.png、h40_csr.pngなどが個別画像です。同名PPMもあります。
全画像の対応・各回MSE・元ファイルのSHA256は [images.csv](images.csv)、検算結果は [verification.json](verification.json) へ保存しています。
これは保存済みの実測結果の画像化で、新しい測定や再学習ではありません。
''')
    verification = dict(created_date_japan='2026-11-13', actual_physical_captures=10, output_chart_images=80,
        reference_sha256=sha(package / 'reference/result.ppm'), raw_PPM_and_PNG_copies='BYTE_IDENTICAL_TO_RECEIVED_RECORDS',
        all80_PNG_rgb_blocks_match_PPM='PASS', all80_image_MSE_match_recorded_metrics='PASS',
        all8_ten_read_means_match_existing_report='PASS', last_read_displayed=10,
        composite_chart_swatches='Numeric RGB values rendered as solid blocks; labels outside charts.',
        no_new_measurement=True, no_retraining=True, mean_MSE_not_presented_as_single_image_MSE=True)
    (output / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    print('PASS:80 byte-identical PNG/PPM copies,exact RGB blocks/MSE,latest8 comparison and4 all10 galleries; no new capture.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=DEFAULT_PACKAGE)
    build_images(parser.parse_args().package)
