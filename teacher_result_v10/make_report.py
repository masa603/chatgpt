"""Make a report figure and Japanese results table from the verified device run."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    summary=json.loads((HERE/'hardware/run_01/summary.json').read_text())
    pairs=summary['pairs'];memory=summary['memory'];figures=HERE/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(13,4.4));x=np.arange(4);width=.35
    configs=[('MSE on archived 9 inputs','MSE (RGB 0-255)',[r['dense_mse'] for r in pairs],[r['csr_mse'] for r in pairs],None),
        ('Numeric model arrays','KiB',[r['dense_model_bytes']/1024 for r in pairs],[r['csr_model_bytes']/1024 for r in pairs],None),
        ('UNO R4 inference time','Milliseconds / inference',[r['dense_mean_us']/1000 for r in pairs],[r['csr_mean_us']/1000 for r in pairs],
            ([r['dense_group_sample_stddev_us']/1000 for r in pairs],[r['csr_group_sample_stddev_us']/1000 for r in pairs]))]
    for ax,(title,ylabel,dense,csr,errors) in zip(axes,configs):
        ax.bar(x-width/2,dense,width,label='Dense',color='#3569a5',yerr=None if errors is None else errors[0],capsize=3)
        ax.bar(x+width/2,csr,width,label='Pruning + CSR',color='#da843b',yerr=None if errors is None else errors[1],capsize=3)
        ax.set_xticks(x,[str(r['H']) for r in pairs]);ax.set_xlabel('Hidden size H');ax.set_ylabel(ylabel);ax.set_title(title)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    axes[0].axhline(300,color='#a52c36',ls='--',lw=1.2,label='User goal: 300')
    axes[0].legend(fontsize=8,loc='upper left');axes[0].set_ylim(0,540)
    axes[1].legend(fontsize=8,loc='upper left')
    fig.suptitle('V10: teacher-required result.ppm; same saved physical inputs for all eight models',fontsize=12,y=.99)
    fig.text(.5,.025,'Device replay, not a new sensor capture. Timing: 100 calls x 5 groups; error bars = sample SD of group means.\nModel arrays include biases and CSR indices; they are not whole-firmware Flash/RAM.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.12,1,.95))
    fig.savefig(figures/'device_comparison.png',dpi=200)
    fig.savefig(figures/'device_comparison.pdf')
    plt.close(fig)
    rows='\n'.join(f"| {r['H']} | {r['dense_mse']:.2f} | {r['csr_mse']:.2f} | {r['dense_model_bytes']:,}→{r['csr_model_bytes']:,} | {r['model_payload_reduction_percent']:.2f}% | {r['dense_mean_us']:.3f}→{r['csr_mean_us']:.3f} | {-r['csr_time_change_percent']:.2f}% |" for r in pairs)
    stats='\n'.join(f"| {r['H']} | {r['dense_mean_us']:.4f} ± {r['dense_group_sample_stddev_us']:.4f} | {r['csr_mean_us']:.4f} ± {r['csr_group_sample_stddev_us']:.4f} |" for r in pairs)
    report=f'''# V10の実機比較結果

2026年11月9日（日本時間）に受領したArduino UNO R4 WiFiのログを保存・検算しました。
40・60・80・100次元のDense／枝刈り＋CSR、全8条件で自己検査・RGB一致検査がPASSし、4組すべての実機速度を取得しました。
先生指定の元の **result.ppm** を正解とし、CMYK版の基準へ変更していません。

**MSEは保存済みの実測9入力を実機で再生した値です。今回は新しい紙の読取ではありません。**
この9入力はV9・V10の学習とV10の枝刈り閾値選択には加えていません。
ただし同じ学習チャートの既に観察した測定を使う追試で、未見チャートによる独立最終評価ではありません。

## 精度・モデル配列容量・実機速度

| H | Dense MSE | CSR MSE | 配列byte Dense→CSR | 配列削減率 | 推論時間µs Dense→CSR | 時間短縮率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

Denseは全4モデルで300以下、枝刈り＋CSRはH40・60が300以下です。CSRのH80・100は目標未達です。
**全HでCSRのMSEはDenseより増えています。** この保存9入力で「精度劣化5%以内」を満たした条件はありません。
閾値選択に使った別の検証128入力の条件と、今回の9入力での結果を混同しません。
本結果から容量・時間の削減は確認できますが、精度を維持できたとは結論しません。

例えばH60はMSE33.96→273.19（+239.22、約8.04倍）となる代わりに、配列容量60.51%削減・推論時間28.97%短縮です。
H40のCSRはH60のCSRよりMSEが高くなる一方、今回の中で最小の配列容量・最短の推論時間です。
H80・100の目標未達も結果として残し、同じ評価入力のMSEを見て閾値を選び直しません。

![精度・配列容量・実機推論時間](figures/device_comparison.png)

図1　精度・容量・速度の比較。MSEの破線は利用者目標300。速度の誤差棒は5組の平均時間の標本標準偏差です。
[PDF図](figures/device_comparison.pdf) と [集計CSV](device_results.csv) も保存しています。

## 実機速度の条件

入力は同じ保存済み9組を巡回して使用し、10回のウォームアップ後、100推論を5組計時しました。
Dense先行／CSR先行を交互に切り替え、各組の出力チェックサムが凍結済みモデルのPC計算と整合することを確認しました。
計時に推論と共通のチェックサム加算を含め、センサ読取・シリアル出力・MSE計算を含めません。
CPU48MHz、コンパイラ `{summary['compiler']}` です。

| H | Dense 平均±標本SD（µs/推論） | CSR 平均±標本SD（µs/推論） |
| --- | ---: | ---: |
{stats}

このSDは100推論の平均を5組比べた値です。5回の独立した紙の測定でも、個々の推論時間のばらつきでもありません。
旧33入力モデルの実機時間とは異なるモデル・入力なので、旧版の値を今回へ流用していません。

## Flash・RAM

今回のteacher_compare_v10は全8条件入りの比較用プログラムです。

| 項目 | Arduino IDEの表示 |
| --- | ---: |
| スケッチのFlash | {memory['sketch_flash_bytes']:,} / {memory['flash_capacity_bytes']:,} byte（IDE表示{memory['ide_flash_percent']}%） |
| グローバル変数の静的RAM | {memory['static_global_ram_bytes']:,} / {memory['ram_capacity_bytes']:,} byte（IDE表示{memory['ide_static_ram_percent']}%） |
| 静的割当後に残るRAM | {memory['remaining_ram_after_static_bytes']:,} byte |

上のFlash/RAMはプログラム全体の値です。個々のDense/CSRモデル単独の使用量ではありません。
残るRAM26,028byteは、ローカル変数が実際に26,028byte使ったという意味ではありません。実行時のスタック使用量は未測定です。
書込ツールの187,896byteは別の値として保存し、スケッチFlash187,888byteと区別します。

## 元データと検算

[受領したシリアルログ](hardware/run_01/submitted_log.txt)、[表示用のバックスラッシュを除いたログ](hardware/run_01/normalized_log.txt)、
[IDE・書込ログ](hardware/run_01/compile_submitted.txt)、[全検算結果](hardware/run_01/summary.json) を保存しました。
重み・学習データ・元のresult.ppm・保存入力のSHA256を照合し、全8PPMの216成分は凍結済みPC出力と完全一致しました。
全8出力のMSEは先生から受領した元のHTMLでも一致しました。速度の平均・標本SD・短縮率を各組の整数µsから再計算しました。
`record_device_run.py` は保存ログから同じ検算・集計を再現します。`make_report.py` がこの表と図を作ります。

各条件のPPM・PNG、[9色RGB一覧](hardware/run_01/rgb9.csv)、[MSE一覧](hardware/run_01/mse.csv)、
[各組の計時値](hardware/run_01/timing_groups.csv) は [run_01](hardware/run_01) にあります。

## 計画から残る項目

新V10の実機再生・速度・全8条件入りFlash/RAMは記録できました。
当初計画の独立した最終評価チャート、10回の物理読取、H50、各条件単独のFlash/RAMは未実施です。
学習全135行の重みなしMSEが300以下だったという結果でもありません。学習全体の値は [学習結果](results.csv) に残しています。
[進捗と制約](remaining_experiments.md) を含めて報告し、当初計画の全項目完了とはしません。
'''
    (HERE/'RESULTS.md').write_text(report)
    print('Wrote verified-device results table and standalone PNG/PDF figure; no model, threshold or input changed.')


if __name__=='__main__':main()
