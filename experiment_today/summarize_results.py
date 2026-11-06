"""Summarize received measurements, preserving acquisition exclusions and missing data."""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/chatgpt-experiment-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/chatgpt-experiment-cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main(render_plot=True):
    timing = {int(r["hidden_size"]): r for r in read_csv(HERE / "device_results.csv")}
    fresh = {(int(r["hidden_size"]), r["format"]): r for r in read_csv(HERE / "fresh_readings/mse_summary.csv")}
    captures = [json.loads(path.read_text()) for path in sorted((HERE / "fresh_readings").glob("capture_*/summary.json"))]
    counters = [r["capture_counter"] for r in captures]
    exclusions = json.loads((HERE / "fresh_readings/exclusions.json").read_text())
    assert counters == exclusions["valid_received_capture_counters"]
    rows = []
    for h in (40, 60, 100):
        t, d, c = timing[h], fresh[h, "DENSE"], fresh[h, "CSR"]
        assert int(d["physical_captures"]) == int(c["physical_captures"]) == len(captures)
        dense_time, csr_time = float(t["dense_mean_us"]), float(t["csr_mean_us"])
        dense_bytes, csr_bytes = int(t["dense_model_bytes"]), int(t["csr_model_bytes"])
        rows.append({"hidden_size": h, "dense_model_bytes": dense_bytes, "csr_model_bytes": csr_bytes,
                     "model_bytes_reduction_percent": (1 - csr_bytes / dense_bytes) * 100,
                     "dense_mean_us": dense_time, "csr_mean_us": csr_time,
                     "dense_sample_stddev_us": float(t["dense_sample_stddev_us"]),
                     "csr_sample_stddev_us": float(t["csr_sample_stddev_us"]),
                     "dense_over_csr_speed_ratio": dense_time / csr_time,
                     "fresh_valid_capture_counters": ";".join(map(str, counters)),
                     "fresh_physical_replicates": len(captures), "fresh_chart_role": "learning_3x3_repeat",
                     "dense_fresh_mse_mean": float(d["mean_mse"]), "dense_fresh_mse_sample_sd": float(d["sample_stddev_mse"]),
                     "csr_fresh_mse_mean": float(c["mean_mse"]), "csr_fresh_mse_sample_sd": float(c["sample_stddev_mse"]),
                     "fresh_mse_mean_change_percent": (float(c["mean_mse"]) / float(d["mean_mse"]) - 1) * 100,
                     "independent_final_evaluation": "NOT_PERFORMED"})
    with (HERE / "results_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    capacity_table = "\n".join(f"| {r['hidden_size']} | {r['dense_model_bytes']:,}→{r['csr_model_bytes']:,} | {r['model_bytes_reduction_percent']:.2f}% | {r['dense_mean_us']:.3f}→{r['csr_mean_us']:.3f} | {r['dense_over_csr_speed_ratio']:.3f}倍 |" for r in rows)
    mse_table = "\n".join(f"| {r['hidden_size']} | {r['dense_fresh_mse_mean']:.2f} ± {r['dense_fresh_mse_sample_sd']:.2f} | {r['csr_fresh_mse_mean']:.2f} ± {r['csr_fresh_mse_sample_sd']:.2f} | +{r['fresh_mse_mean_change_percent']:.2f}% |" for r in rows)
    memory_path = HERE / "firmware_memory.csv"
    memory_text = "ビルド全体の使用量は未受領で、ピークRAMは測定していない。"
    if memory_path.exists():
        memory_rows = read_csv(memory_path)
        table = "\n".join(f"| {m['sketch_name']} | {int(m['flash_used_bytes']):,} / {int(m['flash_max_bytes']):,} | {int(m['static_ram_used_bytes']):,} / {int(m['ram_max_bytes']):,} | {int(m['ram_remaining_for_stack_and_locals_bytes']):,} |" for m in memory_rows)
        memory_text = """全6モデル入りの読取スケッチについて、Arduino IDEのコンパイル結果を受領した。

| スケッチ | Flash 使用量 / 上限 [byte] | 静的RAM 使用量 / 上限 [byte] | スタック等に残る領域 [byte] |
| --- | ---: | ---: | ---: |
""" + table + """

この値は全6モデルを同時に含むスケッチのビルド全体であり、各方式単独の最小使用量ではない。
RAMの使用量はグローバル変数等の静的領域で、ピークRAMは未測定である。
40・60・100各Hの2モデル比較スケッチ単独のFlash・RAMは未記録のまま区別する。
各方式を1つずつ含むビルドの測定用コードは [memory_probe](memory_probe/README.md) に用意した。
その6条件のArduino IDEのビルド使用量は未受領で、現時点の表には含めていない。
原文と構造化データは [firmware_memory.csv](firmware_memory.csv) とhardwareフォルダに保存した。
"""
    report = f"""# 受領した実験結果：2026年11月6日

40・60・100次元について、Arduino UNO R4 WiFiでの計算確認と推論時間、モデル配列容量、
学習用3×3を新しく読み直した精度を記録した。
速度は保存済みの入力で測定し、再読取MSEは今回新しくセンサで取得した入力から計算した。

## データ数と除外

有効な紙の読取はCAPTURE={', '.join(map(str, counters))}の{len(captures)}回分、各9色である。
各回で同じ新しい入力を6モデルへ与え、6枚の画像を作った。各条件の物理反復数は{len(captures)}である。
CAPTURE=2は、利用者が違う色を読んだと申告したため操作ミスとして除外した。
除外したログそのものは未受領で、9色を完了したかどうかは未確認である。
番号を繰り上げず、理由を [exclusions.json](fresh_readings/exclusions.json) に残した。

## モデル容量と速度

| H | 配列容量 Dense→CSR [byte] | 容量削減 | 平均時間 Dense→CSR [μs] | Dense/CSR速度比 |
| --- | ---: | ---: | ---: | ---: |
{capacity_table}

10回ウォームアップ後、同じ入力列で100推論を5組測定した。
Dense先行・CSR先行を交互にし、時間にはセンサ読取、待機、MSE計算、シリアル出力を含めない。
5組の生の合計時間、平均、標本標準偏差を原文とJSONに保存した。
CSR索引と固定128入力のDense/CSR出力比較は3次元すべてPASSだった。
配列容量は実配列の合計であり、スケッチ全体のFlash・静的RAM使用量とは異なる。
{memory_text}

## 新たに読み取った学習チャートの精度

MSEは0〜255のRGB整数画像で計算した。表は平均 ± 標本標準偏差（分母n−1）、n={len(captures)}である。
増加率は各方式の平均MSEの比から求めた。これは各回の増加率を単純平均した値ではない。

| H | Dense MSE | CSR MSE | 平均MSEの増加率 |
| --- | ---: | ---: | ---: |
{mse_table}

受領した2回では、全3次元でCSRの平均MSEが増加した。
この学習チャートの再読取結果では、平均MSEの増加5%以内という品質目安を満たさなかった。
一方、閾値を選んだ保存済み検証入力では増加5%以下だった。データの用途を混同しない。
この2回の平均MSEはDense・CSRとも60次元が最小だったが、少ない反復数と学習チャートでの結果であり、
未見チャートで最良の次元だとは判断しない。再読取結果を見て重みや閾値を変更していない。

## 結果から言えることと限界

枝刈り＋CSRによるモデル配列の小型化と推論の高速化を確認した。
今回の新しい読取入力では精度が悪化しており、精度維持まで達成したとは報告しない。
CSRは枝刈り後Denseの参照値と照合しており、保存形式の正しさと枝刈りによる精度変化を分けて扱う。
100次元はDenseとCSRの両方が動作した。モデル配列はconstで、CSRだけが搭載可能だったという結果ではない。

紙は手持ちで、校正値・距離・角度・位置の変動が残る。反復数は2で、有意差や原因の特定は行わない。
参照RGBは元Notebookの保存済み表示から復元した値で、元PNGとの照合は未実施である。
新スキャナの起動時チェック部分のログは未受領だが、受領した新しい全入力についてPC再計算と実機RGB8は完全一致した。
チャートは学習に使った9色なので、独立した最終評価は未実施である。
当初計画のH=50と各条件10回の物理読取は今回の測定に含めていない。

## 保存先

- [容量・速度と再読取MSEのCSV](results_summary.csv)
- [比較図PNG](results_overview.png) ／ [図のPDF](results_overview.pdf)
- [実機の速度ログ](hardware/README.md)
- [新しい紙の読取ログとPPM](fresh_readings/README.md)
- [測定方法の変更記録](protocol_change.json)

![受領した測定結果の比較](results_overview.png)
"""
    (HERE / "RESULTS.md").write_text(report, encoding="utf-8")
    if render_plot:
        plot(rows, captures)
    print("Wrote results_summary.csv and RESULTS.md" + (" and standalone PNG/PDF figure" if render_plot else "") + "; counters", counters)


def plot(rows, captures):
    colors = ["#4869a6", "#dd8f38"]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.5))
    for mode, offset, color in [("dense", -.18, colors[0]), ("csr", .18, colors[1])]:
        label = "Dense FP32" if mode == "dense" else "Prune + CSR"
        axes[0].bar(x + offset, [r[f"{mode}_model_bytes"] for r in rows], .34, color=color, label=label)
        axes[1].bar(x + offset, [r[f"{mode}_mean_us"] for r in rows], .34, color=color)
        means = [r[f"{mode}_fresh_mse_mean"] for r in rows]
        sds = [r[f"{mode}_fresh_mse_sample_sd"] for r in rows]
        axes[2].bar(x + offset, means, .34, color=color, alpha=.55, yerr=sds, capsize=4)
        for i, capture in enumerate(captures):
            values = [next(r["mse_rgb8"] for r in capture["results"] if r["hidden_size"] == h and r["format"] == mode.upper()) for h in (40,60,100)]
            jitter = (i - (len(captures) - 1) / 2) * .055
            axes[2].scatter(x + offset + jitter, values, s=24, color=color, edgecolors="black", linewidths=.5, zorder=4)
    for ax in axes:
        ax.set_xticks(x, ["40", "60", "100"])
        ax.set_xlabel("Hidden units per layer (H)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set(title="Model arrays", ylabel="Bytes (log scale)", yscale="log")
    axes[1].set(title="Device inference time", ylabel="Microseconds / inference (log scale)", yscale="log")
    axes[2].set(title=f"Fresh learning-chart reads (n={len(captures)})", ylabel="RGB8 MSE (mean +/- sample SD)", ylim=(0,None))
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Arduino UNO R4 WiFi: pruning + CSR comparison", fontsize=13)
    fig.text(.03,.015,"Timing: 5 groups x 100 calls, after 10 warmups. Fresh chart: valid captures 1 and 3; capture 2 excluded (wrong patch).\nReference RGB recovered from Notebook display. Learning-chart re-read; not an independent final evaluation. Error bars are SD, not confidence intervals.",fontsize=8)
    fig.tight_layout(rect=(0,.12,1,.94))
    fig.savefig(HERE / "results_overview.png", dpi=180)
    fig.savefig(HERE / "results_overview.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
